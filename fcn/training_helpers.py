import sys
sys.path.insert(0, '..')

import numpy as np
import torch
from scipy import signal

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from comms import *


def gen_training_data(preamble_seq, num_examples = 1024, signal_length = 200, payload=128,
                      snr = 10, normalize=False, add_phase_offset=False, add_channel=False, 
                      sample_rate=1e6, add_carrier_offset=False, max_carrier_offset=10e3):

    # Pre-define array to contain complex-valued waveforms
    waveforms = np.zeros((num_examples,signal_length),dtype=np.complex128)

    # Predefine labels array
    labels = np.zeros((num_examples,),dtype=int)
    
    # Generate random payload bits
    bits = np.random.randint(0,2,(num_examples,payload))

    # Add preambles to payloads
    if preamble_seq is not None:
        packets = np.concatenate((np.tile(preamble_seq,(num_examples,1)), bits), axis=1)
    else:
        packets = bits
        preamble_seq = []

    # Map to BPSK symbols
    packets = np.where(packets < 1, -1+0j, 1+0j)

    # Insert into random offset and save offset as label
    for idx, waveform in enumerate(waveforms):
        
        # Get random time offset
        tau = np.random.randint(0,signal_length-payload-len(preamble_seq))
        
        # Insert packet at offset tau
        waveform[tau:tau+payload+len(preamble_seq)] = packets[idx]
        
        # Our label is the same time offset
        labels[idx] = tau
        
        # Add random phase offset
        if add_phase_offset:
            waveforms[idx] = phase_offset(waveform,offset=np.random.randint(-180,high=181))
        else:
            waveforms[idx] = waveform
            
        # Add frequency offset
        if add_carrier_offset:
            carrier_offset = np.random.randint(0, max_carrier_offset)
            offset_sine = np.exp(1j*2*np.pi*(carrier_offset/sample_rate)*np.arange(signal_length))
            waveforms[idx] = offset_sine*waveforms[idx]
            
        # Add flat fading channel
        if add_channel:
            gains = 1/np.sqrt(2)*(np.random.randn()+1j*np.random.randn())
            waveforms[idx] = gains*waveforms[idx]
    
    # Add noise
    noisy_waveforms = awgn(waveforms,snr)
    
    # normalize
    if normalize:
        noisy_waveforms = (noisy_waveforms/np.max(np.abs(noisy_waveforms),axis=1)[:,None])

    return noisy_waveforms, labels

# This function takes an array of complex waveforms and associated true offset
# indexes, and returns a 4-D training data tensor compatible with pytorch conv2d layers.
def preprocess(data, labels, to_onehot=True, gpu=True):
    
    # Convert labels to pytorch tensors
    if to_onehot:
        labels_oh = np.zeros(data.shape)
        for idx, label in enumerate(labels):
            labels_oh[idx,label] = 1
        labels = torch.FloatTensor(labels_oh)
    else:
        labels = torch.LongTensor(labels)

    # Split into real and imaginary channels
    train_data = torch.FloatTensor(np.expand_dims(np.stack((data.real, data.imag),axis=1),axis=1))

    # Prep dataset for cuda if gpu true
    if gpu:
        train_data = train_data.cuda()
        labels = labels.cuda()
        
    return train_data, labels

# General purpose training loop
def train_model(model, optimizer, loss_fn, train_loader, val_loader, num_epochs=30, verbose=False):

    train_accs, val_accs = [], []
    losses, val_losses = [], []
    
    best_loss = np.inf

    for epoch in range(num_epochs):
        running_loss = 0
        accuracies = torch.zeros((len(train_loader), train_loader.batch_size)).cuda()
        val_accuracies = torch.zeros((len(val_loader), val_loader.batch_size)).cuda()
        
        for i, (x_train, y_train) in enumerate(train_loader):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x_train)
            loss = loss_fn(outputs, y_train)

            loss.backward()
            optimizer.step()

            running_loss = running_loss + loss.item()
            
            with torch.no_grad():
                accuracies[i] = torch.sum(y_train.argmax(axis=1) == outputs.argmax(axis=1))/train_loader.batch_size

        losses.append(running_loss/len(train_loader))
        
        with torch.no_grad():

            running_val_loss = 0

            for j, (x_val, y_val) in enumerate(val_loader):

                # evaluate validation loss
                val_outputs = model(x_val)
                val_loss = loss_fn(val_outputs, y_val)
                
                running_val_loss = running_val_loss + val_loss.item()
                val_accuracies[j] = torch.sum(y_val.argmax(axis=1) == val_outputs.argmax(axis=1))/val_loader.batch_size

            val_losses.append(running_val_loss/len(val_loader))
            
        if val_losses[-1] < best_loss:
            saved_model = model.state_dict()
            best_loss = val_losses[-1]
            if verbose:
                print(f"Validation loss improved {best_loss}, saving model...")
    
        train_accs.append(torch.mean(accuracies).cpu())
        val_accs.append(torch.mean(val_accuracies).cpu())
    
    model.load_state_dict(saved_model)
            
    return model, losses, val_losses, train_accs, val_accs
    
def test_baselines(preamble_seq, snr_range=None, num_iter=1500, payload=128, signal_length=200, random_offset=True, carrier_offset=None):
    
    if snr_range is None:
        snr_range = np.arange(-15,10)
    
    preamble = np.where(preamble_seq < 1, -1+0j, 1+0j)

    corr_ers = np.zeros(len(snr_range),)
    expert_ers = np.zeros(len(snr_range),)
    
    if carrier_offset:
        offset_sine = np.exp(1j*2*np.pi*(carrier_offset/1e6)*np.arange(signal_length))

    for idx, snr in enumerate(snr_range):
        corr_err, expert_err = float(0), float(0)
        for i in range(num_iter):
            # Create new frame
            ph = np.random.randint(-180,high=181)
        
            if random_offset:
                tau = np.random.randint(len(preamble),signal_length-payload-len(preamble))
            else:
                tau = 40

            if carrier_offset:
                my_frame = awgn(phase_offset(create_frame(preamble_seq, payload=payload, signal_length=signal_length, offset=tau), offset=ph)*offset_sine, snr)
            else:
                my_frame = awgn(phase_offset(create_frame(preamble_seq, payload=payload, signal_length=signal_length, offset=tau), offset=ph), snr)
            
            # Calculate baselines
            correlation = np.abs(np.correlate(my_frame, preamble, mode='valid'))
            y_corr = np.argmax(correlation)

            # Get corrective terms
            r_i = np.convolve(np.abs(my_frame.real), np.ones(len(preamble)), 'valid')
            r_q = np.convolve(np.abs(my_frame.imag), np.ones(len(preamble)), 'valid')
            r_iq = np.convolve(np.abs(my_frame.real+my_frame.imag), np.ones(len(preamble)), 'valid')/np.sqrt(2)
            r_i_q = np.convolve(np.abs(my_frame.real-my_frame.imag), np.ones(len(preamble)), 'valid')/np.sqrt(2)

            corr_expert = correlation - np.max((r_i, r_q, r_iq, r_i_q), axis=0)
            y_expert = np.argmax(corr_expert[:-len(preamble_seq)])

            # Calculate if error
            corr_err = corr_err + (y_corr != tau)
            expert_err = expert_err + (y_expert != tau)

        corr_ers[idx] = corr_err
        expert_ers[idx] = expert_err

        
    corr_ders = corr_ers/num_iter
    expert_ders = expert_ers/num_iter
        
    return corr_ders, expert_ders

# Returns DER for given pytorch detector, make sure trained detector matches the
# preamble you are testing against
def test_detector(detector, preamble_seq, snr_range=None, num_iter=1500, payload=128, signal_length=200, add_channel=False, 
             abs_phase=False, tau=None, plot=True, add_phase_offset=True, normalize=True, carrier_offset=None):
    
    if snr_range is None:
        snr_range = np.arange(-10,15)

    nn_ers = np.zeros(len(snr_range),)
    
    if carrier_offset:
        offset_sine = np.exp(1j*2*np.pi*(carrier_offset/1e6)*np.arange(signal_length))
    
    for idx, snr in enumerate(snr_range):
        nn_err = float(0)
        for i in range(num_iter):
            
            # Create new frame
            ph = np.random.randint(-180,high=181)
            tau = np.random.randint(0,signal_length-payload-len(preamble_seq))

            my_frame = create_frame(preamble_seq, payload=payload, signal_length=signal_length, offset=tau)
    
            if add_phase_offset:
                my_frame = phase_offset(my_frame, offset=ph)
                
            if carrier_offset is not None:
                my_frame = my_frame*offset_sine

            if add_channel:
                gains = 1/np.sqrt(2)*(np.random.normal(0)+1j*np.random.normal(0))
                my_frame = gains*my_frame
                
            # Add noise
            my_frame = awgn(my_frame, snr)
            
            # normalize
            if normalize:
                my_frame = my_frame/np.max(np.abs(my_frame))

            # FCN prediction
            new_frame = np.expand_dims(np.vstack((my_frame.real, my_frame.imag)),axis=(0,1))
            new_frame = torch.tensor(new_frame).float()
            nn_output = detector(new_frame)

            y_nn = torch.argmax(nn_output)

            # Calculate if error
            nn_err = nn_err + (y_nn != tau)

        nn_ers[idx] = nn_err

    nn_ders = nn_ers/num_iter
        
    return nn_ders

# This function takes an array of complex waveforms and associated true offset
# indexes, and returns a 4-D training data tensor compatible with pytorch
# conv2d layers.
def preprocess(data, labels, to_onehot=True, gpu=True):
    
    # Convert labels to pytorch tensors
    if to_onehot:
        labels_oh = np.zeros(data.shape)
        for idx, label in enumerate(labels):
            labels_oh[idx,label] = 1
        labels = torch.FloatTensor(labels_oh)
    else:
        labels = torch.LongTensor(labels)

    # Split into real and imaginary channels
    train_data = torch.FloatTensor(np.expand_dims(np.stack((data.real, data.imag),axis=1),axis=1))

    # Prep dataset for cuda if gpu true
    if gpu:
        train_data = train_data.cuda()
        labels = labels.cuda()
        
    return train_data, labels

# Function returns the DER of a binary sequence preamble_seq by using the standard
# correlation under selected impairments.
# Function returns the DER of a binary sequence preamble_seq by using the standard
# correlation under selected impairments.
def calculate_baseline(preamble_seq, snr_range, num_iter=1500, payload=128,
                       signal_length=200, add_phase_offset=True, carrier_offset=None,
                       sample_rate=1e6, add_channel=False):
    
    
    preamble = np.where(preamble_seq < 1, -1+0j, 1+0j)

    corr_ers = np.zeros(len(snr_range),)
    
    if carrier_offset:
        offset_sine = np.exp(1j*2*np.pi*(carrier_offset/sample_rate)*np.arange(signal_length))

    for idx, snr in enumerate(snr_range):
        corr_err = float(0)
        for i in range(num_iter):

            # Create new frame with a random tau
            tau = np.random.randint(0,signal_length-payload-len(preamble_seq))
            my_frame = create_frame(preamble_seq, payload=payload, signal_length=signal_length, offset=tau)
            
            if add_phase_offset:
                ph = np.random.randint(-180,high=181)
                my_frame = phase_offset(my_frame, offset=ph)
                
            if carrier_offset is not None:
                my_frame = my_frame*offset_sine
                
            if add_channel:
                gains = 1/np.sqrt(2)*(np.random.normal(0)+1j*np.random.normal(0))
                my_frame = gains*my_frame
                
            # Add noise
            my_frame = awgn(my_frame, snr)
            
            # Find peaks using correlation
            correlation = np.abs(np.correlate(my_frame, preamble, mode='valid'))
            y_corr = np.argmax(correlation)

            # Calculate if error
            corr_err = corr_err + (y_corr != tau)

        corr_ers[idx] = corr_err
  
    corr_ders = corr_ers/num_iter
        
    return corr_ders