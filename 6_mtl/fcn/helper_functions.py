import numpy as np
from comms import *
import torch

# This function takes an array of complex waveforms and associated true offset
# indexes, and returns a 4-D training data tensor compatible with pytorch
# conv2d layers.
def preprocess(data, labels, to_onehot=True, abs_phase=False, gpu=True):
    
    # Convert labels to pytorch tensors
    if to_onehot:
        labels_oh = np.zeros(data.shape)
        for idx, label in enumerate(labels):
            labels_oh[idx,label] = 1
        labels = torch.FloatTensor(labels_oh)
    else:
        labels = torch.LongTensor(labels)

    if abs_phase:
         # Get abs and phase
        train_data = torch.FloatTensor(np.expand_dims(np.stack((data.real, data.imag, np.abs(data)),axis=1),axis=1))
    else:
        # Split into real and imaginary channels
        train_data = torch.FloatTensor(np.expand_dims(np.stack((data.real, data.imag),axis=1),axis=1))

    # Prep dataset for cuda if gpu true
    if gpu:
        train_data = train_data.cuda()
        labels = labels.cuda()
        
    return train_data, labels



# Returns DER for given pytorch detector, make sure trained detector matches the
# preamble you are testing against
def test_detector(detector, preamble_seq, snr_range=None, num_iter=1500, payload=128, signal_length=200, add_channel=False,
                  add_phase_offset=True, normalize=True, carrier_offset=None, sample_rate=1e6):
    
    if snr_range is None:
        snr_range = np.arange(-10,15)

    nn_ers = np.zeros(len(snr_range),)
    
    if carrier_offset:
        offset_sine = np.exp(1j*2*np.pi*(carrier_offset/sample_rate)*np.arange(signal_length))
    
    for idx, snr in enumerate(snr_range):
        nn_err = float(0)
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

# Function returns the DER of a binary sequence preamble_seq by using the standard
# correlation under selected impairments.
def calculate_baseline(preamble_seq, snr_range=None, num_iter=1500, payload=128,
                       signal_length=200, add_phase_offset=True, carrier_offset=None,
                       sample_rate=1e6, add_channel=False):
    
    if snr_range is None:
        snr_range = np.arange(-15,10)
    
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

# Data generation function. Takes binary preamble sequence as an input and 
# outputs an array of waveforms with random BPSK modulated packets.
def gen_training_data(preamble_seq, num_examples = 1024, signal_length = 200, payload=128, snr = 10, normalize=False, add_phase_offset=False, add_channel=False, 
                      clean_outputs=False, sample_rate=1e6, add_carrier_offset=False, max_freq_offset=10e3):

    # Pre-define array to contain complex-valued waveforms
    waveforms = np.zeros((num_examples,signal_length),dtype=np.complex128)

    # Predefine labels array
    labels = np.zeros((num_examples,),dtype=np.int)
    
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

        # Add flat fading channel
        if add_channel:
            gains = 1/np.sqrt(2)*(np.random.randn()+1j*np.random.randn())
            waveforms[idx] = gains*waveforms[idx]
            
        # Add frequency offset
        if add_carrier_offset:
            carrier_offset = np.random.randint(0, max_freq_offset)
            offset_sine = np.exp(1j*2*np.pi*(carrier_offset/sample_rate)*np.arange(signal_length))
            waveforms[idx] = offset_sine*waveforms[idx]
    
    # Add noise
    noisy_waveforms = awgn(waveforms,snr)
    
    # normalize
    if normalize:
        noisy_waveforms = (noisy_waveforms/np.max(np.abs(noisy_waveforms),axis=1)[:,None])

    if clean_outputs:
        return noisy_waveforms, labels, waveforms
    else:
        return noisy_waveforms, labels

# Main training loop
def train_network(detector, optimizer, scheduler, loss_fn, training_data, batch_size = 32, num_epochs=100, validation_data=None):
    
    # Unpack training data
    train_data, train_labels = training_data
    
    # Create dataset and loader
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Lists to hold losses per epoch
    losses = []
    
    # If exists, unpack validation data
    if validation_data:
        val_data, val_labels = validation_data
        val_losses = []

    # Main training loop
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0

        best_loss = np.inf

        # Loop over entire training loader
        for x_train, y_train in train_loader:

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = detector(x_train)
            loss = loss_fn(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Add to running loss, average later
            running_loss = running_loss + loss.item()

        # Append average loss for this epoch to losses list
        losses.append(running_loss/len(train_loader))

        # Optionally evaluate validation loss
        if validation_data:

            with torch.no_grad():
                # evaluate validation loss
                val_outputs = detector(val_data)
                val_loss = loss_fn(val_outputs, val_labels)

            val_losses.append(val_loss.item())

            if val_losses[-1] < best_loss:
                saved_model = detector.state_dict()
                best_loss = val_losses[-1]
        else:
            if losses[-1] < best_loss:
                saved_model = detector.state_dict()
                best_loss = losses[-1]
    
        scheduler.step()
#         print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        
    # Load best model
    detector.load_state_dict(saved_model)
    
    if validation_data:
        return detector, losses, val_losses
    else:
        return detector, losses