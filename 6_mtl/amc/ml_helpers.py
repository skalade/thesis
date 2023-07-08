import numpy as np
import matplotlib.pyplot as plt

from comms_helpers import *

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

def gen_tensor_data(mod_scheme, num_frames=32, samples_per_frame=128, sps=5, snr=30):
            
    symbols_required = int(np.ceil(samples_per_frame/sps))*num_frames
        
    # Mod scheme has to be one of: 'BPSK', 'QPSK', '16-QAM'
    if mod_scheme == 'BPSK':
        symbols = pulse_shape(generate_bpsk(symbols_required), sps=sps)[:num_frames*samples_per_frame]
    elif mod_scheme == 'QPSK':
        symbols = pulse_shape(generate_qpsk(symbols_required), sps=sps)[:num_frames*samples_per_frame]
    elif mod_scheme == '8-PSK':
        symbols = pulse_shape(generate_psk8(symbols_required), sps=sps)[:num_frames*samples_per_frame]
    elif mod_scheme == '16-QAM':
        symbols = pulse_shape(generate_qam(symbols_required), sps=sps)[:num_frames*samples_per_frame]
    elif mod_scheme == '4-ASK':
        symbols = pulse_shape(generate_ask4(symbols_required), sps=sps)[:num_frames*samples_per_frame]

    # Add noise and split into frames
    frames = awgn(symbols.reshape(num_frames,-1), snr)

    # Normalize to unit energy per frame
    for i, frame in enumerate(frames):
        power = np.mean((np.abs(frame)))
        frames[i] = frame / power

    # Split into I/Q, add extra channel to make a 4-D tensor
    return torch.FloatTensor(np.stack((frames.real, frames.imag),axis=1),axis=1)

def gen_data_from_list(mod_scheme, snr_range, num_frames=32, samples_per_frame=128):
    
    # total dataset size
    frames = torch.zeros((num_frames*len(snr_range), 2, samples_per_frame), dtype=torch.float)
    
    # snrs dataset for multitask
    snrs = torch.zeros(num_frames*len(snr_range), dtype=torch.float)
    
    for i, snr in enumerate(snr_range):
        frames[i*num_frames:(i+1)*num_frames] = gen_tensor_data(mod_scheme, num_frames=num_frames, samples_per_frame=samples_per_frame, snr=snr)
        snrs[i*num_frames:(i+1)*num_frames] = snr
    
    return frames, snrs

def gen_loader(num_frames=32, samples_per_frame=128, snr=[30], batch_size=32):
    
    bpsk_data, bpsk_snrs = gen_data_from_list('BPSK', snr, num_frames=num_frames, samples_per_frame=samples_per_frame)
    qpsk_data, qpsk_snrs = gen_data_from_list('QPSK', snr, num_frames=num_frames, samples_per_frame=samples_per_frame)
    psk_data, psk_snrs = gen_data_from_list('8-PSK', snr, num_frames=num_frames, samples_per_frame=samples_per_frame)
    qam_data, qam_snrs = gen_data_from_list('16-QAM', snr, num_frames=num_frames, samples_per_frame=samples_per_frame)
    ask_data, ask_snrs = gen_data_from_list('4-ASK', snr, num_frames=num_frames, samples_per_frame=samples_per_frame)
    
    train_data = torch.cat((bpsk_data, qpsk_data, psk_data, qam_data, ask_data))
    
    # Modulation labels
    bpsk_labels = torch.zeros(bpsk_data.shape[0])
    qpsk_labels = torch.ones(qpsk_data.shape[0])
    psk_labels = torch.ones(qam_data.shape[0])*2
    qam_labels = torch.ones(ask_data.shape[0])*3
    ask_labels = torch.ones(ask_data.shape[0])*4

    train_labels = torch.cat((bpsk_labels, qpsk_labels, psk_labels, qam_labels, ask_labels)).long()
    
    # SNR labels
    train_labels_snr = 1/10**(torch.cat((bpsk_snrs, qpsk_snrs, psk_snrs, qam_snrs, ask_snrs))/10) # linear noise as labels
    
    # if gpu
    train_data = train_data.cuda()
    train_labels = train_labels.cuda()
    train_labels_snr = train_labels_snr.cuda()
    
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels, train_labels_snr)
    
    return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def train_model(model, optimizer, train_loader, val_loader, num_epochs=5, verbose=True):

    losses, val_losses = [], []
    
    loss_fn = nn.CrossEntropyLoss()#nn.MSELoss()

    # Main training loop
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0

        best_loss = np.inf

        # Loop over entire training loader
        for x_train, y_train, _ in train_loader:

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x_train)
            loss = loss_fn(outputs, y_train)
            loss.backward()
            optimizer.step()

            # Add to running loss, average later
            running_loss = running_loss + loss.item()

        # Append average loss for this epoch to losses list
        losses.append(running_loss/len(train_loader))

        # Evaluate validation loss

        with torch.no_grad():

            running_val_loss = 0

            for x_val, y_val, _ in val_loader:

                # evaluate validation loss
                val_outputs = model(x_val)
                val_loss = loss_fn(val_outputs, y_val)

                running_val_loss = running_val_loss + val_loss.item()

            val_losses.append(running_val_loss/len(val_loader))

        if val_losses[-1] < best_loss:
            saved_model = model.state_dict()
            best_loss = val_losses[-1]
            
        if verbose:
            print('Epoch {}: {}'.format(epoch,losses[-1]))

    # Load best model
    model.load_state_dict(saved_model)
    
    return model, losses, val_losses

def test_model(model, snr_range):
    accs = []

    correct = 0
    total = 0

    model.eval().cpu()

    with torch.no_grad():
        for snr in snr_range:

            bpsk_data = gen_tensor_data('BPSK', num_frames=512, samples_per_frame=128, snr=snr)
            qpsk_data = gen_tensor_data('QPSK', num_frames=512, samples_per_frame=128, snr=snr)
            psk_data = gen_tensor_data('8-PSK', num_frames=512, samples_per_frame=128, snr=snr)
            qam_data = gen_tensor_data('16-QAM', num_frames=512, samples_per_frame=128, snr=snr)
            ask_data = gen_tensor_data('4-ASK', num_frames=512, samples_per_frame=128, snr=snr)

            test_data = torch.cat((bpsk_data, qpsk_data, psk_data, qam_data, ask_data))

            bpsk_labels = torch.zeros(bpsk_data.shape[0])
            qpsk_labels = torch.ones(qpsk_data.shape[0])
            psk_labels = torch.ones(qam_data.shape[0])*2
            qam_labels = torch.ones(qam_data.shape[0])*3
            ask_labels = torch.ones(ask_data.shape[0])*4

            test_labels = torch.cat((bpsk_labels, qpsk_labels, psk_labels, qam_labels, ask_labels))

            results = torch.argmax(model(test_data),axis=1)
            accs.append(torch.sum(results == test_labels).float() / test_data.shape[0])
            
    return accs

def train_model_mtl(model, optimizer, train_loader, val_loader, num_epochs=5, verbose=True):

    losses, val_losses = [], []
    losses_mod, val_losses_mod = [], []
    losses_snr, val_losses_snr = [], []
    
    loss_class = nn.CrossEntropyLoss()#nn.MSELoss()
    loss_snr = nn.MSELoss()

    # Main training loop
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0

        best_loss = np.inf

        # Loop over entire training loader
        for x_train, y_train, z_train in train_loader:

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_hat, n_estim = model(x_train)
            loss_1 = loss_class(y_hat, y_train)
            loss_2 = loss_snr(n_estim, z_train)
            
            loss = loss_1 + loss_2
            
            loss.backward()
            optimizer.step()
            
            losses_mod.append(loss_1.item())
            losses_snr.append(loss_2.item())

            # Add to running loss, average later
            running_loss = running_loss + loss.item()

        # Append average loss for this epoch to losses list
        losses.append(running_loss/len(train_loader))

        # Evaluate validation loss

        with torch.no_grad():

            running_val_loss = 0

            for x_val, y_val, z_val in val_loader:

                # evaluate validation loss
                val_outputs, val_snr = model(x_val)
                val_loss1 = loss_class(val_outputs, y_val)
                
                val_loss2 = loss_snr(val_snr, z_val)

                running_val_loss = running_val_loss + val_loss1.item()
                
                val_losses_mod.append(val_loss1.item())
                val_losses_snr.append(val_loss2.item())

            val_losses.append(running_val_loss/len(val_loader))

        if val_losses[-1] < best_loss:
            saved_model = model.state_dict()
            best_loss = val_losses[-1]
            
        if verbose:
            print('Epoch {}: {}'.format(epoch,losses[-1]))

    # Load best model
    model.load_state_dict(saved_model)
    
    return model, losses, val_losses, losses_mod, losses_snr, val_losses_mod, val_losses_snr

def test_model_mtl(model, snr_range, num_frames=128):
    accs = []
    snr_errs = []

    model.eval().cpu()

    with torch.no_grad():
        for snr in snr_range:

            bpsk_data = gen_tensor_data('BPSK', num_frames=512, samples_per_frame=128, snr=snr)
            qpsk_data = gen_tensor_data('QPSK', num_frames=512, samples_per_frame=128, snr=snr)
            psk_data = gen_tensor_data('8-PSK', num_frames=512, samples_per_frame=128, snr=snr)
            qam_data = gen_tensor_data('16-QAM', num_frames=512, samples_per_frame=128, snr=snr)
            ask_data = gen_tensor_data('4-ASK', num_frames=512, samples_per_frame=128, snr=snr)

            test_data = torch.cat((bpsk_data, qpsk_data, psk_data, qam_data, ask_data))

            bpsk_labels = torch.zeros(bpsk_data.shape[0])
            qpsk_labels = torch.ones(qpsk_data.shape[0])
            psk_labels = torch.ones(qam_data.shape[0])*2
            qam_labels = torch.ones(qam_data.shape[0])*3
            ask_labels = torch.ones(ask_data.shape[0])*4

            test_labels = torch.cat((bpsk_labels, qpsk_labels, psk_labels, qam_labels, ask_labels))

            y_hat, snr_hat = model(test_data)
            
            results = torch.argmax(y_hat,axis=1)
            
            snr_errs.append(F.mse_loss(snr_hat, 1/10**(torch.ones_like(snr_hat)*snr/10)))
            
            accs.append(torch.sum(results == test_labels).float() / test_data.shape[0])
            
    return accs, snr_errs

