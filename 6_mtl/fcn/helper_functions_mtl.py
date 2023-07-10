import numpy as np
from comms import *
import torch

def gen_training_data_mtl(preamble_seq, num_examples = 1024, signal_length = 200, payload=128, snr = 10, normalize=False, add_phase_offset=False, add_channel=False, 
                      sample_rate=1e6, add_carrier_offset=False, max_freq_offset=35e3):

    # Pre-define array to contain complex-valued waveforms
    waveforms = np.zeros((num_examples,signal_length),dtype=np.complex128)

    # Predefine labels array
    labels = np.zeros((num_examples,),dtype=int)
    labels_cfo = np.zeros((num_examples,),dtype=int)
    
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
            labels_cfo[idx] = carrier_offset
    
    # Add noise
    noisy_waveforms = awgn(waveforms,snr)
    
    # normalize
    if normalize:
        noisy_waveforms = (noisy_waveforms/np.max(np.abs(noisy_waveforms),axis=1)[:,None])

    if add_carrier_offset:
        return noisy_waveforms, labels, labels_cfo
    else:
        return noisy_waveforms, labels
    
def preprocess_mtl(data, labels, labels_cfo, to_onehot=True, gpu=True):
    
    # Convert labels to pytorch tensors
    if to_onehot:
        labels_oh = np.zeros(data.shape)
        for idx, label in enumerate(labels):
            labels_oh[idx,label] = 1
        labels = torch.FloatTensor(labels_oh)
    else:
        labels = torch.LongTensor(labels)
        
    labels_cfo = torch.FloatTensor(labels_cfo/35e3)

    # Split into real and imaginary channels
    train_data = torch.FloatTensor(np.expand_dims(np.stack((data.real, data.imag),axis=1),axis=1))

    # Prep dataset for cuda if gpu true
    if gpu:
        train_data = train_data.cuda()
        labels = labels.cuda()
        labels_cfo = labels_cfo.cuda()
        
    return train_data, labels, labels_cfo


# Returns DER for given pytorch detector, make sure trained detector matches the
# preamble you are testing against
def test_detector_mtl(detector, preamble_seq, snr_range=None, num_iter=1500, payload=128, signal_length=200, add_channel=False,
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
            nn_output, _ = detector(new_frame)

            y_nn = torch.argmax(nn_output)

            # Calculate if error
            nn_err = nn_err + (y_nn != tau)

        nn_ers[idx] = nn_err

    nn_ders = nn_ers/num_iter
        
    return nn_ders

