import numpy as np

# add noise to signal
def awgn(signal,SNR,measured=False):
    
    if measured:
        # Measure signal power 
        s_p = np.mean(abs(signal)**2)
    else:
        s_p = 1
    
    # Calculate noise power 
    n_p = s_p/(10 **(SNR/10))
    
    # Generate complex noise
    noise = np.sqrt(n_p/2)*(np.random.randn(*signal.shape) + \
                                np.random.randn(*signal.shape)*1j)
    
    # Add signal and noise 
    signal_noisy = signal + noise 
    
    return signal_noisy 

# add phase offset in degrees
def phase_offset(x, offset):
    
    # Convert to polar form so we can add degrees
    mag, ang = np.abs(x), np.angle(x, deg=True)
    
    return mag * np.exp( 1j * ((ang + offset)*np.pi/180) )

# make one frame of data with preamble attached with random time offset
def create_frame(preamble_seq, payload=128, signal_length=200, offset=40):
    waveform = np.zeros((signal_length,),dtype=np.complex128)
    bits = np.random.randint(0,2,payload)
    packet = np.concatenate((preamble_seq,bits))
    packet = np.where(packet < 1, -1+0j, 1+0j)
    waveform[offset:offset+payload+len(preamble_seq)] = packet
    
    return waveform
