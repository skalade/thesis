import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal
	
# def awgn(signal,SNR):
#     # Measure signal power 
#     s_p = np.mean(abs(signal)**2)
    
#     # Calculate noise power 
#     n_p = s_p/(10 **(SNR/10))
    
#     # Generate complex noise 
#     noise = np.sqrt(n_p/2)*(np.random.randn(len(signal)) + \
#                                     np.random.randn(len(signal))*1j)
    
#     # Add signal and noise 
#     signal_noisy = signal + noise 
    
#     return signal_noisy    

# add noise to signal
def awgn(signal, SNR, measured=False, return_true_snr=False):
    
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
    
    if not return_true_snr:
        return signal_noisy
    else:
        return signal_noisy, s_p/(np.mean(abs(noise)**2))

# Function to generate a block of BPSK, QPSK or 16-QAM symbols
def symbol_gen(nsym,mod_scheme):
    
    if mod_scheme == 'BPSK':
        # 1 bit per symbol for BPSK 
        m = 1  
        M = 2 ** m 
    
        # BPSK symbol values
        bpsk = [-1+0j, 1+0j]
        
        # Generate random integers 
        ints = np.random.randint(0,M,nsym)
        
        # Generate BPSK symbols 
        data = [bpsk[i] for i in ints]
        data = np.array(data,np.complex64)

    elif mod_scheme == 'QPSK': 
        # 2 bits per symbol for QPSK 
        m = 2
        M = 2 ** m 
    
        # QPSK symbol values 
        qpsk = [1+1j, -1+1j, 1-1j, -1-1j] / np.sqrt(2)
        
        # Generate random integers 
        ints = np.random.randint(0,M,nsym)
    
        # Map to QPSK symbols 
        data = [qpsk[i] for i in ints]
        data = np.array(data,np.complex64)
        
    elif mod_scheme == '16-QAM': 
        # 4 bits per symbol for 16-QAM 
        m = 4 
        M = 2 ** m 
        
        # 16-QAM symbol values  
        qam16 = [-3-3j, -3-1j, -3+3j, -3+1j,  \
                -1-3j, -1-1j, -1+3j, -1+1j,  \
                 3-3j,  3-1j,  3+3j,  3+1j,  \
                 1-3j,  1-1j,  1+3j,  1+1j] / np.sqrt(10)
        
        # Generate random integers 
        ints = np.random.randint(0,M,nsym)
        
        # Map to 16-QAM symbols 
        data = [qam16[i] for i in ints]
        data = np.array(data,np.complex64)
        
    else: 
        raise Exception('Modulation method must be BPSK, QPSK or 16-QAM.')
    
    return data 

def calculate_evm(symbols_tx, symbols_rx):
    evm_rms = np.sqrt(np.mean(np.abs(symbols_rx - symbols_tx )**2)) / \
              np.sqrt(np.mean(np.abs(symbols_tx)**2))
    
    return evm_rms*100

# Function to generate BPSK
def generate_bpsk(num_symbols, noise=50):
    bits = np.random.randint(0,2,num_symbols)
    bpsk_scheme = [1+0j, -1+0j]
    bpsk_symbols = np.array([bpsk_scheme[i] for i in bits])
    
    bpsk_symbols = awgn(bpsk_symbols, noise)
    
    return bpsk_symbols

# Function to generate QPSK
def generate_qpsk(num_symbols, noise=50):
    qpsk_scheme= [1+1j, 1-1j, -1+1j, -1-1j]
    ints = np.random.randint(0,4,num_symbols)
    qpsk_symbols = np.array([qpsk_scheme[i] for i in ints])/np.sqrt(2)

    qpsk_symbols = awgn(qpsk_symbols, noise)
    
    return qpsk_symbols

# Function to generate QAM
def generate_qam(num_symbols, noise=50):
    qam_scheme = [-3-3j, -3-1j, -3+3j, -3+1j,  \
                  -1-3j, -1-1j, -1+3j, -1+1j,  \
                   3-3j,  3-1j,  3+3j,  3+1j,  \
                   1-3j,  1-1j,  1+3j,  1+1j]
    ints = np.random.randint(0,16,num_symbols)
    qam_symbols = np.array([qam_scheme[i] for i in ints])
    qam_symbols = qam_symbols/np.mean(np.abs(qam_scheme))
    
    qam_symbols = awgn(qam_symbols, noise)
    
    return qam_symbols

# Function to generate 4-ASK
def generate_ask4(num_symbols, noise=50):
    ask4_scheme = [3+0j, 1+0j, -1+0j, -3+0j]
    ints = np.random.randint(0,4,num_symbols)
    ask4_symbols = np.array([ask4_scheme[i] for i in ints])
    ask4_symbols = ask4_symbols/np.mean(np.abs(ask4_scheme))
    
    ask4_symbols = awgn(ask4_symbols, noise)
    
    return ask4_symbols

def generate_psk8(num_symbols, noise=50):
    psk8_scheme = [ 1+0j, 0.7071+0.7071j, 0+1j, -0.7071+0.7071j, \
                   -1+0j, -0.7071-0.7071j, 0-1j, 0.7071-0.7071j]
    
    ints = np.random.randint(0,8,num_symbols)
    psk8_symbols = np.array([psk8_scheme[i] for i in ints])
    psk8_symbols = psk8_symbols/np.mean(np.abs(psk8_scheme))
    
    psk8_symbols = awgn(psk8_symbols, noise)
    
    return psk8_symbols

def calculate_statistics(x):
    
    # Extract instantaneous amplitude and phase
    inst_a = np.abs(x) 
    inst_p = np.angle(x)

    # Amplitude statistics
    m2_a = np.mean((inst_a-np.mean(inst_a))**2) # variance of amplitude
    m3_a = np.mean((inst_a-np.mean(inst_a))**3)/(np.mean((inst_a-np.mean(inst_a))**2)**(3/2)) # skewness of amplitude
    m4_a = np.mean((inst_a-np.mean(inst_a))**4)/(np.mean((inst_a-np.mean(inst_a))**2)**(2)) # kurtosis of amplitude
    
    # Phase statistics
    m2_p = np.mean((inst_p-np.mean(inst_p))**2) # variance of phase
    m3_p = np.mean((inst_p-np.mean(inst_p))**3)/(np.mean((inst_p-np.mean(inst_p))**2)**(3/2)) # skewness of phase
    m4_p = np.mean((inst_p-np.mean(inst_p))**4)/(np.mean((inst_p-np.mean(inst_p))**2)**(2)) # kurtosis of phase
    
    return  m2_a, m3_a, m4_a, m2_p, m3_p, m4_p

def to_onehot(labels, num_classes):
    num_labels = len(labels)
    labels_onehot = np.zeros((num_labels, num_classes))
    labels_onehot[np.arange(num_labels),labels] = 1
    
    return labels_onehot

def pulse_shape(symbols, sps=5):
    num_weights = 251
    x = np.arange(-int(num_weights/2),int(num_weights/2)+1,1)/sps
    sinc_weights = np.sinc(x)
    
    padded_symbols = np.zeros(len(symbols)*sps, dtype=np.complex)
    padded_symbols[np.arange(0,len(padded_symbols),sps)] = symbols
    
    return np.convolve(padded_symbols, sinc_weights, mode='same')