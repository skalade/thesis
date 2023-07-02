import numpy as np
import math
from scipy import signal

from typing import Tuple

def awgn(signal: np.ndarray, SNR: float, measured: bool=True) -> np.ndarray:
    '''
    Add AWGN to complex signal.

    Parameters:
    signal (numpy.ndarray): complex signal (I haven't tested real)
    SNR (float): SNR in dB
    measured (bool): MATLAB does measured by default I think

    Returns:
    numpy.ndarray: signal at SNR dB
    '''

    if measured:
        # Calculcate signal power 
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

def phase_offset(x: np.ndarray, offset: float) -> np.ndarray:
    '''
    Add a phase offset to an array of complex numbers or number.

    Parameters:
    x (numpy.ndarray): input signal
    offset (float): phase offset in degrees

    Returns:
    numpy.ndarray: signal + phase offset
    '''
    
    # Convert to polar form so we can reason in degrees
    mag, ang = np.abs(x), np.angle(x, deg=True)
    
    return mag * np.exp( 1j * ((ang + offset)*np.pi/180) )

def create_frame(preamble_seq: np.ndarray, payload: int=128, signal_length: int=200, offset: int=40) -> np.ndarray:
    '''
    Create a frame of data with a preamble and random time offset.

    Parameters:
    preamble_seq (numpy.ndarray): The preamble sequence.
    payload (int): The number of bits in the payload. Default is 128.
    signal_length (int): The total length of the signal. Default is 200.
    offset (int): The offset for the packet in the waveform. Default is 40.

    Returns:
    numpy.ndarray: The created frame as a complex waveform.
    '''

    # Create empty frame
    waveform = np.zeros((signal_length,),dtype=np.complex128)
    
    # Generate packet with preamble + payload
    bits = np.random.randint(0,2,payload)
    packet = np.concatenate((preamble_seq,bits))

    # BPSK modulate
    packet = np.where(packet < 1, -1+0j, 1+0j)

    # Add time offset
    waveform[offset:offset+payload+len(preamble_seq)] = packet
    
    return waveform

def generate_qpsk(num_symbols: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generate random QPSK symbols.

    Parameters:
    num_symbols (int): The number of QPSK symbols to generate.

    Returns:
    tuple: QPSK symbols as a numpy.ndarray and the corresponding integers (labels).
    '''

    qpsk_scheme= [1+1j, 1-1j, -1+1j, -1-1j]

    # Generate random bits
    ints = np.random.randint(0,4,num_symbols)

    # QPSK modulate
    qpsk_symbols = np.array([qpsk_scheme[i] for i in ints])/np.sqrt(2)
    
    return qpsk_symbols, ints

def generate_bpsk(num_symbols: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generate random BPSK symbols.

    Parameters:
    num_symbols (int): The number of BPSK symbols to generate.

    Returns:
    tuple: The generated BPSK symbols as a numpy.ndarray and the corresponding integers (labels).
    '''
    bpsk_scheme= [1+0j, -1+0j]
    ints = np.random.randint(0,2,num_symbols)
    bpsk_symbols = np.array([bpsk_scheme[i] for i in ints])
    
    return bpsk_symbols, ints

def make_rrc(num_weights: int=41, alpha: float=0.35, fs: int=5) -> np.ndarray:
    '''
    Generate a Root Raised Cosine (RRC) filter.

    Parameters:
    num_weights (int): The number of weights in the filter. Default is 41.
    alpha (float): The roll-off factor. Default is 0.35.
    fs (int): The sampling frequency. Default is 5.

    Returns:
    numpy.ndarray: The generated RRC filter weights.
    '''

    Ts =  1/fs
    
    # Calculate time indexes of filter weights
    x = np.arange(-int(num_weights/2),int(num_weights/2),1)/fs
    
    # Preallocate memory for filter weights
    h_rrc = np.zeros(num_weights,)

    # Check for special cases, otherwise apply the main formula
    for idx, weight in enumerate(x):
        if weight == Ts/(4*alpha):
            h_rrc[idx] = (alpha/np.sqrt(2))*(((1+2/np.pi) * (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        elif weight == -Ts/(4*alpha):
            h_rrc[idx] = (alpha/np.sqrt(2))*(((1+2/np.pi) * (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        elif weight == 0:
            h_rrc[int(num_weights/2)] = 1/Ts*(1+alpha*(4/np.pi - 1))
        else:
            h_rrc[idx] = 1/Ts*(np.sin(np.pi*weight*(1-alpha)) + 4*alpha*weight*np.cos(np.pi*weight*(1+alpha)))/(np.pi*weight*(1-(4*alpha*weight)**2))
            
    # Normalize the weights
    h_rrc = h_rrc/np.max(h_rrc)
    
    return h_rrc

def pulse_shape(symbols: np.ndarray, hrrc: np.ndarray, sps: int=4) -> np.ndarray:
    '''
    Pulse shape the symbols using an RRC filter.

    Parameters:
    symbols (numpy.ndarray): The symbols to pulse shape.
    hrrc (numpy.ndarray): The RRC filter to use for pulse shaping.
    sps (int): The number of samples per symbol. Default is 4.

    Returns:
    numpy.ndarray: The pulse-shaped symbols.
    '''
    
    num_weights = len(hrrc)
    
    padded_symbols = np.zeros(len(symbols)*sps, dtype=complex)
    padded_symbols[np.arange(0,len(padded_symbols),sps)] = symbols
    
    # to take care of the transient we append a bunch of zeros
    shaped_symbols = np.convolve(np.concatenate((np.zeros(int((num_weights-1)/2)),padded_symbols)), hrrc, mode='same')
    
    return shaped_symbols[int((num_weights-1)/2):]


def sym_to_bit(data: np.ndarray, M: int=4) -> np.ndarray:
    '''
    Convert symbols to bits, based on the specified modulation scheme.
    This is a helper for demodulation baselines, to get BER and stuff.

    Parameters:
    data (numpy.ndarray): Array of symbols.
    M (int): Modulation scheme. Default is 4 (QPSK).

    Returns:
    numpy.ndarray: Array of bits.
    '''
    if M == 2:
        
        symbol_dict = {0 : (0,),
                       1 : (1,)}
        
    elif M == 4:

        symbol_dict = {0 : (0, 0),
                       1 : (0, 1),
                       2 : (1, 0),
                       3 : (1, 1)}
        
    elif M == 8:
        
        symbol_dict = {0 : (0,0,0),
                       1 : (0,0,1),
                       2 : (0,1,0),
                       3 : (0,1,1),
                       4 : (1,0,0),
                       5 : (1,0,1),
                       6 : (1,1,0),
                       7 : (1,1,1)}
        
    elif M ==16:
        
        symbol_dict = {0 : (0,0,0,0),
                       1 : (0,0,0,1),
                       2 : (0,0,1,0),
                       3 : (0,0,1,1),
                       4 : (0,1,0,0),
                       5 : (0,1,0,1),
                       6 : (0,1,1,0),
                       7 : (0,1,1,1),
                       8 : (1,0,0,0),
                       9 : (1,0,0,1),
                       10 : (1,0,1,0),
                       11 : (1,0,1,1),
                       12 : (1,1,0,0),
                       13 : (1,1,0,1),
                       14 : (1,1,1,0),
                       15 : (1,1,1,1)}
        
    else:
        return 1
    
    bits = np.array([symbol_dict[symbol] for symbol in data]).reshape(-1)
    
    return bits

def bit_to_sym(bits: np.ndarray, M: int=4) -> np.ndarray:
    '''
    Convert bits to symbols, based on the specified modulation scheme.

    Parameters:
    bits (numpy.ndarray): Array of bits.
    M (int): Modulation scheme. Default is 4 (QPSK).

    Returns:
    numpy.ndarray: Array of symbols.
    '''

    if M == 2:
        
        symbol_dict = {(0,): 0,
                       (1,): 1}
        
    elif M == 4:

        symbol_dict = {(0, 0): 0,
                       (0, 1): 1,
                       (1, 0): 2,
                       (1, 1): 3}
    elif M == 8:
        
        symbol_dict = {(0,0,0,0) : 0,
                       (0,0,0,1) : 1,
                       (0,0,1,0) : 2,
                       (0,0,1,1) : 3,
                       (0,1,0,0) : 4,
                       (0,1,0,1) : 5,
                       (0,1,1,0) : 6,
                       (0,1,1,1) : 7}
        
    elif M ==16:
        
        symbol_dict = {(0,0,0,0) : 0,
                       (0,0,0,1) : 1,
                       (0,0,1,0) : 2,
                       (0,0,1,1) : 3,
                       (0,1,0,0) : 4,
                       (0,1,0,1) : 5,
                       (0,1,1,0) : 6,
                       (0,1,1,1) : 7,
                       (1,0,0,0) : 8,
                       (1,0,0,1) : 9,
                       (1,0,1,0) : 10,
                       (1,0,1,1) : 11,
                       (1,1,0,0) : 12,
                       (1,1,0,1) : 13,
                       (1,1,1,0) : 14,
                       (1,1,1,1) : 15}
        
    else:
        return 1
    
    if M != 2:
        bits = np.reshape(bits, (-1,int(np.sqrt(M))))
    
    symbols = np.array([symbol_dict[tuple(bit)] for bit in bits])
    
    return symbols

def modulate(data: np.ndarray, M: int=4, normalized: bool=True) -> np.ndarray:
    '''
    Modulates data using BPSK/QPSK/8-PSK/16-QAM schemes.

    Parameters:
    data (numpy.ndarray): Array of data to be modulated.
    M (int): Modulation scheme, accepted values are 2 (BPSK), 4 (QPSK), 8 (8PSK), 16 (16QAM). Default is 4 (QPSK).
    normalized (bool): If True, the average symbol power is normalized to 1. Default is True.

    Returns:
    numpy.ndarray: Modulated symbols array.
    '''
    
    bpsk_constellation = [-1+0j, 1+0j]
    qpsk_constellation = [1+1j, 1-1j, -1+1j, -1-1j]
    psk8_constellation = [1+0j, 0.7071+0.7071j, 0+1j, -0.7071+0.7071j, -1+0j, -0.7071-0.7071j, 0-1j, 0.7071-0.7071j]
    qam16_constellation = [-3-3j,-3-1j,-3+3j,-3+1j,-1-3j,-1-1j,-1+3j,-1+1j,3-3j,3-1j,3+3j,3+1j,1-3j,1-1j,1+3j,1+1j]
    
    bpsk_normalized = bpsk_constellation/np.sqrt(np.mean(np.abs(bpsk_constellation)**2))
    qpsk_normalized = qpsk_constellation/np.sqrt(np.mean(np.abs(qpsk_constellation)**2))
    psk8_normalized = psk8_constellation
    qam16_normalized = qam16_constellation/np.sqrt(np.mean(np.abs(qam16_constellation)**2))
    
    if M == 2:
        symbol_dict = bpsk_normalized
        
    elif M == 4:
        symbol_dict = qpsk_normalized
        
    elif M == 8:
        symbol_dict = psk8_normalized
        
    elif M ==16:
        
        symbol_dict = qam16_normalized
        
    else:
        return 1
    
    symbols = np.array([symbol_dict[bits] for bits in data])
    
    return symbols

def demodulate(symbols: np.ndarray, M: int=4) -> np.ndarray:
    '''
    Demodulates data using BPSK/QPSK/8-PSK/16-QAM schemes. Probably not the most efficient method...

    Parameters:
    symbols (numpy.ndarray): Array of modulated symbols to be demodulated.
    M (int): Modulation scheme, accepted values are 2 (BPSK), 4 (QPSK), 8 (8PSK), 16 (16QAM). Default is 4 (QPSK).

    Returns:
    numpy.ndarray: Demodulated data array.
    '''

    bpsk_constellation = [-1+0j, 1+0j]
    qpsk_constellation = [1+1j, 1-1j, -1+1j, -1-1j]
    psk8_constellation = [1+0j, 0.7071+0.7071j, 0+1j, -0.7071+0.7071j, -1+0j, -0.7071-0.7071j, 0-1j, 0.7071-0.7071j]
    qam16_constellation = [-3-3j,-3-1j,-3+3j,-3+1j,-1-3j,-1-1j,-1+3j,-1+1j,3-3j,3-1j,3+3j,3+1j,1-3j,1-1j,1+3j,1+1j]
    
    if M == 2:
        symbol_dict = bpsk_constellation/np.sqrt(np.mean(np.abs(bpsk_constellation)**2))
        
    elif M == 4:
        symbol_dict = qpsk_constellation/np.sqrt(np.mean(np.abs(qpsk_constellation)**2))
        
    elif M == 8:
        symbol_dict = psk8_constellation

    elif M ==16:
        symbol_dict = qam16_constellation/np.sqrt(np.mean(np.abs(qam16_constellation)**2))
    
    data = np.zeros(symbols.shape[0], dtype=int)
    
    for i,symbol in enumerate(symbols):
        data[i] = np.argmin([np.abs(point - symbol) for point in symbol_dict])
                
    return data