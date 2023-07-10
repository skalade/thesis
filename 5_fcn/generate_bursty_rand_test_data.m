function [data, labels] = generate_bursty_rand_test_data(num_examples, packet_length, snr, preamble_seq, pad_length, phase_offset, channel, varargin)
                 
    
    if preamble_seq > 0
        max_len_seq = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1]';
        seq = max_len_seq(1:preamble_seq);
    else
        seq = [];
    end
    
%     pad = complex(zeros(pad_length,1));
    
    % Preallocate data and label arrays
    data = zeros(num_examples, length(seq)+packet_length+pad_length);
    labels = zeros(num_examples,1);
    
    % Optionally create phase offset object
    if phase_offset
        pfo = comm.PhaseFrequencyOffset('PhaseOffset',60);
    end
    
    if nargin > 7
        
        switch channel
            case 'flat rayleigh'
                % nothing to be done here
            case 'flat rician'
                K = varargin{1};
            case 'multipath rayleigh'
                delays = varargin{1};
                gains = varargin{2};
                
                chan = comm.RayleighChannel(...
                    'SampleRate',100e3, ...
                    'PathDelays', delays, ...
                    'AveragePathGains', gains, ...
                    'NormalizePathGains',true, ...
                    'MaximumDopplerShift',0, ...
                    'RandomStream','mt19937ar with seed', ...
                    'Seed',0, ...
                    'PathGainsOutputPort',true);
                
            case 'multipath rician'
                delays = varargin{1};
                gains = varargin{2};
                K = varargin{3};
                
                chan = comm.RicianChannel( ...
                'SampleRate',100e3, ...
                'PathDelays',delays, ...
                'AveragePathGains',gains, ...
                'NormalizePathGains',true, ...
                'KFactor', K, ...
                'DirectPathDopplerShift',0, ...
                'MaximumDopplerShift',0, ...
                'RandomStream','mt19937ar with seed', ...
                'Seed',1, ...
                'PathGainsOutputPort',true);
            
            otherwise
                disp("Valid names are 'flat rayleigh', 'flat rician', 'multipath rayleigh', 'multipath rician'")
        end
    end
    
    for i = 1:num_examples
        
        front_pad = pad_length - randi([0 pad_length]);
%        front_pad = pad_length - randi([30 pad_length]);
        back_pad = pad_length - front_pad;

        packet = randi([0 1], packet_length, 1);

        symbols = pskmod([seq; packet],2,pi);

        waveform = [complex(zeros(front_pad,1)); symbols; complex(zeros(back_pad,1))];
        
        % Optionally add phase offsets
        if phase_offset
            pfo.release;
            pfo.PhaseOffset = randi([-180 180]);
            waveform = pfo(waveform);
        end
        
        
        % Optionally add channel effects
        if strcmp(channel, 'multipath rayleigh') || strcmp(channel, 'multipath rician')
            chan.release;
            chan.Seed = randi([0 10000],1);
            [waveform, pathGains] = chan(waveform);

            chan_info = info(chan);
            pathFilters = chan_info.ChannelFilterCoefficients;
            chan_delay = channelDelay(pathGains, pathFilters);

            true_index = 1 + front_pad + chan_delay;

        elseif strcmp(channel, 'flat rayleigh')
            gains = ones(size(waveform))*1/sqrt(2)*(randn(1)+1i*randn(1));
            waveform = gains.*waveform;

            true_index = 1 + front_pad;

        elseif strcmp(channel, 'flat rician')

            nonlos = sqrt(1/(K+1)) * (randn(1)+1i*randn(1));
            los = sqrt(K/(K+1)) * 1*ones(size(waveform)) * exp(1i*randn(1));

            ric_samples = los+nonlos;
            waveform = ric_samples.*waveform;

            true_index = 1 + front_pad;
        else

            true_index = 1 + front_pad;

        end
        
        % We don't do 'measured' because we're appending a lot of zeros
        waveform_noisy = awgn(waveform, snr);
        
        labels(i) = true_index;
        data(i,:) = waveform_noisy;
        
    end

end

