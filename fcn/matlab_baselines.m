snr_range = -10:15;

% packet_sizes = [64, 128, 256, 512];
packet_sizes = 128;
preamble_sizes = [8, 16, 32];
% pad_sizes = [10, 50];

num_examples = 100;

delays = [0, 5e-6, 10e-6];
gains = [-6, 0, -3];
K = 10;

% for j = 1:length(preamble_sizes)
%     [corr_der, ~, expert_der, ~] = calc_baselines(num_examples, 128, snr_range, preamble_sizes(j), 250-128-preamble_sizes(j), true, '', 'awgn');
%     semilogy(snr_range, corr_der)
%     hold on; grid on;
%     semilogy(snr_range, expert_der)
% %     save(sprintf('matlab_baselines_%d',preamble_sizes(j)), 'corr_der', 'expert_der');
% end

for j = 1:length(preamble_sizes)
    [corr_der, ~, expert_der, ~] = calc_baselines(num_examples, 128, snr_range, preamble_sizes(j), 250-128-preamble_sizes(j), true, '', 'multipath rician', delays, gains, K);
    semilogy(snr_range, corr_der)
    hold on; grid on;
    semilogy(snr_range, expert_der)
%     save(sprintf('matlab_baselines_rician_%d',preamble_sizes(j)), 'corr_der', 'expert_der');
end

function [corr_der, l3_2_der, l3_4_der, l4_der] = calc_baselines(num_examples, packet_length, snr_range, preamble_seq, padding, add_phase_offset, save_dir, channel, varargin)

    if preamble_seq > 0
        max_len_seq = [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1]';
        seq = max_len_seq(1:preamble_seq);
    else
        seq = [];
    end

    preamble = pskmod(seq, 2, pi);

    corr_der = zeros(length(snr_range),1);
    l3_2_der = zeros(length(snr_range),1);
    l3_4_der = zeros(length(snr_range),1);
    l4_der = zeros(length(snr_range),1);

    avg_filter = ones(size(preamble));
    
    if nargin > 8
        
        switch channel
            case 'awgn'
                % nothing to do here
            case 'flat rayleigh'
                % nothing to do here
            case 'flat rician'
                K = varargin{1};
            case 'multipath rayleigh'
                delays = varargin{1};
                gains = varargin{2};
            case 'multipath rician'
                delays = varargin{1};
                gains = varargin{2};
                K = varargin{3};
            otherwise
                disp("Valid names are 'flat rayleigh', 'flat rician', 'multipath rayleigh', 'multipath rician'")
        end
    end

    for k = 1:length(snr_range)
        
        switch channel
            case 'awgn'
                [data, labels] = generate_bursty_rand_test_data(num_examples, packet_length, snr_range(k), preamble_seq, padding, add_phase_offset, channel);
            case 'flat rayleigh'
                [data, labels] = generate_bursty_rand_test_data(num_examples, packet_length, snr_range(k), preamble_seq, padding, add_phase_offset, channel);
            case 'flat rician'
                [data, labels] = generate_bursty_rand_test_data(num_examples, packet_length, snr_range(k), preamble_seq, padding, add_phase_offset, channel, K);
            case 'multipath rayleigh'
                [data, labels] = generate_bursty_rand_test_data(num_examples, packet_length, snr_range(k), preamble_seq, padding, add_phase_offset, channel, delays, gains);
            case 'multipath rician'
                [data, labels] = generate_bursty_rand_test_data(num_examples, packet_length, snr_range(k), preamble_seq, padding, add_phase_offset, channel, delays, gains, K);
        end
        
        if exist(save_dir)
            save(save_dir + sprintf("%d", snr_range(k)), 'data', 'labels');
        end

        num_errors_corr = 0;
        num_errors_l3_2 = 0;
        num_errors_l3_4 = 0;
        num_errors_l4 = 0;

        for i = 1:size(data,1)

            signal_noisy = data(i,:)';

            correlation = xcorr(signal_noisy,preamble);
            correlation = correlation(size(signal_noisy,1):end);

            abs_average = xcorr(abs(signal_noisy), avg_filter);
            abs_average = abs_average(size(signal_noisy,1):end);

            abs_average_i = xcorr(abs(real(signal_noisy)), avg_filter);
            abs_average_i = abs_average_i(size(signal_noisy,1):end);

            abs_average_q = xcorr(abs(imag(signal_noisy)), avg_filter);
            abs_average_q = abs_average_q(size(signal_noisy,1):end);

            abs_average_sum = xcorr(abs(imag(signal_noisy)+real(signal_noisy)), avg_filter);
            abs_average_sum = abs_average_sum(size(signal_noisy,1):end)/sqrt(2);

            abs_average_diff = xcorr(abs(real(signal_noisy)-imag(signal_noisy)), avg_filter);
            abs_average_diff = abs_average_diff(size(signal_noisy,1):end)/sqrt(2);

            lambda3_2 = abs(correlation) - max([abs_average_i, abs_average_q], [], 2);

            lambda3_4 = abs(correlation) - max([abs_average_i, abs_average_q, abs_average_sum, abs_average_diff], [], 2);
            
            lambda4 = abs(correlation) - abs_average;

            [~, peak_index] = max(correlation(1:end-length(preamble)));

            [~, peak_index2] = max(lambda3_2(1:end-length(preamble)));

            [~, peak_index3] = max(lambda3_4(1:end-length(preamble)));

            [~, peak_index4] = max(lambda4(1:end-length(preamble)));

            true_index = labels(i);

            if true_index ~= peak_index
                num_errors_corr = num_errors_corr + 1;
            end

            if true_index ~= peak_index2
                num_errors_l3_2 = num_errors_l3_2 + 1;
            end

            if true_index ~= peak_index3
                num_errors_l3_4 = num_errors_l3_4 + 1;
            end

            if true_index ~= peak_index4
                num_errors_l4 = num_errors_l4 + 1;
            end

        end

        corr_der(k) = num_errors_corr/size(data,1);
        l3_2_der(k) = num_errors_l3_2/size(data,1);
        l3_4_der(k) = num_errors_l3_4/size(data,1);
        l4_der(k) = num_errors_l4/size(data,1);
    end

%     figure; %hold on; grid on;
%     semilogy(snr_range, corr_der)
%     hold on; grid on;
%     semilogy(snr_range, l3_2_der)
%     semilogy(snr_range, l3_4_der)
%     semilogy(snr_range, l4_der)
%     legend('correlation', 'lambda3 Nq=2', 'lambda3 Nq=4','lambda4')
    
end