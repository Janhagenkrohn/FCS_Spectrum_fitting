clear



file_paths = {
    '\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\PAM_simulations\3f\single_filament_run1\3peaks_mu5-20-50_sigma2-30-10_label1e-0_simData.mat'
    '\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\PAM_simulations\3f\single_filament_run1\3peaks_mu5-20-50_sigma2-30-10_label1e-1_simData.mat'
    '\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\PAM_simulations\3f\single_filament_run1\3peaks_mu5-20-50_sigma2-30-10_label1e-2_simData.mat'
    '\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\PAM_simulations\3f\single_filament_run1\3peaks_mu5-20-50_sigma2-30-10_label1e-3_simData.mat'
    '\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\PAM_simulations\3f\single_filament_run1\3peaks_mu5-20-50_sigma2-30-10_label1e-4_simData.mat'
    '\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\PAM_simulations\3f\single_filament_run1\1peaks_mu10_sigma5_label1e-0_simData.mat'
    };


lagmin_s = 1e-3;
lagmax_s = 3;
Sampling = 12;

sim_time_s = 60;

sd_method = 'Bootstrap'; % 'Wohland', 'Bootstrap'

for i_file = 1:length(file_paths)

%     try % Try-catch, as correlation algorithm can crash for awkward combinations of dataset and correlation settigs
    
        %%% Load simulation data
        file_path = file_paths{i_file};
        disp(['Processing ' file_path '...'])
        load(file_path)
            
        macro_times = Sim_Photons{1,1};
        lagmin_sync = lagmin_s .* Header.Freq;
        lagmax_sync = lagmax_s .* Header.Freq;
    
        [folder, baseFileNameNoExt, ~] = fileparts(file_path);

        
        
        %%% FCS
        savename = fullfile(folder, [baseFileNameNoExt, '_ACF_ch0.csv']);
        disp(['Correlation function'])
        % Correlation curve (CC) calculation
        [CC_raw, lags] = cross_corr(...
            macro_times, macro_times,...
            lagmin_sync, lagmax_sync, Sampling, 0);

        lags = lags' ./ Header.Freq;
        CC = CC_raw' - 1;


        if strcmp(sd_method, 'Wohland')
            nSegments = 10;

            disp(['Wohland SD: ' num2str(nSegments) ' segments'])

        
            % Calculation of CC standard deviation
            segment_length = sim_time_s ./ nSegments .* Header.Freq;
            segmentCCs = zeros(length(CC_raw), nSegments);
            segmentCC_m = zeros(size(CC_raw));
            % Minimization target when amplitude-matching the segment CCs to
            % the full-length curve 
            fun_inner = @(x, segmentCC_m, CC_raw) sum((x .* segmentCC_m - CC_raw).^2);
            fun_outer = @(x) fun_inner(x, segmentCC_m, CC_raw);
        
        
            % Calculate (and scale) segment CCs
        
            parfor m = 1:nSegments
        
                macro_times_segment = macro_times...
                    (macro_times >= segment_length .* (m-1) &...
                     macro_times < segment_length .* (m));
                macro_times_segment = macro_times_segment - segment_length .* (m-1);
                
                [segmentCC_m, ~ ] = cross_corr(macro_times_segment,macro_times_segment, lagmin_sync, lagmax_sync, Sampling, 0);
                
                segmentCCs(:, m) = segmentCC_m .* fminsearch(fun_outer, 1);
            end % for iSegment = 1:nSegments
        
            SD_CC = std(segmentCCs, 0, 2) ./ sqrt(nSegments);

        else % implies strcmp(sd_method, 'Bootstrap')
            n_bs_reps = 10;
            disp(['Bootstrap SD: ' num2str(n_bs_reps) ' replicates'])

            bs_rep_CCs = zeros(length(CC_raw), n_bs_reps);

            parfor i_rep = 1:n_bs_reps
                rand_indices = randi(numel(macro_times), [numel(macro_times), 1]);
                resampled_macro_times = macro_times(rand_indices);
                [bs_rep_CC, ~ ] = cross_corr(resampled_macro_times,resampled_macro_times, lagmin_sync, lagmax_sync, Sampling, 0);
                bs_rep_CCs(:,i_rep) = bs_rep_CC

            end % parfor i_rep = 1:n_bs_reps


            SD_CC = std(bs_rep_CCs, 0, 2);

        end % if strcmp(sd_method, 'Wohland')


        % Finish up
        disp(['Wrap-up and saving'])

        cntrate_scalar = numel(macro_times) ./sim_time_s;

        cntrate_col = [cntrate_scalar;...
                       cntrate_scalar;...
                       zeros(length(lags) - 2, 1)];

        Out_ChiSurf = [lags(2:end), CC(2:end), cntrate_col(1:end-1), SD_CC(2:end)];

        writematrix(Out_ChiSurf, savename)

        %%% PCH

        disp(['Starting PCH'])

        % us-time scale time tags
        macro_times_us = macro_times ./ Header.Freq .* 1E6; 
        sim_time_us = sim_time_s .* 1E6;

        % Get log-spaced bin widths
        bin_widths = []; % start with 1 us
        curr_bw = 2 ; % start with 2 us
        i_bw = 1;
        while curr_bw <= 1E4 % bin_widths < 10 ms
            bin_widths(i_bw) = curr_bw;
            curr_bw = curr_bw .* 2;
            i_bw = i_bw + 1;
        end % while curr_bw <= 1E4
        
        % Largest bin width first as it defines the overall size of the
        % required maxtrix

        disp(['PCH: longest bin width'])

        n_bins = floor(sim_time_us / bin_widths(end)) + 1;
        [trace, ~] = histcounts(macro_times_us, linspace(0, n_bins .* bin_widths(end), n_bins));
        max_photons = max(trace);
        pch = histcounts(trace, linspace(0, max_photons + 1));
        pcmh = zeros(length(pch) + 1, length(bin_widths));
        pcmh(1,end) = bin_widths(end) * 1E-6;
        pcmh(2:end,end) = pch;

        disp(['PCH: Other bin widths (' num2str(length(bin_widths)-1) ' of them)'])
        parfor i_bw = 1:(length(bin_widths)-1)

            [trace, ~] = histcounts(macro_times_us, linspace(0, n_bins .* bin_widths(i_bw), n_bins));
            pch = histcounts(trace, linspace(0, max_photons + 1));
            pcmh(:,i_bw) = cat(1, bin_widths(i_bw) * 1E-6, pch');
        end


        disp(['Wrap-up and saving'])

        savename = fullfile(folder, [baseFileNameNoExt, '_PCMH_ch0.csv']);        
        writematrix(pcmh, savename)

%     end % try ...


end % for i_file = 1:length(file_names)

disp('Job done.')
