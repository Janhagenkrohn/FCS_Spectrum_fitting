clear



file_names = {
    'batch3e_1_label5e-1_simData.mat'
    };

folders = {
    '\\samba-pool-schwille-spt.biochem.mpg.de\pool-schwille-spt\P6_FCS_HOassociation\Data\PAM_simulations\3e'
    };

lagmin_s = 1e-5;
lagmax_s = 1;
Sampling = 6;

sim_time_s = 60;


for i_file = 1:length(file_names)

%     try % Try-catch, as correlation algorithm can crash for awkward combinations of dataset and correlation settigs
    
        file_path = fullfile(folders{i_file}, file_names{i_file});
        disp(['Processing ' file_path '...'])
        load(file_path)
            
        macro_times = Sim_Photons{1,1};
        lagmin_sync = lagmin_s .* Header.Freq;
        lagmax_sync = lagmax_s .* Header.Freq;
    
        [~, baseFileNameNoExt, ~] = fileparts(file_path);
        savename = fullfile(folders{i_file}, [baseFileNameNoExt, '_ACF_ch0.csv']);

        % Correlation curve (CC) calculation
        [CC_raw, lags] = cross_corr(...
            macro_times, macro_times,...
            lagmin_sync, lagmax_sync, Sampling, 0);

        lags = lags' ./ Header.Freq;
        CC = CC_raw' - 1;

        nSegments = 10;

        % Calculation of CC standard deviation
        segment_length = sim_time_s ./ nSegments .* Header.Freq;
        segmentCCs = zeros(length(CC_raw), nSegments);
        segmentCC_m = zeros(size(CC_raw));
        % Minimization target when amplitude-matching the segment CCs to
        % the full-length curve 
        fun_inner = @(x, segmentCC_m, CC_raw) sum((x .* segmentCC_m - CC_raw).^2);
        fun_outer = @(x) fun_inner(x, segmentCC_m, CC_raw);

            % Calculate (and scale) segment CCs
            for m = 1:nSegments
                macro_times_segment = macro_times...
                    (macro_times >= segment_length .* (m-1) &...
                     macro_times < segment_length .* (m));
                macro_times_segment = macro_times_segment - segment_length .* (m-1);
                
                [segmentCC_m, ~ ] = cross_corr(macro_times_segment,macro_times_segment, lagmin_sync, lagmax_sync, Sampling, 0);
                
                segmentCCs(:, m) = segmentCC_m .* fminsearch(fun_outer, 1);
            end % for iSegment = 1:nSegments

            SD_CC = std(segmentCCs, 0, 2) ./ sqrt(nSegments);

            % Finish up
            cntrate_scalar = numel(macro_times) ./sim_time_s;

            cntrate_col = [cntrate_scalar;...
                           cntrate_scalar;...
                           zeros(length(lags) - 2, 1)];

            Out_ChiSurf = [lags(2:end), CC(2:end), cntrate_col(1:end-1), SD_CC(2:end)];

        writematrix(Out_ChiSurf, savename)

%     end % try ...


end % for i_file = 1:length(file_names)

disp('Job done.')
