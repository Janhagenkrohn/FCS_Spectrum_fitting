function [corr_norm, lags] = cross_corr(ch1,ch2, start_time, stop_time, coarseness, offset_lag)

%--------------Inputs--------------%
%ch1 - first array of photon arrival times
%ch2 - second array of photon arrival times
%start_time - smallest lag time to be analyzed (the lowest value is 1 time
%   unit)
%stop_time - largest lag time
%coarseness - a number usually between 1 and 10, that determines the number
%   of points in the acf curve for each logarithmic cascade (1-low
%   resolution, 10-high). 

%------------Outputs---------------%
% corr_norm - normalized cross correlation between ch1 and ch2
% lags - log spaced lag times

cascade_start = floor(log2(start_time)-2);% if above 0, the algorithm skips the 2^start_cascade correlation times
cascade_start = max([cascade_start, 1]); % Force positive-nonzero
[lag_bin_edges, lags, division_factor] = generate_log2_lags(stop_time, coarseness); % generates lags bins with log2 spacing

corr = photons_in_bins(ch1, ch2, lag_bin_edges, ...
            cascade_start, coarseness, offset_lag);

num_ch1 = numel(ch1); num_ch2 = numel(ch2); %number of photons in each channel
ch1_max = ch1(end); ch2_max = ch2(end);
tcor = min(ch1_max, ch2_max-lags);
skip_lags = cascade_start*coarseness;

%%Normalization
corr_div = corr./division_factor(2:end);
%cor_part_norm = corr_div./tcor;
corr_norm = 2.*(corr_div./tcor)./((num_ch1./ch1_max).*(num_ch2./ch2_max));

corr_norm = corr_norm(:, skip_lags:end);
%corr_div1 = cor_part_norm(:, skip_lags:end);
lags = lags(skip_lags:end);


end

function [lag_bin_edges, lags, division_factor] = generate_log2_lags(t_end, coarseness)      
           %%Generates log2 spaced photon bins
           cascade_end = floor(log2(t_end)-2); % cascades are collections of lags with equal bin spacing 2^cascade
           nper_cascade = coarseness; % number of equal
           
           n_edges = cascade_end*nper_cascade;
           lag_bin_edges = zeros(1, n_edges); %lag bins
           
           for j = 1:n_edges
               if j == 1
                   lag_bin_edges(j) = 1;
               else
                   lag_bin_edges(j) = lag_bin_edges(j-1)+2^(floor((j-1)./nper_cascade));
               end
           end
           lags = diff(lag_bin_edges)./2+lag_bin_edges(1:end-1); %centers of the lag bins
           division_factor = kron(2.^(1:cascade_end), ones(1, nper_cascade)); %for normalization
end


function acf = photons_in_bins(ch1, ch2, lag_bin_edges, ...
               cascade_start, nper_cascade, offset_lag)
           %counts the number of photons separated from each individual
           %photon by the distance specified in the lag_bin_edges
           %The time-saving step is to keep track of the earliest and
           %latest photon in each bin, that way when going from one photon
           %to the next, a minimum amount of comparison operations is
           %performed.
           % Based on A. Laurence, Samantha Fore, and Thomas Huser, 
           % "Fast, flexible algorithm for calculating photon correlations," 
           % Opt. Lett. 31, 829-831 (2006)
           
           num_ch1 = numel(ch1);
           n_edges = numel(lag_bin_edges);
           low_inds = ones(1, n_edges-1); 
           low_inds(1) = 2;
           max_inds = ones(1, n_edges-1);
           acf = zeros(1, n_edges-1);

           for phot_ind = 1:num_ch1
               bin_edges = ch1(phot_ind)+lag_bin_edges+offset_lag; %shifts the log bins for each photon
               
               for k = cascade_start*nper_cascade:n_edges-1 % counts photons in bins
                   while low_inds(k) <= numel(ch2) && ch2(low_inds(k)) < bin_edges(k)
                       low_inds(k) = low_inds(k)+1;
                   end

                   while max_inds(k) <= numel(ch2) && ch2(max_inds(k)) <= bin_edges(k+1)
                       max_inds(k) = max_inds(k)+1;
                   end
                   low_inds(k+1) = max_inds(k);
                   acf(k) = acf(k)+(max_inds(k)-low_inds(k));
               end
           end
       end