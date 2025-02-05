function [I_train, J_train, mask_train, kk_train, I_test, J_test, mask_test, kk_test] = randmask(m, n, k, test_ratio, fencemax)
%
% Extended version of the randmask function to generate both training and testing datasets
% with non-overlapping samples.
%
% Parameters:
% m, n        - Dimensions of the matrix
% k           - Total number of train samples
% test_ratio  - Fraction of samples allocated to the test set (e.g., 0.2 for 20%)
% fencemax    - Maximum number of tries (optional)
%
% Returns:
% I_train, J_train, mask_train, kk_train - Training set indices and mask
% I_test, J_test, mask_test, kk_test     - Test set indices and mask
%

    if nargin < 4 || isempty(test_ratio)
        test_ratio = 1; % Default test set ratio
    end
    if nargin < 5 || isempty(fencemax)
        fencemax = 5;
    end

    % Number of test and training samples
    k_test = round(k * test_ratio);
    k_train = k;

    % Helper function for sampling unique subsets
    function [I, J, mask, kk, used_indices] = sample_subset(num_samples, excluded_indices)
        used_indices = zeros(0, 2, 'uint32'); % Ensure excluded_indices is a 2-column matrix
        if m*n > 1e7 || ~exist('randsample') % randsample requires a toolbox
            pos = zeros(0, 2, 'uint32');
            kk = 0;
            fence = 1;
            while kk < num_samples
                if fence > fencemax
                    warning('randmask: Could not select exactly %d entries; selected %d instead.\n', num_samples, kk);
                    break;
                end
                rows = randi(m, num_samples-kk, 1, 'uint32');
                cols = randi(n, num_samples-kk, 1, 'uint32');
                new_pos = [rows, cols];
                new_pos = unique(new_pos, 'rows', 'stable'); % Remove duplicates in new samples
                % Exclude already used indices
                if ~isempty(excluded_indices)
                    new_pos = setdiff(new_pos, excluded_indices, 'rows', 'stable');
                end
                pos = [pos; new_pos]; % Accumulate positions
                pos = unique(pos, 'rows', 'stable'); % Ensure global uniqueness
                kk = size(pos, 1);
                fence = fence + 1;
            end

            I = pos(:, 1);
            J = pos(:, 2);
            used_indices = pos;

        else
            all_indices = 1:m*n;
            excluded_linear = sub2ind([m, n], double(excluded_indices(:, 1)), double(excluded_indices(:, 2)));
            available_indices = setdiff(all_indices, excluded_linear);
            idx = uint32(randsample(available_indices, num_samples));
            [I, J] = ind2sub([m n], idx);
            kk = length(I);
            used_indices = [I, J];
        end

        mask = sparse(double(I), double(J), ones(kk, 1), m, n, kk);
        [I, J] = find(mask);
        I = uint32(I);
        J = uint32(J);
    end

    % Generate training set
    [I_train, J_train, mask_train, kk_train, train_indices] = sample_subset(k_train, []);

    % Generate test set, avoiding training indices
    [I_test, J_test, mask_test, kk_test] = sample_subset(k_test, train_indices);

end