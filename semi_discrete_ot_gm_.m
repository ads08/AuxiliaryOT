function [heights, average_samples_per_cell] = semi_discrete_ot_gm_(gm, source, weights, thresh)

[n, d] = size(source);
if nargin < 3
    weights = 1 / n;
end
if nargin < 4
    thresh = 0.1;
end

% parameter setting
average_samples_per_cell = 4;
samples_total = average_samples_per_cell * n;
samples_each_loop = 2000;
sample_loops = ceil(samples_total / samples_each_loop);
samples_total = samples_each_loop * sample_loops;

% heights and weights initialize
% source_heights = heights_initialize_box(source);
% source_heights = zeros(n,1);
source_heights = rand(n,1);
source_heights = source_heights - mean(source_heights);

% put data into gpu
source_gpu = gpuArray(source);
source_heights_gpu = gpuArray(source_heights);

% running params
loops = 50000;
mu = 1.0;
alpha_source = sqrt(n*d)/1000;
beta_1 = 0.9;
beta_2 = 0.5;
epsilon = 1e-16;
mt_source = 0;
Vt_source = 0;

% source
area_diff_record_source = zeros(loops, 1);
hyperplane_num_record_source = zeros(loops, 1);
E = zeros(loops, 1);

for i = 1 : loops
    Idx_source = zeros(samples_each_loop, sample_loops);
    Idx_source_gpu = gpuArray(Idx_source);
    energy = 0;
    for j = 1 : sample_loops
        x = random(gm, samples_each_loop);
        x = x';
        x = gpuArray(single(x));
        hyperplanes = source_gpu*x + repmat(source_heights_gpu, 1, samples_each_loop);
        [value, idx] =  max(hyperplanes, [], 1);
        Idx_source_gpu(:, j) = idx;
        energy = energy + sum(value) / sample_loops;
    end
    Idx = gather(Idx_source_gpu);
    areas = zeros(n, 1);
    for k = 1 : samples_total
        areas(Idx(k)) = areas(Idx(k)) + 1;
    end
    areas = areas / samples_total;
    delta_h = areas - weights;
    
%     non_valid_index = find(areas == 0);
%     if length(non_valid_index) > 0
%         s = source(non_valid_index, :);
%         [Idx_s, Dists] = knnsearch(source, s, 'K', 10);
%         Idx_s_1 = Dists(:,2)<0.01;
%         Idx_s_2 = Dists(:,2)>=0.01;
%         delta_h(non_valid_index(Idx_s_1)) = 1.5 * delta_h(non_valid_index(Idx_s_1));
%         delta_h(non_valid_index(Idx_s_2)) = 3 * delta_h(non_valid_index(Idx_s_2));
%     else
%         delta_h(non_valid_index) = 1 * delta_h(non_valid_index);
%     end

%     delta_h(non_valid_index) = 1 * delta_h(non_valid_index);
    
    % adam sgd
    mt_source = beta_1 * mt_source + (1-beta_1) * delta_h;
    Vt_source = beta_2 * Vt_source + (1-beta_2) * sum(delta_h.^2);
    eta_t = alpha_source * mt_source / (sqrt(Vt_source) + epsilon);
    source_heights = update_height(source_heights, eta_t, mu);
    source_heights_gpu = gpuArray(source_heights);
    
    area_diff_record_source(i) = sum(abs(areas - weights));
    hyperplane_num_record_source(i) = sum(areas~=0);
    E(i) = gather(energy);
    
    if i >= 2 && i <= 20
        if area_diff_record_source(i) > area_diff_record_source(i-1) && hyperplane_num_record_source(i) < hyperplane_num_record_source(i-1)
            alpha_source = alpha_source / 1.5;
        end
    end
    
    if i > 21
        curr_hyper_num = mean(hyperplane_num_record_source(i-3:i));
        prev_hyper_num = mean(hyperplane_num_record_source(i-6:i-4));
        curr_area_diff = mean(area_diff_record_source(i-3:i));
        prev_area_diff = mean(area_diff_record_source(i-6:i-4));
        curr_energy = mean(E(i-3:i));
        prev_energy = mean(E(i-6:i-4));
               
        if curr_hyper_num <= prev_hyper_num && curr_area_diff >= prev_area_diff
%         if curr_energy > prev_energy
            if alpha_source > 0.01 / n^2
                alpha_source = alpha_source / 1.05;
            else
                average_samples_per_cell = average_samples_per_cell * 2;
                samples_total = average_samples_per_cell * n;
                sample_loops = ceil(samples_total / samples_each_loop);
                samples_total = samples_each_loop * sample_loops;
            end
        end
    end
    
    fprintf('%d %f %d %f %f %d\n', i, alpha_source, average_samples_per_cell, ...
            E(i), area_diff_record_source(i), hyperplane_num_record_source(i));
    
%     if hyperplane_num_record_source(i) > 0.9*n &&
%     area_diff_record_source(i) <= thresh
    if area_diff_record_source(i) <= thresh

            break;
    end
end

heights = source_heights;

end

