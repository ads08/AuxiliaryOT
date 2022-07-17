function M = compute_connectivity(source, target, source_heights, target_heights)
[m, d] = size(source);
n = size(target, 1);

source_gpu = gpuArray(source);
source_heights_gpu = gpuArray(source_heights);
target_gpu = gpuArray(target);
target_heights_gpu = gpuArray(target_heights);

% find the connectivity matrix
sample_loops = 1000;
samples_each_loop = 5000;
Idx_source_gpu = gpuArray(zeros(samples_each_loop, sample_loops));
Idx_target_gpu = gpuArray(zeros(samples_each_loop, sample_loops));
for j = 1 : sample_loops
    x = rand(d, samples_each_loop);
    x = gpuArray(single(x));
    hyperplanes_source = source_gpu*x + repmat(source_heights_gpu, 1, samples_each_loop);
    [~, idx_source] =  max(hyperplanes_source, [], 1);
    Idx_source_gpu(:, j) = idx_source;
    hyperplanes_target = target_gpu*x + repmat(target_heights_gpu, 1, samples_each_loop);
    [~, idx_target] =  max(hyperplanes_target, [], 1);
    Idx_target_gpu(:, j) = idx_target;
end

idx_source = gather(Idx_source_gpu);
idx_target = gather(Idx_target_gpu);
M = sparse(idx_source, idx_target, 1, m, n);

end

