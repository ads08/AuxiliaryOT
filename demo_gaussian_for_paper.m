
load('data_1000x2000x20.mat');
m = size(source, 1);
n = size(target, 1);
d = size(source, 2);

% threshold of the semi-discrete OT
T = [0.6 0.5 0.4 0.3 0.2 0.1 0.06];

% save the results of OT plan
Pi = zeros(7, 39, m, n);

% save the results of OT cost
Cost = zeros(7,39);
Non_zeros = zeros(7,39);
Times = zeros(7,39);

for loop = 4 : length(T)
    fprintf('compute semi discrete OT ... \n');
    tic;
    %% compute omt
    sigma = compute_sigma(double(source), 100, 'single');
    gm{1} = gpuArray(source');
    gm{2} = gpuArray(sigma);
    gm{3} = weights_source;
    
    sigma1 = double(reshape(repmat(sigma',d,1), 1, d, m));
    gm1 = gmdistribution(double(source), sigma1, weights_source);
    
    [target_heights, samples_per_cell] = semi_discrete_ot_gm_(gm1, target, weights_target, T(loop));
    %%
    target_gpu = gpuArray(target);
    target_heights_gpu = gpuArray(target_heights);
    
    % find the connectivity matrix
    samples_each_loop = m;
    sample_loops = ceil(samples_per_cell * n / samples_each_loop);
    Idx_source = zeros(samples_each_loop, sample_loops);
    Idx_target_gpu = gpuArray(zeros(samples_each_loop, sample_loops));
    for j = 1 : sample_loops
        %             x = random(gm1, samples_each_loop);
        % %             % compute source index
        %             dists_source = pdist2(x, source);
        %             [~, I] = sort(dists_source, 2);
        %             Idx_source(:, j) = I(:,1);
        %
        %             % compute target
        %             x = x';
        %             x = gpuArray(single(x));
        
        % compute source index
        x = gaussian_sampling(gm{1}, gm{2}, samples_each_loop, d);
        dists_source = pdist2(gather(x'), source);
        [~, I] = sort(dists_source, 2);
        Idx_source(:, j) = I(:,1);
        
        hyperplanes_target = target_gpu*x + repmat(target_heights_gpu, 1, samples_each_loop);
        [~, idx_target] =  max(hyperplanes_target, [], 1);
        Idx_target_gpu(:, j) = idx_target;
    end
    
    idx_source = gather(Idx_source);
    idx_target = gather(Idx_target_gpu);
    M_record = sparse(idx_source, idx_target, 1, m, n);
    
    
    t1 = toc;
    t2 = t1;
    %%
    for ss = 14:40
        fprintf('merge conncectivity: step %d \n', ss);
        P = M_record~=0;
        M = P;
        M1 = M~=0;
        M3 = M1;
        
        M2 = M3;
        Idx_s = knnsearch(source, source, 'K', ss);
        Idx_t = knnsearch(target, target, 'K', ss);
        for i = 1 : m
            M2(i,:) = sum(M1(Idx_s(i,:), :), 1);
        end
        for j = 1 : n
            M2(:,j) = M2(:,j) + sum(M1(:, Idx_t(j,:)), 2);
        end
        M2 = M2 + P;
        M1 = M2~=0;
        
        A = sparsity_to_constraints(M1);
        b = [weights_source; weights_target];
        D1 = D(M1>0);
        M_nonzeros = size(A, 2);
        
        fprintf('linear programming begins ... \n');
        [x, cost1, flag] = linprog(double(D1), [], [], A, b, zeros(M_nonzeros, 1), ones(M_nonzeros, 1));
        p = x;
        
        t_tmp = t2;
        t2 = toc;
        
        Times(loop, ss-1) = t2-t_tmp+t1;
        if flag <=0
            Cost(loop, ss-1) = 0;
            Non_zeros(loop, ss-1) = 0;
        else
            P(:) = 0;
            P(M1>0) = p;
            fprintf('%.3f %d %d %d %.10f %.6f\n', T(loop), ss, M_nonzeros, sum(sum(p~=0)), cost1, t2-t_tmp+t1);
            Pi(loop, ss-1, :, :) = P;
            Cost(loop, ss-1) = cost1 * data_max;
            Non_zeros(loop, ss-1) = M_nonzeros;
        end
        
    end
end




















