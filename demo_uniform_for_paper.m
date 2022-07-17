
load('data_1000x2000x20.mat');

m = size(source, 1);
n = size(target, 1);
d = size(source, 2);

% threshold of the semi-discrete OT
T = [0.6 0.5 0.4 0.3 0.2 0.1 0.06];

% save the results of OT plan
Pi = zeros(7, 15, m, n);

% save the results of OT cost
Cost = zeros(7,15);
Non_zeros = zeros(7,15);
Times = zeros(7,15);
for loop = 1 : length(T)
    %% compute omt
    fprintf('compute semi discrete OT ... \n');
    tic;
    source_heights = semi_discrete_ot(source, weights_source, T(loop), 1);
    target_heights = semi_discrete_ot(target, weights_target, T(loop), 1);
    
    %%
    fprintf('compute connectivity ... \n');
    M_record = compute_connectivity(source, target, source_heights, target_heights);
    t1 = toc;
    t2 = t1;
    
    % run with different number of clusters: ss
    for ss = 2:2:30
        P = M_record;
        fprintf('merge conncectivity: step %d \n', ss-1);
        
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
        
        Times(loop, ss/2) = t2-t_tmp+t1;
        if flag <=0
            Cost(loop, ss/2) = 0;
            Non_zeros(loop, ss/2) = 0;
        else
            P(:) = 0;
            P(M1>0) = p;
            fprintf('threshold: %.3f, clusters: %d, variable num: %d, nonzeros: %d, cost: %.10f, running time: %.6f\n', T(loop), ss, M_nonzeros, sum(sum(p~=0)), cost1, t2-t_tmp+t1);
            Pi(loop, ss/2, :, :) = P;
            Cost(loop, ss/2) = cost1 * data_max;
            Non_zeros(loop, ss/2) = M_nonzeros;
        end
    end
    
end












