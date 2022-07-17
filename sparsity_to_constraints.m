function Aeq = sparsity_to_constraints(M)
[m, n] = size(M);
M_nonzeros = nnz(M);
K = find(M);
M(K) = K;
[ind_mat_I, ind_mat_J] = ind2sub([m,n], K);

ind_i = zeros(2*M_nonzeros, 1);
ind_j = zeros(2*M_nonzeros, 1);
start = 1;
for i = 1 : m
    ind = find(ind_mat_I==i);
    num = length(ind);
    ind_i(start:start+num-1) = i;
    ind_j(start:start+num-1) = ind;
    start = start + num;
end

for j = 1 : n
    ind = find(ind_mat_J==j);
    num = length(ind);
    ind_i(start:start+num-1) = m+j;
    ind_j(start:start+num-1) = ind;
    start = start + num;
end

Aeq = sparse(ind_i, ind_j, ones(2*M_nonzeros, 1), m+n, M_nonzeros);