function sigma = compute_sigma(data, k, mode)
[n, d] = size(data);
dists = pdist2(data, data, 'euclidean');

sigma = zeros(n, 1);

if strcmp(mode, 'single')
    batch_size = 5000;
    if n < batch_size
        batch_size = n;
    end
    batch_num = ceil(n/batch_size);
    for i = 1 : batch_num
        index = [(i-1)*batch_size+1 : i*batch_size];
        dist = sort(dists(index,:),2);
        %         dist = dist(:, 1: k);
        %         sigma(index) = mean(dist, 2) / time;
        sigma(index) = dist(:, k) / 5;
    end
end

if strcmp(mode, 'whole')
    dists(dists==0) = max(max(dists));
    dist = min(min(dists));
    sigma(:) = dist / 10;
end

end

