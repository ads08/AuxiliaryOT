function heights = heights_initialize_box(data)
    [n,d] = size(data);
    heights_tmp = zeros(n,d);
    heights_tmp(:,1) = 1.0;
    for i = 1 : d
        data_1d = data(:, i);
        [data_sorted, idx] = sort(data_1d);
        for j = 2 : n
            heights_tmp(j, i) = heights_tmp(j-1, i) ...
                                +(data_sorted(j-1)-data_sorted(j))*(j-1)/n;
        end
        heights_tmp(:, i) = heights_tmp(:, i) - mean(heights_tmp(:, i));
        heights_tmp(idx, i) = heights_tmp(:, i);
    end
    heights = sum(heights_tmp, 2);
end