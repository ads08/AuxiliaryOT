function y = gaussian_sampling(mu, sigma, n, d)
    x = randn(d, n, "gpuArray");
    y = x * sigma + mu;
end

