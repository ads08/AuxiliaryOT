function h_hat = update_height(h, delta_h, mu)
    delta_h = delta_h - mean(delta_h);
    h_hat = h - mu * delta_h;
end