function m = ida_mask_finite(y, X)
y = y(:);
m = isfinite(y) & builtin('all', isfinite(X), 2);
end
