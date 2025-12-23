function X = ida_designMatrix3(m, x1, x2, x3)
x1 = x1(:); x2 = x2(:); x3 = x3(:);
n = numel(x1);

switch m
    case 1
        X = [ones(n,1), x1, x2, x3];
    case 2
        X = [ones(n,1), x1, x2, x3, x1.^2, x2.^2, x3.^2];
    case 3
        X = [ones(n,1), x1, x2, x3, x1.^2, x2.^2, x3.^2, x1.*x2, x1.*x3, x2.*x3];
    case 4
        X = [ones(n,1), x1, x2, x3, x1.^2, x2.^2, x3.^2, x1.^3, x2.^3, x3.^3];
    case 5
        X = [ones(n,1), x1, x2, x3, ...
             x1.*x2, x1.*x3, x2.*x3, ...
             (x1.^2).*x2, x1.*(x2.^2), ...
             (x1.^2).*x3, x1.*(x3.^2), ...
             (x2.^2).*x3, x2.*(x3.^2)];
    otherwise
        error('Unknown model id');
end
end
