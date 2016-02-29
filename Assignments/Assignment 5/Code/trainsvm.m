function [w,b] = trainsvm(train_data, train_label, C)
% Train linear SVM (primal form)
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  C: tradeoff parameter (on slack variable side)
%
% Output:
%  w: feature vector (column vector)
%  b: bias term
%

    [m, n] = size(train_data);

    H = diag([ones(n,1); zeros(m+1,1)]);
    f = [zeros(n+1,1); C*ones(m,1)];

    A = repmat(train_label,1,n+1) .* [train_data ones(m,1)];  
    A = -1 * [A eye(m)];
    b = -1 * ones(m, 1);

    lb = -[inf(n+1,1); zeros(m,1)];

    opts = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');

    [result, ~]= quadprog(H, f, A, b, [], [], lb, [],[],opts);

    w = result(1:n);
    b = result(n+1);


end