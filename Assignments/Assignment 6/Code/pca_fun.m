function eigenvecs = pca_fun(X, d)

% Implementation of PCA
% input:
%   X - N*D data matrix, each row as a data sample
%   d - target dimensionality, d <= D
% output:
%   eigenvecs: D*d matrix
%
% usage:
%   eigenvecs = pca_fun(X, d);
%   projection = X*eigenvecs;
%

X = X - repmat(mean(X), size(X,1),1);
[eigenVector, D] = eig(X' * X);
[~, I]=sort(diag(D),'descend');
eigenvecs = eigenVector(:,I(1:d));