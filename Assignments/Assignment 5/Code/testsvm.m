function accu = testsvm(test_data, test_label, w, b)
% Test linear SVM 
% Input:
%  test_data: M*D matrix, each row as a sample and each column as a
%  feature
%  test_label: M*1 vector, each row as a label
%  w: feature vector 
%  b: bias term
%
% Output:
%  accu: test accuracy (between [0, 1])
%
% linear model formula
    predictionValue = w' * test_data' + b;

    predictedLabel = zeros(length(predictionValue),1);
    predictedLabel(predictionValue >= 0) = 1;
    predictedLabel(predictionValue < 0) = -1;

    accu = sum(predictedLabel == test_label) / length(test_label);


end