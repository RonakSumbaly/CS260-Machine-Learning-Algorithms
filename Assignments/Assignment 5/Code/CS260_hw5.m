function CS260_hw5()

% load training data
load('splice_train.mat');
trainLabel = label;
trainData = data;

% load testing data
load('splice_test.mat');
testLabel = label;
testData = data;

% calculate the mean and variance from the training data
meanTrainData = mean(trainData);
stdTrainData = std(trainData);

trainData = bsxfun(@rdivide,bsxfun(@minus,trainData,meanTrainData),stdTrainData);  % normalize the training data
testData = bsxfun(@rdivide,bsxfun(@minus,testData,meanTrainData),stdTrainData);  % normalize the testing data

meanTestData = mean(testData);
stdTestData = std(testData);
disp('*************** QUESTION 5.1 ****************');
disp('Mean & Variance of 3rd Feature in Testing Data');
disp('Mean:');
disp(meanTestData(3));
disp('Variance:');
disp(stdTestData(3));

disp('Mean & Variance of 10th Feature in Testing Data');
disp('Mean:');
disp(meanTestData(10));
disp('Variance:');
disp(stdTestData(10));

disp('*********************************************');

disp('*************** QUESTION 5.3 ****************');
calcCrossValidation(trainData, trainLabel, testData, testLabel);


disp('*********************************************');
disp('*************** QUESTION 5.4 - LIBSVM ****************');
% add LibSVM into path 
path = pwd;
addpath((strcat(path,'/libsvm-3.20/matlab')));

libSVM(trainData,trainLabel);


disp('*********************************************');
disp('************ QUESTION 5.5 - Kernel SVM ****************');

disp('');

nonLinearlibSVM(trainData,trainLabel,testData,testLabel);


end