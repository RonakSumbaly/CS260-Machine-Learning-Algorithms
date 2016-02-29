function [new_accu, train_accu] = knn_classify(train_data, train_label, new_data, new_label, k)

% k-nearest neighbor classifier
% Input:
%  train_data: N*D matrix, each row as a sample and each column as a
%  feature
%  train_label: N*1 vector, each row as a label
%  new_data: M*D matrix, each row as a sample and each column as a
%  feature
%  new_label: M*1 vector, each row as a label
%  k: number of nearest neighbors
%  All inputs were imported as cell-array format using inbuilt script
%  generated using Matlab
%
% Output:
%  new_accu: accuracy of classifying new_data
%  train_accu: accuracy of classifying train_data (using leave-one-out
%  strategy)
%
% CS260 2015 Fall, Homework 1
% 
% PLEASE NOTE: The problem was discussed with Manika Mittal and Yashaswi
% Alladi, however each of us implemented the solutions independently. 
%
% Author - Ronak Sumbaly {604591897}


% converting categorical features to binary features

    trainDataRows = size(train_data,1); % train data rows
    trainDataCols = size(train_data,2); % train data cols
    
    testDataRows = size(new_data,1); % test/valid data rows
    testDataCols = size(new_data,2); % test/valid data cols
    
    oneHotEncodedData = oneHotEncoding(); % one hot encoding performed

    binTrainData = oneHotEncodedData(1:trainDataRows,:)  ;  % binary features for training data
    binTestData = oneHotEncodedData(trainDataRows+1:end,:); % binary features for test data
    
    train_accu = leaveOneOutModel(binTrainData); % perform leave one out model
    new_accu = kNNModel(binTrainData, binTestData); % perform knn classification model
    
    disp('Train Accuracy:')
    disp(train_accu) % display train accuracy
    disp('Test/Valid Accuracy:')
    disp(new_accu) % display test accuracy
    
    function encodedData = oneHotEncoding() 
        % one hot encoding = categorical to binary features
        data = vertcat(train_data, new_data); % combine training and test data vertically
        encodedData = [];       
        for i = 1:size(data,2)
            encodedData = horzcat(encodedData,num2cell(dummyvar(categorical(data(:,i))))); % convert to binary form
        end   
    end    
    
    function hammingDistance = calcHammingDistance(inputData, inputData2)        
        % calculate hamming distance between the input instance and train/test data
        inputData = repmat(inputData, length(inputData2), 1);               
        hammingDistance = sum(abs(cell2mat(inputData)-cell2mat(inputData2)), 2);

    end

    function accuracy = getAccuracy(outputClass, outputLabel)
        % calculate the accuracy of the model
        accuracy = (sum(strcmp(outputClass,outputLabel))/length(outputClass))*100;
    end


    function trainAccuracy = leaveOneOutModel(trainData)
        % leave one out model
        outputLabel = cell(length(trainData),1); % predicted labels
        
        for loop = 1:length(trainData)
        
            testInstance = trainData(loop,:);
            hammingDistance = calcHammingDistance(testInstance, trainData);
            hammingDistance(loop,1) = 666; % disregard the test instance in test data - LOO
            [~, index] = sort(hammingDistance);
            outputLabel(loop,1) = findNearestNeighbor(index); % find nearest neighbor
        
        end
        
        trainAccuracy = getAccuracy(outputLabel, train_label); % compute LOO accuracy
         
    end

    function testAccuracy = kNNModel(trainData, testData)
        % k nearest neighbor classification model
        outputLabel = cell(length(testData),1); % predicted labels
        
        for loop = 1:length(testData)
            
            testInstance = testData(loop,:);
            hammingDistance = calcHammingDistance(testInstance, trainData);
            [~, index] = sort(hammingDistance);
            outputLabel(loop,1) = findNearestNeighbor(index); % find nearest neighbor
            
        end
        
        testAccuracy = getAccuracy(outputLabel, new_label);    % compute KNN accuracy           
            
    end

    function majOutputLabel = findNearestNeighbor(outputIndex)       
       % find k nearest neighbors
       classLabels = unique(train_label);
              
       if length(classLabels) < length(unique(new_label))
           classLabels = unique(new_label);
       end
       
       majorityCount = zeros(length(classLabels),1);
       
       for loop = 1:k
           index = find(strcmp(train_label(outputIndex(loop,1),1),classLabels));
           majorityCount(index,1) = majorityCount(index,1) + 1;   
       end 
       
       [~, I] = max(majorityCount); % neighbor with majority count
       
       majOutputLabel = classLabels(I,1); % predicted label
        
    end

end




    