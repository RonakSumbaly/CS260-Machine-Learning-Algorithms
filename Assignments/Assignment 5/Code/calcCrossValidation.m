function calcCrossValidation(trainData, trainLabel, testData,testLabel)

    noOfFolds = 5; % number of cross validation folds
    indices = crossvalind('Kfold', size(trainData, 1), noOfFolds); % generate random indices for 5 fold cross validation

    tradeOff = [4^-6,4^-5,4^-4,4^-3,4^-2,4^-1,1,4,4^2]; % test against various tradeOff parameter values

    accuracy = zeros(length(tradeOff),1); % accuracies for all tradeoff parameter C
    time = zeros(length(tradeOff),1); % time elasped for all tradeoff parameter C
   
    
    for C = 1:length(tradeOff) % loop through each value of tradeoff parameter C
        accuracyC = 0; % start accuracy 
        timeC = 0; % start time
        for CV = 1:noOfFolds % 5 fold cross validation
            bool = indices~=CV; % get only indices in the current cross validation set

            startTime = cputime;
            [w,b] = trainsvm(trainData(bool,:),trainLabel(bool),tradeOff(C)); % train data 
            endTime = cputime;

            timeC = endTime - startTime;
            accuracyC = accuracyC + testsvm(trainData(~bool,:),trainLabel(~bool),w,b);        
        end
        accuracy(C) = accuracyC/noOfFolds; % average accuracy over all folds
        time(C) = timeC/noOfFolds; % average time take over all folds
    end

    % get the maximum accuracy w.r.t tradeoff values
    [Value,Index] = max(accuracy); 
    bestTradeOff = tradeOff(Index);
    output = [accuracy,time,tradeOff'];
    disp('    Accuracy  Time    TradeOff Values');
    disp(output);
    disp(['TradeOff = ',num2str(bestTradeOff),' Max Accuracy = ',num2str(Value * 100),' Execution Time = ',num2str(time(Index))]);

    % train and test data w.r.t. the tradeoff value with maximum accuracy 
    [w,b] = trainsvm(trainData,trainLabel, bestTradeOff);        
    testAccuracy = testsvm(testData,testLabel,w,b);
    disp(['Test Accuracy using tradeoff = ',num2str(bestTradeOff),' is ',num2str(testAccuracy * 100)]);

end