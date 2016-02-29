function nonLinearlibSVM(trainData, trainLabel, testData, testLabel)

    tradeOff = [4^-4,4^-3,4^-2,4^-1,1,4,4^2,4^3,4^4,4^5,4^6,4^7]; % test against various tradeOff parameter values
   
    degree = [1,2,3];
    
    accuracy = zeros(length(tradeOff),3,1); % accuracies for all tradeoff parameter C
    time = zeros(length(tradeOff),1); % time elasped for all tradeoff parameter C
    
    
    
    for C = 1:length(tradeOff) % loop through each value of tradeoff parameter C
        for D = 1:length(degree) % loop through
            startTime = cputime;
            
            f = strcat({'-q -s 0 -t 1 -v 5 -c '}, num2str(tradeOff(C)), {' -d '}, num2str(degree(D)));
            accuracyC = svmtrain(trainLabel,trainData,f{1});

            endTime = cputime;
            accuracy(C,D,1) = accuracyC;
            time(C,D,1) = (endTime-startTime);
        end
        
    end
    disp('*************** Polynomial Kernel ****************')
    disp('TradeOff   Accuracy   Time');
    %output = [accuracy(:,1),time(:,1),accuracy(:,2),time(:,2),accuracy(:,3),time(:,3)];
    %disp(output);
   
    disp([tradeOff',sum(accuracy,2)/length(degree),sum(time,2)/length(degree)])
    
    disp(['Total Time taken by Polynomial Kernel = ',num2str(sum(sum(time)))]); 

    [maxAccuracy, index] = max(accuracy(:));
    
    [n,m] = ind2sub(size(accuracy),index);

    bestCoefficient = tradeOff(n);
    bestDegree = degree(m);
    
    disp(['Best Coefficient = ',num2str(bestCoefficient),' Max Accuracy = ',num2str(maxAccuracy),' Best Degree = ',num2str(bestDegree)]);

    f = strcat({'-q -s 0 -t 1 -c '}, num2str(bestCoefficient), {' -d '}, num2str(bestDegree));
    model = svmtrain(trainLabel,trainData,f{1});
    result = svmpredict(testLabel,testData,model);
    testAccuracy = sum(result == testLabel) / length(testLabel);
    
    disp(['Polynomial Kernel: Test Accuracy using TradeOff = ',num2str(bestCoefficient),' and degree = ',num2str(bestDegree),' is ',num2str(testAccuracy*100)]);
        
    
    disp('*************** RBF Kernel ****************')
    
    
    gamma = [4^-7,4^-6,4^-5,4^-4,4^-3,4^-2,4^-1,4^0,4^1,4^2];
    
    accuracy = zeros(length(tradeOff),length(gamma));
    time = zeros(length(tradeOff),length(gamma));
    
    for C = 1:length(tradeOff) % loop through each value of tradeoff parameter C
        for G = 1:length(gamma) % loop through
            startTime = cputime;
            
            f = strcat({'-q -s 0 -t 1 -v 5 -c '}, num2str(tradeOff(C)), {' -g '}, num2str(gamma(G)));
            accuracyC = svmtrain(trainLabel,trainData,f{1});
            endTime = cputime;
            accuracy(C,G,1) = accuracyC;
            time(C,G,1) = (endTime-startTime);
        end
    end
    
    disp('*************** RBF Kernel ****************')
    disp('TradeOff   Accuracy   Time');
    disp([tradeOff',sum(accuracy,2)/length(gamma),sum(time,2)/length(gamma)]) 
    % disp('Accuracy1  Accuracy2  Accuracy3 Accuracy4 Accuracy5 Accuracy6  Accuracy7  Accuracy8  Accuracy9  Accuracy10');
   % disp(accuracy)
    
   % disp('Time1         Time2     Time3       Time4    Time5    Time6      Time7    Time8    Time9     Time10');
   % disp(time)
    
     disp(['Total Time taken by RBF Kernel = ',num2str(sum(sum(time)))]); 

    [maxAccuracy, index] = max(accuracy(:));
    
    [n,m] = ind2sub(size(accuracy),index);

    bestCoefficient = tradeOff(n);
    bestGamma = gamma(m);
    
    disp(['Best Coefficient = ',num2str(bestCoefficient),' Max Accuracy = ',num2str(maxAccuracy),' Best Gamma = ',num2str(bestGamma)]);

    f = strcat({'-q -s 0 -t 1 -c '}, num2str(bestCoefficient), {' -g '}, num2str(bestGamma));
    model = svmtrain(trainLabel,trainData,f{1});
    result = svmpredict(testLabel,testData,model);
    testAccuracy = sum(result == testLabel) / length(testLabel);
    
    disp(['RBF Kernel: Test Accuracy using TradeOff = ',num2str(bestCoefficient),' and gamma = ',num2str(bestGamma),' is ',num2str(testAccuracy*100)]);

     
     
end