function libSVM(trainData, trainLabel)

    tradeOff = [4^-6,4^-5,4^-4,4^-3,4^-2,4^-1,1,4,4^2]; % test against various tradeOff parameter values

    accuracy = zeros(length(tradeOff),1); % accuracies for all tradeoff parameter C
    time = zeros(length(tradeOff),1); % time elasped for all tradeoff parameter C

     for C = 1:length(tradeOff)
        startTime = cputime;
       % options = ['-q -s 0 -t 0 -v 5 -c',' ',num2str(tradeOff(C))];
        f = strcat({'-t 0 -c '}, num2str(tradeOff(C)), {' -v 5 -q'});
        accuracyC = svmtrain(trainLabel,trainData,f{1});
        endTime = cputime;
        
        accuracy(C) = accuracyC;
        time(C) = (endTime - startTime)/5;  
     end
     
    disp('');
    % get the maximum accuracy w.r.t tradeoff values
    [Value,Index] = max(accuracy); 
    bestTradeOff = tradeOff(Index);
    output = [accuracy,time,tradeOff'];
    disp('    Accuracy  Time    TradeOff Values');
    disp(output);
    disp(['TradeOff = ',num2str(bestTradeOff),' Max Accuracy = ',num2str(Value),' Execution Time = ',num2str(time(Index))]);
     
end
