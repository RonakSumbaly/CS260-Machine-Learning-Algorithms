function [CrossEntropy, L2Norm, Weights,CrossEntropyTrain,CrossEntropyTest] = gradientDescentReg(trainData, trainLabel,testORtrain)

    format short g
    eta = [0.001, 0.01, 0.05, 0.1, 0.5]; %step size 
    lambda = linspace(0,0.5,11);
   
    CrossEntropy = zeros(length(eta),50); 
    L2Norm = zeros(length(lambda),2);
    CrossEntropyTrain = zeros(length(eta),length(lambda),1);
    CrossEntropyTest = zeros(length(eta),length(lambda),1);

    
    
    for e = 1 : length(eta)
         for lamb = 1 : length(lambda)
            weights = zeros(size(trainData,2),1); % initially set to zero
            bias = 0.1 ;
                for step = 1:50
                      [cEntrValue, sigmaEntropyX,sigmaEntropy] = crossEntropyUnregular(bias, weights, trainData, trainLabel,lambda(lamb));  
                       weights = weights + eta(e).*(sigmaEntropyX');
                       bias = bias + eta(e)*(sum(sigmaEntropy));                   
                   if lambda(lamb) == 0.1
                        CrossEntropy(e,step) = cEntrValue;
                   end
          
                end            
                CrossEntropyTrain(e,lamb,1) = cEntrValue;
                if testORtrain == 1
                    weightTest = weights ;
                    for step = 1:50
                        [cEntrValue, sigmaEntropyX,sigmaEntropy] = crossEntropyUnregular(bias, weightTest, trainData, trainLabel,lambda(lamb));  
                         weightTest = weightTest + eta(e).*(sigmaEntropyX');
                         bias = bias + eta(e)*(sum(sigmaEntropy));  
                       
                    end
                    CrossEntropyTest(e,lamb,1) = cEntrValue;
                end 
          if eta(e) == 0.01
            L2Norm(lamb,1) = lambda(lamb);
            L2Norm(lamb,2) = norm(weights,2);
         end
         end  
    end
    Weights = weights';     
end

function[cEntrValue, sigmaEntropyX, sigmaEntropy] = crossEntropyUnregular(bias, weights, trainData, trainLabel,lambda)
    sigmaValue = sigma(bias+ (trainData)*(weights));
    cEntrValue = -(trainLabel.*log(sigmaValue) + (1-trainLabel).*log(1-sigmaValue));
    cEntrValue = sum(cEntrValue);
    cEntrValue = cEntrValue + (lambda * (weights' * weights));
 
    sigmaEntropy = (trainLabel - sigmaValue);    
    sigmaEntropyX = (sigmaEntropy'* (trainData));
    sigmaEntropyX = sigmaEntropyX - (2* lambda *(weights'));
    
end

function value = sigma(inp)

value = 1.0 ./ ( 1.0 + exp(-inp) );
for i=1:length(value)
    if value(i)< 1e-16 
        value(i) = 1e-16;
    else if 1 - value(i) < 1e-16 
            value(i) = 1 - 1e-16;
        end
    end
end

end




