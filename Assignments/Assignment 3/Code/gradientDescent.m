function [CrossEntropy, L2Norm, Weights] = gradientDescent(trainData, trainLabel)

    eta = [0.001, 0.01, 0.05, 0.1, 0.5];
    
    L2Norm = zeros(length(eta),2);
   
    CrossEntropy = zeros(length(eta),50); 
    
    for e = 1 : length(eta)
            weights = zeros(size(trainData,2),1); % initially set to zero
            bias = 0.1 ;
                for step = 1:50
                      [cEntrValue, sigmaEntropyX,sigmaEntropy] = crossEntropyCalculate(bias, weights, trainData, trainLabel);
                       weights = weights + eta(e).*(sigmaEntropyX');
                       bias = bias + eta(e)*(sum(sigmaEntropy));                   
                       CrossEntropy(e,step) = cEntrValue;
                end
                
            L2Norm(e,1) = eta(e);
            L2Norm(e,2) = norm(weights,2);
           
    end
    
    Weights = weights';

end

function[cEntrValue, sigmaEntropyX, sigmaEntropy] = crossEntropyCalculate(bias, weights, trainData, trainLabel)
    sigmaValue = sigma(bias+ (trainData)*(weights));
    cEntrValue = -(trainLabel.*log(sigmaValue) + (1-trainLabel).*log(1-sigmaValue));
    cEntrValue = sum(cEntrValue);
    sigmaEntropy = (trainLabel - sigmaValue);    
    sigmaEntropyX = (sigmaEntropy'* (trainData));
    
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




