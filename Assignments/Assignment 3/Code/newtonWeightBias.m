function [weightsNewton,biasNewton] = newtonWeightBias(trainData, trainLabel,testORtrain)

eta = 0.01; %step size 
lambda = 0.05;
weightsNewton = zeros(size(trainData,2),1); % initially set to zero
biasNewton = 0.1;

for step = 1:6
    [cEntrValue, sigmaEntropyX,sigmaEntropy] = crossEntropyCalculate(biasNewton, weightsNewton, trainData, trainLabel,lambda);
    weightsNewton = weightsNewton + eta.*(sigmaEntropyX');
    biasNewton = biasNewton + eta*(sum(sigmaEntropy));       
    
end

end


function[cEntrValue, sigmaEntropyX, sigmaEntropy] = crossEntropyCalculate(bias, weights, trainData, trainLabel,lambda)
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

