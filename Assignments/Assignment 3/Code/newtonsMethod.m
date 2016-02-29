function [] = newtonsMethod( trainData, trainLabel, testData, testLabel, name)

warning('off');

    [initialWeights,initialBias] = newtonWeightBias(trainData, trainLabel);

    % add intercept term to data
    trainData = [ones(size(trainData,1), 1), trainData];
    % with 1 bias
    weights = [initialBias,initialWeights']';
    
    crsEnt = zeros(50,1);
	
	for step = 1:50
        [crossEntropy, grad , Hessian] = getCrossEntropy(weights, trainData,trainLabel);
        weights = weights - (pinv(Hessian) * grad);
        crsEnt(step) = crossEntropy;
    end
    
    disp(strcat(name, ': Newtons Method: No Regularization - Training'));
   
    disp(strcat('L2 Norm = ',num2str(norm(weights,2))));
    disp(strcat('CrossEntropy = ',num2str(crsEnt(50))))
    
    if strcmp(name,'Ionosphere')
        figure(7)
    else
        figure(8)
    end
    plot(crsEnt', 'linewidth',2, 'color','R');
    xlabel('Iterations');
    ylabel('CrossEntropy');
    title(strcat(name,': Newtons Method: No Regularization')); 

    crsEnt = zeros(50,1);  
    
    testData = [ones(size(testData,1), 1), testData];
    for step = 1: 50
       [crossEntropy, grad , Hessian] = getCrossEntropy(weights, testData,testLabel);
        weights = weights - (pinv(Hessian) * grad);
        crsEnt(step) = crossEntropy; 
    end
    
    disp(strcat(name, ': Newtons Method: No Regularization - Testing'));
    disp(strcat('CrossEntropy = ',num2str(crsEnt(50))))
    
    
    
% ----------------------------------------------------------
% With Regularization

    
    lambda = linspace(0,.5,11);
    L2normR = zeros(length(lambda),2);
    CrossEntrR = zeros(length(lambda),2);
    crsEnt = zeros(length(lambda),50);
    for lamb=1:length(lambda)
        step = 1;
        weights = [initialBias,initialWeights']';
        while step<51
            [crossEntropy, grad , Hessian] = getCrossEntropyRegularized(weights, trainData,trainLabel,lambda(lamb));
            weights = weights - (pinv(Hessian) * grad);
            crsEnt(lamb,step) = crossEntropy;
            step = step+1;
        end
        L2normR(lamb,1) = lambda(lamb);
        L2normR(lamb,2) = norm(weights,2);
        CrossEntrR(lamb,1) = lambda(lamb);
        CrossEntrR(lamb,2) = crsEnt(lamb,50);
      
    end
    disp('Newtons Regularized L2 Norm');
    disp(L2normR);
    
    disp('Newtons Regularized Cross Entropy');
    disp(CrossEntrR);
    
    
if strcmp(name,'Ionosphere')
        figure(9)
    else
        figure(10)
end
plot(crsEnt(:,:)','linewidth',2);   
legendCell = cellstr(num2str(lambda', '%-f'));
legend(legendCell);
xlabel('Iterations');
ylabel('CrossEntropy');
title(strcat(name,': Newtons Method: With Regularization at different lambdas')); 

% ----------------------------------------------------------

warning('on');
end


function [crossEntropy, grad , Hessian] = getCrossEntropy(w, x, y)

    hypothesis = sigmoid(x*w);     
    crossEntropy = -(y.*log(hypothesis) + (1-y).*log(1-hypothesis));
    crossEntropy = sum(crossEntropy);

    grad = x'* (hypothesis-y);
    Hessian = x' * diag(hypothesis) * diag(1-hypothesis) * x;
end


function [crossEntropy, grad , Hessian] = getCrossEntropyRegularized(w, x, y, lambda)

    hypothesis = sigmoid(x*w);     
    crossEntropy = -(y.*log(hypothesis) + (1-y).*log(1-hypothesis));
    crossEntropy = sum(crossEntropy) + lambda*norm(w(2:length(w)))^2;

    G = 2*lambda.*w; 
    G(1) = 0;
    H = 2*lambda.*eye(size(x,2)); 
    H(1) = 0;
    grad = (x' * (hypothesis-y)) + G;
    Hessian = (x' * diag(hypothesis) * diag(1-hypothesis) * x) + H;
end


function value = sigmoid(inp)

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
