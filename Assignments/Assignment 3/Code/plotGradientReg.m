function plotGradientReg(trainCE,testCE,i)

f = figure(i);
lambda = linspace(0,0.5,11);
if i == 6
    set(f,'name','Cross Entropy - Gradient Descent - Regularized - EmailSpam - Test + Train');
end
if i == 5
    set(f,'name','Cross Entropy - Gradient Descent - Regularized - Ionosphere - Test + Train');
end



xlabel('Lambda');
ylabel('Cross-Entropy');


subplot(3,2,[1,2])
plot(lambda,trainCE(1,:),'--', lambda, testCE(1,:), ':', 'LineWidth',2);
xlabel('Lambda');
ylabel('Cross-Entropy');
legend('Step size = 0.001');

subplot(3,2,3)
plot(lambda,trainCE(2,:),'--', lambda, testCE(2,:), ':', 'LineWidth',2);
xlabel('Iterations');
ylabel('Cross-Entropy');
legend('Train Data','Test Data');

subplot(3,2,4)
plot(lambda,trainCE(3,:),'--', lambda, testCE(3,:), ':', 'LineWidth',2);
xlabel('Iterations');
ylabel('Cross-Entropy');
legend('Train Data','Test Data');

subplot(3,2,5)
plot(lambda,trainCE(4,:),'--', lambda, testCE(4,:), ':', 'LineWidth',2);
xlabel('Iterations');
ylabel('Cross-Entropy');
legend('Train Data','Test Data');

subplot(3,2,6)
plot(lambda,trainCE(5,:),'--', lambda, testCE(5,:), ':', 'LineWidth',2);
xlabel('Iterations');
ylabel('Cross-Entropy');
legend('Train Data','Test Data');



end