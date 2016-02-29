function plotGradientUnReg(CrossEntropy,i)

f = figure(i);
if i == 2
    set(f,'name','Cross Entropy - Gradient Descent - UNRegularized - EmailSpam');
end
if i == 1
    set(f,'name','Cross Entropy - Gradient Descent - UNRegularized - Ionosphere');
end
if i == 3
    set(f,'name','Cross Entropy - Gradient Descent - Regularized - Ionosphere - Lambda - 0.1');
end
if i == 4 
    set(f,'name','Cross Entropy - Gradient Descent - Regularized - EmailSpam - Lambda - 0.1');
end
if i == 7
    set(f,'name','Cross Entropy - Newton Method - UNRegularized - IonoSpam');
end
if i == 8
    set(f,'name','Cross Entropy - Newton Method - UNRegularized - EmailSpam');
end

xlabel('Iterations');
ylabel('Cross-Entropy');

subplot(3,2,[1,2])
plot(CrossEntropy(:,1),'LineWidth',2);
xlabel('Iterations');
ylabel('Cross-Entropy');
legend('Step size = 0.001');

subplot(3,2,3)
plot(CrossEntropy(:,2),'LineWidth',2);
xlabel('Iterations');
ylabel('Cross-Entropy');
legend('Step size = 0.01');

subplot(3,2,4)
plot(CrossEntropy(:,3),'LineWidth',2);
xlabel('Iterations');
ylabel('Cross-Entropy');
legend('Step size = 0.05');

subplot(3,2,5)
plot(CrossEntropy(:,4),'LineWidth',2);
xlabel('Iterations');
ylabel('Cross-Entropy');
legend('Step size = 0.1');

subplot(3,2,6)
plot(CrossEntropy(:,5),'LineWidth',2);
xlabel('Iterations');
ylabel('Cross-Entropy');
legend('Step size = 0.5');




end