%%%%%% QUESTION 1. %%%%%%
disp('----------------------------------');

disp('Question 1. Most Frequent 3 words');
BagOfWords();
disp('----------------------------------');


%%%%%% QUESTION 3. %%%%%%

% import spam data - test & training
[spamTrainData, spamTrainLabel, spamTestData, spamTestLabel] = loadDataFile('spam');

% import ionosphere data - test & training
[ionoTrainData, ionoTrainLabel, ionoTestData, ionoTestLabel] = loadDataFile('iono');


disp('Question 3. B. Batch Gradient Descent - Without Regularization - L2 Norm');
disp('----------------------------------');
disp('Ionosphere Dataset');
[ionoCrossEntropyGD1, ionoL2NormGD1, ionoWeightsGD1] = gradientDescent(ionoTrainData,ionoTrainLabel);
disp('      Eta    L2Norm');
disp(ionoL2NormGD1)

plotIonoCrossEntropy = transpose(reshape(ionoCrossEntropyGD1(:,:,1),  5, 50));

plotGradientUnReg(plotIonoCrossEntropy, 1);

disp('----------------------------------');
disp('EmailSpam Dataset');
[spamCrossEntropyGD1, spamL2NormGD1, spamWeightsGD1] = gradientDescent(spamTrainData,spamTrainLabel);
disp('      Eta    L2Norm');
disp(spamL2NormGD1);
disp('----------------------------------');

plotSpamCrossEntropy = transpose(reshape(spamCrossEntropyGD1(:,:,1),  5, 50));
plotGradientUnReg(plotSpamCrossEntropy, 2);


%%%%%% QUESTION 4. A %%%%%%

[ionoCrossEntropyGD2, ionoL2NormGD2, ionoWeightsGD2] = gradientDescentReg(ionoTrainData,ionoTrainLabel,0);
[spamCrossEntropyGD2, spamL2NormGD2, spamWeightsGD2] = gradientDescentReg(spamTrainData,spamTrainLabel,0);

plotIonoCrossEntropy2 = transpose(reshape(ionoCrossEntropyGD2(:,:,1),  5, 50));
plotSpamCrossEntropy2 = transpose(reshape(spamCrossEntropyGD2(:,:,1),  5, 50));

plotGradientUnReg(plotIonoCrossEntropy2, 3);
plotGradientUnReg(plotSpamCrossEntropy2, 4);


%%%%%% QUESTION 4. B %%%%%%
disp('Question 4. B. Batch Gradient Descent - Regularization - L2 Norm - eta = 0.01');
disp('----------------------------------');
disp('Ionosphere Dataset');
[ionoCrossEntropyGD3, ionoL2NormGD3, ionoWeightsGD3] = gradientDescentReg(ionoTrainData,ionoTrainLabel,0);
disp('      Lambda        L2Norm');
disp(ionoL2NormGD3);

disp('----------------------------------');
disp('EmailSpam Dataset');
[spamCrossEntropyGD3, spamL2NormGD3, spamWeightsGD3] = gradientDescentReg(spamTrainData,spamTrainLabel,0);
disp('      Lambda        L2Norm');
disp(spamL2NormGD3);
disp('----------------------------------');


%%%%%% QUESTION 4. C %%%%%%

[ionoCrossEntropyGD4, ionoL2NormGD4, ionoWeightsGD4,ionoCrossEntropyTrainGD4, ionoCrossEntropyTestGD4] = gradientDescentReg(ionoTestData,ionoTestLabel,1);
[spamCrossEntropyGD4, spamL2NormGD4, spamWeightsGD4,spamCrossEntropyTrainGD4, spamCrossEntropyTestGD4] = gradientDescentReg(spamTestData,spamTestLabel,1);

plotGradientReg(ionoCrossEntropyTrainGD4,ionoCrossEntropyTestGD4,5);
plotGradientReg(spamCrossEntropyTrainGD4,spamCrossEntropyTestGD4,6);

%%%%%% QUESTION 6. %%%%%%

disp('Question 6. Newton Method');
disp('----------------------------------');
disp('Ionosphere Dataset');
newtonsMethod(ionoTrainData,ionoTrainLabel,ionoTestData,ionoTestLabel,'Ionosphere');

disp('----------------------------------');
disp('EmailSpam Dataset');
newtonsMethod(spamTrainData,spamTrainLabel,spamTestData,spamTestLabel,'EmailSpam');





