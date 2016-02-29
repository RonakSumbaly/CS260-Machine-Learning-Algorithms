function[trainData, trainLabel,testData,testLabel] = loadDataFile(fileName)

if strcmp(fileName, 'iono')
    trainDataRaw = importdata('data/ionosphere_final/ionosphere_train.dat.final');
    testDataRaw = importdata('data/ionosphere_final/ionosphere_test.dat.final');
    
    trainData = trainDataRaw(:,1:end-1);
    trainLabel = trainDataRaw(:,end);
    
    testData = testDataRaw(:,1:end-1);
    testLabel = testDataRaw(:,end);  

else
    
    trainDataSpam = importdata('data/spam_final/train/spam/train_spam.final');
    trainDataHam = importdata('data/spam_final/train/ham/train_ham.final');
    
    trainData = [trainDataSpam; trainDataHam];
    trainLabel = [ones(size(trainDataSpam, 1),1); zeros(size(trainDataHam, 1),1)];
    
    testDataSpam = importdata('data/spam_final/test/spam/test_spam.final');
    testDataHam = importdata('data/spam_final/test/ham/test_ham.final');
    
    testData = [testDataSpam; testDataHam];
    testLabel = [ones(size(testDataSpam, 1),1); zeros(size(testDataHam, 1),1)];
    
end



end