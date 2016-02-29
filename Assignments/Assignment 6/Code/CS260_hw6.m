% add LibSVM into path 
path = pwd;
addpath((strcat(path,'/libsvm-3.20/matlab')));

load('face_data.mat')

disp('Eigen Faces')
% loop through each person's ID
for i = 1:length(personID)
    [m, n] = size(image{i}); % all of size 50 x 50
    imageData(i, :) = reshape(image{i}, [], 1); % vectorize into 2500 dimensional vector
end

eigenVectors = pca_fun(imageData, 200);

figure(1); clf; set(gcf, 'Name', 'EigenFaces');
for d=1:5
    subplot(2,3,d);
    imshow(reshape(eigenVectors(:,d),50,50),[]); % display top 5 eigen faces
    title(num2str(d));
end
drawnow;


d = [20, 25, 50, 100, 200];
% Considering C values from the last assignment
C = [4^-6,4^-5,4^-4,4^-3,4^-2,4^-1,1,4,4^2];
output = zeros(1,9);
maxAccuracy = 0;
maxC = 0;
for D = 1 : length(d)
    noOfEigenVectors = d(D);
    eigenVectors = pca_fun(imageData, noOfEigenVectors);
    modifiedImageData = imageData * eigenVectors;
    
    for j = unique(subsetID)
        
        testData = modifiedImageData(subsetID == j,:);
        testLabel = personID(subsetID == j)';
        
        trainData = modifiedImageData(subsetID ~= j,:);
        trainLabel = personID(subsetID ~= j)';
        
        % parameter tuning using leave one out approach
        for c = 1 : length(C) 
            linearFeature = strcat({'-t 0 -c '}, num2str(C(c)), {' -s 0 -q'});
            model = svmtrain((trainLabel), double(trainData), linearFeature{1});
            result = svmpredict((testLabel), double(testData),model, '-q');
            accuracy(j,c) = sum(result==testLabel)/length(testLabel);            
        end
    end
    
    average = (sum(accuracy,1) / 5);
    [M,I] = max(average);
    if maxAccuracy < M
        maxAccuracy = M;
        maxC = C(I);
    end
    output = [output;average]; 
end

output(1,:)=[]; % delete first row
disp([[0;d'] [C;output]]) % display final output
disp('Max Accuracy')
maxAccuracy
disp('Best Parameter')
maxC
