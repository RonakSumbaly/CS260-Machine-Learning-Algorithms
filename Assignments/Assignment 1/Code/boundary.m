% Input : 
%       features: N*2 matrix, each row as a sample and each column as a features
%       labels: N*1 matrix, each row as a label {1,-1}
%       k : number of nearest neighbor
%
% Output : Graph depicting the predicted class for 10000 random point between [0,1]
%
% CS260 2015 Fall, Homework 1
% 
% PLEASE NOTE: The problem was discussed with Manika Mittal and Yashaswi
% Alladi, however each of us implemented the solutions independently. 
%
% Author - Ronak Sumbaly {604591897}

function boundary(features, label, k)

    testData = combvec(linspace(0,1,100),linspace(0,1,100));    % create 10000 random points
    color = zeros(size(testData,2),1);
    
    for loop = 1:size(testData,2) % loop through each test case
        
        testInstance = testData(:,loop)';                
        euclideanDistance = calcEuclideanDistance(testInstance);  % calculate euclidian distance    
        [~, index] = sort(euclideanDistance);
        outputLabel = findNearestNeighbor(index);    % find k nearest neighbor            
        if outputLabel == 1      
            color(loop,1) = 10;
        else
            color(loop,1) = 100;
        end
    end
    
    scatter(testData(1,:),testData(2,:),10,color,'fill');   % make plot for all test points

    function distance = calcEuclideanDistance(testInstance)
        
        inputData = repmat(testInstance, length(features), 1); 
        distance = sqrt(sum((inputData-features).^2,2));
    
    end

    function majOutputLabel = findNearestNeighbor(outputIndex)       
                     
       majorityCount = zeros(2,1);
              
       for i = 1:k
            if(label(outputIndex(i,1),1)) == 1
                majorityCount(1,1) = majorityCount(1,1) + 1;
            else
                majorityCount(2,1) = majorityCount(2,1) + 1;
            end   
       end 
                          
       [~, I] = max(majorityCount);
       
       if(I == 1)  majOutputLabel = 1;
       else        majOutputLabel = -1;
       end

    end

    end