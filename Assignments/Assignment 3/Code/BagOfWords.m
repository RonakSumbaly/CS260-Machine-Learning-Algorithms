function BagOfWords()

% read the training files
spamDirInfo = dir('data/spam/train/spam');  
hamDirInfo = dir('data/spam/train/ham');

% get the file names of the training data
noFilesHam = size(hamDirInfo,1);
noFilesSpam = size(spamDirInfo,1);

% read the dictionary
dictionary = fileread('data/spam/dic.dat');
dicWords = strsplit(dictionary);
dicWords = dicWords(1:end-1)'; % remove the extra character

bagOfWords = zeros(size(dicWords,1),3);

%Calculate the feature vector for HAM
for i = 3:noFilesHam
   fileName = strcat('data/spam/train/ham/', hamDirInfo(i).name);
   fileContents = fileread(fileName);
   fileContents = lower(strsplit(fileContents,{' ','\f','\n','\r','\t','\v','.',',','?'}));
   fileContents = fileContents';
   
   for word = 1:length(fileContents)
        location = find(strcmp(dicWords,fileContents(word,1)));
        if isnumeric(location)
            bagOfWords(location,1) = bagOfWords(location,1) + 1;
        end     
   end
end

%Calculate the feature vector for SPAM
for i = 3:noFilesSpam
    fileName = strcat('data/spam/train/spam/', spamDirInfo(i).name);
   fileContents = fileread(fileName);
   fileContents = lower(strsplit(fileContents,{' ','\f','\n','\r','\t','\v','.',',','?'}));
   fileContents = fileContents';
   
   for word = 1:length(fileContents)
        location = find(strcmp(dicWords,fileContents(word,1)));
        if isnumeric(location)
            bagOfWords(location,2) = bagOfWords(location,2) + 1;
        end     
   end
end

bagOfWords(:,3) = bagOfWords(:,1) + bagOfWords(:,2);
[V,I] = sort(bagOfWords(:,3));
% print the last three maximum values
output = {char(dicWords(I(end),1)),V(end);char(dicWords(I(end-1),1)),V(end-1);char(dicWords(I(end-2),1)),V(end-2)};

disp(output)
end

