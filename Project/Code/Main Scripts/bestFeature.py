import itertools

from config import conf
import classifications
from csvToArray import featureArray
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC


warnings.filterwarnings("ignore")

trainData, trainLabel = featureArray(conf['train']['feature_vector'])
validationData, validationLabel = featureArray(conf['valid']['feature_vector'])
testData, testLabel = featureArray(conf['test']['feature_vector'])


def runClassifier(classifier, trainData,trainLabel, testData, testLabel, bestParameters):
    if classifier[0] == 'KNN':
        neighTest = KNeighborsClassifier(n_neighbors=int(bestParameters['KNN'][0]), algorithm='auto', p=2,weights=bestParameters['KNN'][1])
        neighTest.fit(trainData, trainLabel)
        scoreTest = neighTest.score(testData, testLabel)
        return scoreTest - classifier[1]
    elif classifier[0] == 'Random Forests':
        neighTest = RandomForestClassifier(n_estimators = int(bestParameters['Random Forests'][0]),criterion=bestParameters['Random Forests'][1])
        neighTest.fit(trainData, trainLabel)
        scoreTest = neighTest.score(testData, testLabel)
        return scoreTest - classifier[1]
    elif classifier[0] == 'Linear Nu-SVC':
        clf = NuSVC(kernel="linear")
        clf.fit(trainData, trainLabel)
        scoreTest = clf.score(testData, testLabel)
        return scoreTest - classifier[1]
    elif classifier[0] == 'RBF Nu-SVC':
        clfFinal = NuSVC(gamma = bestParameters['RBF Nu-SVC'])
        clfFinal.fit(trainData,trainLabel)
        score = clfFinal.score(testData,testLabel)
        return score - classifier[1]
    elif classifier[0] == 'Gradient Boosting':
        neighTest = GradientBoostingClassifier(n_estimators = int(bestParameters['Gradient Boosting'][0]),loss='deviance')
        neighTest.fit(trainData, trainLabel)
        scoreTest = neighTest.score(testData, testLabel)
        return scoreTest - classifier[1]
    elif classifier[0] == 'Multinomial Naive Bayes':
        clfTest = MultinomialNB(alpha = bestParameters['Multinomial Naive Bayes'], fit_prior=True)
        clfTest.fit(trainData, trainLabel)
        scoreTest = clfTest.score(testData, testLabel)
        return scoreTest - classifier[1]
    elif classifier[0] == 'Decision (IG)':
        clf = tree.DecisionTreeClassifier(criterion='entropy')
        clf.fit(trainData, trainLabel)
        scoreTest = clf.score(testData, testLabel)
        return scoreTest - classifier[1]


def plotScoreParameters(scoreOfParameters):
    plotFeatures = range(trainData.shape[1])
    plt.plot(plotFeatures,scoreOfParameters, marker = 'o', linestyle = '--', color = 'r')

    plt.xlabel('Feature Number')
    plt.ylabel('Accuracy Difference')
    plt.title('Accuracy Importance for Top Classifiers')

    plt.legend()
    plt.show()




def selectFeature(accuracy , bestParameters):
    temp = []
    accuracies = []
    for value in xrange(5):
        currentClassifier = accuracy[len(accuracy) - value - 1]
        for features in xrange(trainData.shape[1]):
            print features
            list = range(trainData.shape[1])
            list.remove(features)
            temp.append(runClassifier(currentClassifier, trainData[:,list],trainLabel, testData[:,list], testLabel, bestParameters))
        accuracies.append(temp)
        print "Done"
        temp = []

    print len(accuracies)

    plotFeatures = range(trainData.shape[1])

    plt.plot(plotFeatures, accuracies[0], label = accuracy[len(accuracy) - 1][0],color = 'b')
    plt.plot(plotFeatures, accuracies[1], label = accuracy[len(accuracy) - 2][0], marker='o', linestyle='--', color='y')
    plt.plot(plotFeatures, accuracies[2], label = accuracy[len(accuracy) - 3][0], marker = 'o', linestyle='--', color='r')
    plt.plot(plotFeatures, accuracies[3], label = accuracy[len(accuracy) - 4][0], marker='o', linestyle='-', color='b')
    plt.plot(plotFeatures, accuracies[4], label = accuracy[len(accuracy) - 5][0], marker='o', linestyle='-', color='g')

    print accuracies[0]
    print accuracies[1]
    print accuracies[2]
    print accuracies[3]
    print accuracies[4]

    print "SUM :"
    scoreOfParameters = [sum(i) for i in itertools.izip_longest(*accuracies, fillvalue=0)]

    print scoreOfParameters
    plt.legend()
    plt.show()

    plotScoreParameters(scoreOfParameters)
'''

knn = []
randomForest = []
decisionGini = []
decisionIG =[]
gaussianNB = []
sigmoid = []
gradientBoosting = []
logisticRegression = []
radiusNeighbor = []
polynomialSVC = []
RBFSVC = []
mnb = []
linearSVC = []

for features in xrange(trainData.shape[1]):
    list = range(trainData.shape[1])
    list.remove(features)
    print features
    knn.append(classifications.knnClassifier(trainData[:,list],trainLabel,validationData[:,list], validationLabel, testData[:,list], testLabel) - knnAccuracy)
#    randomForest.append(classifications.randomForestClassify(trainData[:,list],trainLabel,validationData[:,list], validationLabel, testData[:,list], testLabel))
    linearSVC.append(classifications.linearNuSVC(trainData[:,list],trainLabel,validationData[:,list], validationLabel, testData[:,list], testLabel) - linearSVCAccuracy)
    decision = classifications.decisionTreeClassifier(trainData[:,list],trainLabel,validationData[:,list], validationLabel, testData[:,list], testLabel)
    decisionGini.append(decision[0] - decisionGiniAccuracy)
    decisionIG.append(decision[1] - decisionIGAccuracy)
    radiusNeighbor.append(classifications.radiusNeighborClassifier(trainData[:,list],trainLabel,validationData[:,list], validationLabel, testData[:,list], testLabel) - radiusNeighborAccuracy)
#    gaussianNB.append(classifications.gaussianNBClassifier(trainData[:,list],trainLabel,validationData[:,list], validationLabel, testData[:,list], testLabel) - gaussianNBAccuracy)
    gradientBoosting.append(classifications.gradientBoostingClassify(trainData[:,list],trainLabel,validationData[:,list], validationLabel, testData[:,list], testLabel) - gradientBoostingAccuracy)
    regression = classifications.regression(trainData[:,list],trainLabel,validationData[:,list], validationLabel, testData[:,list], testLabel)
    logisticRegression.append(regression[1] - logisticRegressionAccuracy)
    mnb.append(classifications.multinomialNBClassifier(trainData[:,list],trainLabel,validationData[:,list], validationLabel, testData[:,list], testLabel) - mnbAccuracy)
    RBFSVC.append(classifications.rbfNuSVC(trainData[:,list],trainLabel,validationData[:,list], validationLabel, testData[:,list], testLabel) - RBFSVCAccuracy)
    polynomialSVC.append(classifications.polyNuSVC(trainData[:,list],trainLabel,validationData[:,list], validationLabel, testData[:,list], testLabel) - polynomialSVCAccuracy)
    radiusNeighbor.append(classifications.radiusNeighborClassifier(trainData[:,list],trainLabel,validationData[:,list], validationLabel, testData[:,list], testLabel) - radiusNeighborAccuracy)

print knn
print decisionGini
print decisionIG
print linearSVC
plotFeatures = range(trainData.shape[1])
plt.plot(plotFeatures, knn, label = 'knn')
#plt.plot(plotFeatures, randomForest, label = 'randomForest', marker='o', linestyle='--', color='r')
plt.plot(plotFeatures, linearSVC, label = 'linearSVC', marker='o', linestyle='--', color='y')
plt.plot(plotFeatures, radiusNeighbor, label = 'radiusNeighbor', marker = 'o', linestyle='--', color='r')
plt.plot(plotFeatures, decisionGini, label = 'decisionGini', marker='o', linestyle='-', color='b')
plt.plot(plotFeatures, decisionIG, label = 'decisionIG', marker='o', linestyle='-', color='g')

plt.xlabel('Feature Number')
plt.ylabel('Accuracy')
plt.title('Accuracy w.r.t Features')

plt.legend()

plt.show()

'''