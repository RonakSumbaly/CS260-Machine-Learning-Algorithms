from config import conf
from csvToArray import featureArray
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
from plotAccuracy import plotAccuracy
import numpy as np
import warnings
import operator
import bestFeature

warnings.filterwarnings("ignore")
trainData, trainLabel = featureArray(conf['train']['feature_vector'])
validationData, validationLabel = featureArray(conf['valid']['feature_vector'])
testData, testLabel = featureArray(conf['test']['feature_vector'])

guideToGraph = {}
bestParameters = {}

def knnClassifier(trainData, trainLabel, validationData, validationLabel,testData, testLabel):
    # checking for 10 neighbors
    maximumValue = 0
    returnParameters = ['0','0']
    for neighbor in xrange(1,11):
        neighAuto = KNeighborsClassifier(n_neighbors=neighbor, algorithm='auto', p=2)
        neighDistance = KNeighborsClassifier(n_neighbors=neighbor, algorithm='auto', p=2,weights='distance')
        neighAuto.fit(trainData, trainLabel)
        neighDistance.fit(trainData,trainLabel)
        scoreAuto = neighAuto.score(validationData, validationLabel)
        scoreDistance = neighDistance.score(validationData, validationLabel)
        if max(scoreAuto,scoreDistance) > maximumValue:
            maximumValue = max(scoreAuto,scoreDistance)
            returnParameters[0] = str(neighbor)
            returnParameters[1] = 'distance' if scoreDistance>scoreAuto else 'uniform'

    # Do we fit the entire (training+validation) data here ? If yes just need to do a vertical concatenate
    neighTest = KNeighborsClassifier(n_neighbors=int(returnParameters[0]), algorithm='auto', p=2,weights=returnParameters[1])
    neighTest.fit(trainData, trainLabel)
    scoreTest = neighTest.score(testData, testLabel)
    guideToGraph['KNN'] = scoreTest
    bestParameters['KNN'] = returnParameters

    return scoreTest

def radiusNeighborClassifier(trainData, trainLabel, validationData, validationLabel,testData, testLabel):
    maximumValue = 0
    returnParameters = ['0','0']
    for neighbor in xrange(100,1001,100):
        neighAutoRadius = RadiusNeighborsClassifier(radius=neighbor, weights='uniform',algorithm='auto', p=2,metric='minkowski')
        neighAutoRadius.fit(trainData, trainLabel)
        neighDistanceRadius = RadiusNeighborsClassifier(radius=neighbor, weights='distance',algorithm='auto', p=2,metric='minkowski')
        neighDistanceRadius.fit(trainData, trainLabel)
        scoreAuto = neighAutoRadius.score(validationData, validationLabel)
        scoreDistance = neighDistanceRadius.score(validationData, validationLabel)
        if max(scoreAuto,scoreDistance) > maximumValue:
            maximumValue = max(scoreAuto,scoreDistance)
            returnParameters[0] = str(neighbor)
            returnParameters[1] = 'distance' if scoreDistance>scoreAuto else 'uniform'

    neighTest = RadiusNeighborsClassifier(radius=int(returnParameters[0]), weights=returnParameters[1],algorithm='auto', p=2,metric='minkowski')
    neighTest.fit(trainData, trainLabel)
    scoreTest = neighTest.score(testData, testLabel)
    guideToGraph['Radius Neighbor'] = scoreTest
    bestParameters['Radius Neighbor'] = returnParameters

    return scoreTest

# FIX DECISION TREE
def decisionTreeClassifier(trainData, trainLabel, validationData, validationLabel,testData, testLabel):

    maxRandomPerformance = []
    for value in xrange(10):
        clf = tree.DecisionTreeClassifier(criterion='entropy')
        clf.fit(trainData, trainLabel)
        maxRandomPerformance.append(clf.score(testData, testLabel))

    guideToGraph['Decision (IG)'] = max(maxRandomPerformance)

    maxRandomPerformance = []
    for value in xrange(10):
        clf = tree.DecisionTreeClassifier(criterion='gini')
        clf.fit(trainData, trainLabel)
        maxRandomPerformance.append(clf.score(testData, testLabel))

    guideToGraph['Decision (Gini)'] = max(maxRandomPerformance)


    return (guideToGraph['Decision (IG)'] , guideToGraph['Decision (Gini)'])

def randomForestClassify(trainData, trainLabel, validationData, validationLabel,testData, testLabel):
    maximumValue = 0
    returnParameters = ['0','0']
    for value in xrange(1,100):
        clfREntropy = RandomForestClassifier(n_estimators = value,criterion='entropy')
        clfREntropy.fit(trainData, trainLabel)
        clfRGini = RandomForestClassifier(n_estimators = value,criterion='gini')
        clfRGini.fit(trainData, trainLabel)
        scoreEnt = clfREntropy.score(validationData, validationLabel)
        scoreGini = clfRGini.score(validationData, validationLabel)
        if max(scoreEnt,scoreGini) > maximumValue:
            maximumValue = max(scoreEnt,scoreGini)
            returnParameters[0] = str(value)
            returnParameters[1] = 'gini' if scoreGini > scoreEnt else 'entropy'

    neighTest = RandomForestClassifier(n_estimators = int(returnParameters[0]),criterion=returnParameters[1])
    neighTest.fit(trainData, trainLabel)
    scoreTest = neighTest.score(testData, testLabel)
    guideToGraph['Random Forests'] = scoreTest
    bestParameters['Random Forests'] = returnParameters

    return scoreTest


def gradientBoostingClassify(trainData, trainLabel, validationData, validationLabel,testData, testLabel):
    maximumValue = 0
    returnParameters = ['0']
    for value in xrange(50,350,50):
        clfDeviance = GradientBoostingClassifier(n_estimators = value,loss='deviance')
        clfDeviance.fit(trainData, trainLabel)

        scoreEnt = clfDeviance.score(validationData, validationLabel)

        if scoreEnt > maximumValue:
            maximumValue = scoreEnt
            returnParameters[0] = str(value)
    neighTest = GradientBoostingClassifier(n_estimators = int(returnParameters[0]),loss='deviance')

    neighTest.fit(trainData, trainLabel)
    scoreTest = neighTest.score(testData, testLabel)
    guideToGraph['Gradient Boosting'] = scoreTest
    bestParameters['Gradient Boosting'] = returnParameters

    return scoreTest

def regression(trainData, trainLabel, validationData, validationLabel,testData, testLabel):

    maxRandomPerformance = []
    alphaValues = np.linspace(0,0.5,41)
    for value in alphaValues:
        clf = linear_model.Ridge(alpha = value)
        clf = clf.fit(trainData,trainLabel)
        score= clf.score(validationData, validationLabel)
        maxRandomPerformance.append(score)

    indexForMax = maxRandomPerformance.index(max(maxRandomPerformance))
    alphaTest = alphaValues[indexForMax]

    clfTest = linear_model.Ridge(alpha = alphaTest)
    clfTest.fit(trainData, trainLabel)
    scoreTest = clfTest.score(testData, testLabel)
    guideToGraph['Ridge Regression'] = scoreTest

    maxRandomPerformance = []
    cValues = [10**(-7),10**(-6),10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),1]
    for value in cValues:
        clf = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=value)
        clf = clf.fit(trainData,trainLabel)
        score = clf.score(validationData, validationLabel)
        maxRandomPerformance.append(score)

    indexForMax = maxRandomPerformance.index(max(maxRandomPerformance))
    cTest = cValues[indexForMax]

    clfTest = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=cTest)
    clfTest.fit(trainData, trainLabel)
    scoreTest = clfTest.score(testData, testLabel)
    guideToGraph['Logistic Regression'] = scoreTest

    return (guideToGraph['Ridge Regression'],guideToGraph ['Logistic Regression'])


def gaussianNBClassifier(trainData, trainLabel, validationData, validationLabel,testData, testLabel):
    clf = GaussianNB()
    clf.fit(trainData, trainLabel)
    guideToGraph['Gaussian Naive Bayes'] = clf.score(testData, testLabel)

    return guideToGraph['Gaussian Naive Bayes']


def multinomialNBClassifier(trainData, trainLabel, validationData, validationLabel,testData, testLabel):
    maxRandomPerformance = []
    alphaValues = [0,10**(-7),10**(-6),10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),1]
    for value in alphaValues:
        clf = MultinomialNB(alpha=value, fit_prior=True)
        clf = clf.fit(trainData,trainLabel)
        score = clf.score(validationData, validationLabel)
        maxRandomPerformance.append(score)

    indexForMax = maxRandomPerformance.index(max(maxRandomPerformance))
    alphaTest = alphaValues[indexForMax]

    clfTest = MultinomialNB(alpha=alphaTest, fit_prior=True)
    clfTest.fit(trainData, trainLabel)
    scoreTest = clfTest.score(testData, testLabel)
    guideToGraph['Multinomial Naive Bayes'] = scoreTest
    bestParameters['Multinomial Naive Bayes'] = alphaTest

    return guideToGraph['Multinomial Naive Bayes']

'''
def bernaulliNBClassifier():
    clf = BernoulliNB()
    clf.fit(trainData, trainLabel)
    guideToGraph['Bernaulli Naive Bayes'] = clf.score(validationData, validationLabel)
'''

def polyNuSVC(trainData, trainLabel, validationData, validationLabel,testData, testLabel):
    maxRandomPerformance = []

    for deg in xrange(1,200):
        clf = NuSVC(kernel="poly",degree=deg)
        clf.fit(trainData, trainLabel)
        maxRandomPerformance.append(clf.score(validationData, validationLabel))

    gammaValue = maxRandomPerformance.index(max(maxRandomPerformance)) + 1
    clfFinal = NuSVC(kernel='poly', gamma=gammaValue)
    clfFinal.fit(trainData,trainLabel)
    score = clfFinal.score(testData,testLabel)

    guideToGraph['Polynomial Nu-SVC'] = score
    bestParameters['Polynomial Nu-SVC'] = gammaValue

    return score

def linearNuSVC(trainData, trainLabel, validationData, validationLabel,testData, testLabel):
    clf = NuSVC(kernel="linear")
    clf.fit(trainData, trainLabel)
    guideToGraph['Linear Nu-SVC'] = clf.score(testData, testLabel)

    return guideToGraph['Linear Nu-SVC']

def rbfNuSVC(trainData, trainLabel, validationData, validationLabel,testData, testLabel):
    maxRandomPerformance = []

    for gamma in xrange(1,200):
        clf = NuSVC(gamma=gamma)
        clf.fit(trainData, trainLabel)
        maxRandomPerformance.append(clf.score(validationData, validationLabel))

    gammaValue = maxRandomPerformance.index(max(maxRandomPerformance)) + 1
    clfFinal = NuSVC(gamma=gammaValue)
    clfFinal.fit(trainData,trainLabel)
    score = clfFinal.score(testData,testLabel)

    guideToGraph['RBF Nu-SVC'] = score
    bestParameters['RBF Nu-SVC'] = gammaValue

    return guideToGraph['RBF Nu-SVC']

def sigmoidNuSVC(trainData, trainLabel, validationData, validationLabel,testData, testLabel):
    maxRandomPerformance = []
    for gamma in xrange(1,200):
        clf = NuSVC(kernel="sigmoid",gamma=gamma)
        clf.fit(trainData, trainLabel)
        maxRandomPerformance.append(clf.score(validationData, validationLabel))

    gammaValue = maxRandomPerformance.index(max(maxRandomPerformance)) + 1
    clfFinal = NuSVC(kernel='sigmoid', gamma=gammaValue)
    clfFinal.fit(trainData,trainLabel)
    score = clfFinal.score(testData,testLabel)

    guideToGraph['Sigmoid Nu-SVC'] = score
    bestParameters['Sigmoid Nu-SVC'] = gammaValue

    return score
'''
def linearSVCClass():
    maxRandomPerformance = []

    for value in xrange(10):
        clf = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=0.0001, C=1.0, multi_class='crammer_singer', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
        clf = clf.fit(trainData,trainLabel)
        maxRandomPerformance.append(clf.score(validationData, validationLabel))

    guideToGraph['Linear SVC'] = max(maxRandomPerformance)
'''

def main():
    knnClassifier(trainData, trainLabel, validationData, validationLabel,testData, testLabel)
    radiusNeighborClassifier(trainData, trainLabel, validationData, validationLabel,testData, testLabel)
    decisionTreeClassifier(trainData, trainLabel, validationData, validationLabel,testData, testLabel)
    randomForestClassify(trainData, trainLabel, validationData, validationLabel,testData, testLabel)
    regression(trainData, trainLabel, validationData, validationLabel,testData, testLabel)
    gaussianNBClassifier(trainData, trainLabel, validationData, validationLabel,testData, testLabel)
    multinomialNBClassifier(trainData, trainLabel, validationData, validationLabel,testData, testLabel)
    gradientBoostingClassify(trainData, trainLabel, validationData, validationLabel,testData, testLabel)
    # bernaulliNBClassifier()
    polyNuSVC(trainData, trainLabel, validationData, validationLabel,testData, testLabel)
    linearNuSVC(trainData, trainLabel, validationData, validationLabel,testData, testLabel)
    rbfNuSVC(trainData, trainLabel, validationData, validationLabel,testData, testLabel)
    sigmoidNuSVC(trainData, trainLabel, validationData, validationLabel,testData, testLabel)

    print ">Accuracies"
    for attribute, value in guideToGraph.items():
        print('{} : {}'.format(attribute, value * 100))
    plotAccuracy(guideToGraph)

    print bestParameters
    sortedAccuracies = (sorted(guideToGraph.items(), key=operator.itemgetter(1)))
    print sortedAccuracies
    # bestFeature.selectFeature(sortedAccuracies, bestParameters)

if __name__ == "__main__":
    main()
