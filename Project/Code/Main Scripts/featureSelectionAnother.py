from csvToArray import featureArray
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import csv
import operator

def varianceRemoval():
    location = "/Users/RonakSumbaly/Documents/Development/Python/Malware Classification/Data/Feature Copy/trainFeatures.csv"

    features = []
    dict = {}
    wr = csv.reader(open(location,'r'))
    count = 0
    for line in wr:
        if count == 0:
            features = line
            count +=1
        else:
            break
    del features[0]
    data = np.genfromtxt(location,delimiter=',')
    data = np.delete(np.delete(data,0,1),0,0)
    c = 0
    for x in np.nditer(data, flags=['external_loop'], order='F'):
        dict[str(features[c])] = (np.var(x))
        c +=1

    sorted_x = sorted(dict.items(), key=operator.itemgetter(1))
    for key,value in sorted_x:
        print key + " : " + str(value)

varianceRemoval()