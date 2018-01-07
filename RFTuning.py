from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from numpy import genfromtxt

iris = datasets.load_iris()
digits = datasets.load_digits()
wine = datasets.load_wine()
cancer = datasets.load_breast_cancer()
#mnist = datasets.fetch_mldata("MNIST original", "/home/jw/ai/mldata")
#covertypes = datasets.fetch_covtype()
abalonedata = genfromtxt("/home/jw/ai/classifier/abalone/abalone.txt", delimiter='', usecols=(1,2,3,4,5,6,7,8))
abalonetarget = genfromtxt("/home/jw/ai/classifier/abalone/abalone.txt", delimiter='', usecols=(0))

data = [iris.data, digits.data, wine.data, cancer.data, abalonedata]
target = [iris.target, digits.target, wine.target, cancer.target, abalonetarget]
rfresult = []


for ne in numpy.arange(10, 100, 2):
    rfavgscore = 0
    rff1 = []
    rf = RandomForestClassifier(n_estimators=ne)
    for i in range(len(data)):
        rfscore = cross_val_score(rf, data[i], target[i], cv=5, scoring='f1_weighted', n_jobs=3)
        rff1.append(rfscore.mean())
    print rff1
    for i in range(0, 4):
        rfavgscore += rff1[i]
    rfavgscore /= len(rff1)
    rfresult.append(rfavgscore)

print rfresult
plt.plot(numpy.arange(10, 100, 2), rfresult)
plt.xlabel("n estimators")
plt.ylabel("f1 score")
plt.show()
