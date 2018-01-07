from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy
from sklearn.tree import DecisionTreeClassifier
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
dtresult = []

for sp in numpy.arange(1,10,1):
    dtavgscore = 0
    dtf1 = []
    dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=sp)
    for i in range(len(data)) :
        dtscore = cross_val_score(dt, data[i], target[i], cv=5, scoring='f1_weighted', n_jobs=3)
        dtf1.append(dtscore.mean())
    print dtf1
    for i in range(0,4) :
        dtavgscore += dtf1[i]
    dtavgscore /= len(dtf1)
    dtresult.append(dtavgscore)

print dtresult
plt.plot(numpy.arange(1,10,1), dtresult)
plt.xlabel("min impurity split")
plt.ylabel("f1 score")
plt.show()
#plt.savefig('/home/jw/ai/neural network')
