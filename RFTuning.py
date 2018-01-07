import matplotlib.pyplot as plt
import numpy
from numpy import genfromtxt
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
digits = datasets.load_digits()
wine = datasets.load_wine()
cancer = datasets.load_breast_cancer()
#mnist = datasets.fetch_mldata("MNIST original", "/home/jw/ai/mldata")
#covertypes = datasets.fetch_covtype()
abalonedata = genfromtxt("/home/jw/ai/classifier/abalone/abalone.txt", delimiter='', usecols=(1,2,3,4,5,6,7,8))
abalonetarget = genfromtxt("/home/jw/ai/classifier/abalone/abalone.txt", delimiter='', usecols=(0))
mk1data, mk1target = make_classification(n_samples=1000, n_features=10, n_redundant=3, n_informative=6, random_state=1,
                                         n_clusters_per_class=1, n_classes=3)
mk2data, mk2target = make_classification(n_samples=1000, n_features=15, n_redundant=2, n_informative=10, random_state=1,
                                         n_clusters_per_class=1, n_classes=4)
mk3data, mk3target = make_classification(n_samples=1000, n_features=20, n_redundant=1, n_informative=17, random_state=1,
                                         n_clusters_per_class=1, n_classes=2)
mk4data, mk4target = make_classification(n_samples=1000, n_features=5, n_redundant=0, n_informative=5, random_state=1,
                                         n_clusters_per_class=1, n_classes=2)
mk5data, mk5target = make_classification(n_samples=1000, n_features=25, n_redundant=1, n_informative=24, random_state=1,
                                         n_clusters_per_class=1, n_classes=4)

data = [iris.data, digits.data, wine.data, cancer.data, abalonedata, mk1data, mk2data, mk3data, mk4data, mk5data]
target = [iris.target, digits.target, wine.target, cancer.target, abalonetarget, mk1target, mk2target, mk3target,
          mk4target, mk5target]
rfresult = []

for ne in numpy.arange(10, 100, 1):
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
plt.plot(numpy.arange(10, 100, 1), rfresult)
plt.xlabel("n estimators")
plt.ylabel("f1 score")
plt.show()
