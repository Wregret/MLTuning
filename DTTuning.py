import matplotlib.pyplot as plt
import numpy
from numpy import genfromtxt
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

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
