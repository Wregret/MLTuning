import matplotlib.pyplot as plt
import numpy
from numpy import genfromtxt
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

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
nnresult = []

for t in numpy.arange(100, 300, 2):
    nnavgscore = 0
    nnf1 = []
    nn = MLPClassifier(solver='adam', early_stopping=True, learning_rate='adaptive', learning_rate_init=0.03,
                       max_iter=t)
    for i in range(len(data)) :
        nnscore = cross_val_score(nn, data[i], target[i], cv=5, scoring='f1_weighted', n_jobs=3)
        nnf1.append(nnscore.mean())
    print nnf1
    for i in range(0,4) :
        nnavgscore += nnf1[i]
    nnavgscore /= len(nnf1)
    nnresult.append(nnavgscore)

print nnresult
plt.plot(numpy.arange(100, 300, 2), nnresult)
plt.xlabel("Learning rate init")
plt.ylabel("f1 score")
plt.show()
#plt.savefig('/home/jw/ai/neural network')
