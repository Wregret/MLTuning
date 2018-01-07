from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy
from sklearn.neural_network import MLPClassifier
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
nnresult = []

for lri in numpy.arange(0.01, 0.04, 0.001):
    nnavgscore = 0
    nnf1 = []
    nn = MLPClassifier(solver='adam', learning_rate_init=lri, max_iter=400)
    for i in range(len(data)) :
        nnscore = cross_val_score(nn, data[i], target[i], cv=5, scoring='f1_weighted', n_jobs=3)
        nnf1.append(nnscore.mean())
    print nnf1
    for i in range(0,4) :
        nnavgscore += nnf1[i]
    nnavgscore /= len(nnf1)
    nnresult.append(nnavgscore)

print nnresult
plt.plot(numpy.arange(0.01,0.04,0.001),nnresult)
plt.xlabel("Learning rate init")
plt.ylabel("f1 score")
plt.show()
#plt.savefig('/home/jw/ai/neural network')
