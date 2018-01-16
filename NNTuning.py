import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

from Preprocessing import prepareDataset

data, target = prepareDataset()
nnresult = []

for t in numpy.arange(50, 200, 2):
    nnavgscore = 0
    nnf1 = []
    nn = MLPClassifier(solver='adam', learning_rate='adaptive', learning_rate_init=0.03, max_iter=t)
    for i in range(len(data)) :
        nnscore = cross_val_score(nn, data[i], target[i], cv=10, scoring='f1_weighted', n_jobs=3)
        nnf1.append(nnscore.mean())
    print nnf1
    for i in range(0, 9):
        nnavgscore += nnf1[i]
    nnavgscore /= len(nnf1)
    nnresult.append(nnavgscore)

print nnresult
plt.plot(numpy.arange(50, 200, 2), nnresult)
plt.xlabel("Learning rate init")
plt.ylabel("f1 score")
plt.show()
