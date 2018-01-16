import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from Preprocessing import prepareDataset

data, target = prepareDataset()
dtresult = []

for sp in numpy.arange(1,10,1):
    dtavgscore = 0
    dtf1 = []
    dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=sp)
    for i in range(len(data)) :
        dtscore = cross_val_score(dt, data[i], target[i], cv=10, scoring='f1_weighted', n_jobs=3)
        dtf1.append(dtscore.mean())
    print dtf1
    for i in range(0, 9):
        dtavgscore += dtf1[i]
    dtavgscore /= len(dtf1)
    dtresult.append(dtavgscore)

print dtresult
plt.plot(numpy.arange(1,10,1), dtresult)
plt.xlabel("min impurity split")
plt.ylabel("f1 score")
plt.show()
