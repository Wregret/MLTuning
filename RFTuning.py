import matplotlib.pyplot as plt
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from Preprocessing import prepareDataset

data, target = prepareDataset()
rfresult = []

for ne in numpy.arange(10, 70, 1):
    rfavgscore = 0
    rff1 = []
    rf = RandomForestClassifier(n_estimators=ne)
    for i in range(len(data)):
        rfscore = cross_val_score(rf, data[i], target[i], cv=10, scoring='f1_weighted', n_jobs=3)
        rff1.append(rfscore.mean())
    print rff1
    for i in range(0, 9):
        rfavgscore += rff1[i]
    rfavgscore /= len(rff1)
    rfresult.append(rfavgscore)

print rfresult
plt.plot(numpy.arange(10, 70, 1), rfresult)
plt.xlabel("n estimators")
plt.ylabel("f1 score")
plt.show()
