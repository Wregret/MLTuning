import arff
import numpy as np
from sklearn import datasets


def prepareDataset():
    iris = datasets.load_iris()
    digits = datasets.load_digits()
    wine = datasets.load_wine()
    cancer = datasets.load_breast_cancer()

    raw = arff.load(open('/home/jw/ai/arff-datasets-master/classification/ecoli.arff', 'rb'))
    ecolidata = np.array(np.delete(np.array(raw['data']), [7], axis=1), dtype=float)
    ecolitarget = np.delete(np.array(raw['data']), [0, 1, 2, 3, 4, 5, 6], axis=1).flatten()

    raw = arff.load(open('/home/jw/ai/arff-datasets-master/classification/diabetes.arff', 'rb'))
    diabetesdata = np.array(np.delete(np.array(raw['data']), [8], axis=1), dtype=float)
    diabetestarget = np.delete(np.array(raw['data']), [0, 1, 2, 3, 4, 5, 6, 7], axis=1).flatten()

    raw = arff.load(open('/home/jw/ai/arff-datasets-master/classification/letter.arff', 'rb'))
    letterdata = np.array(np.delete(np.array(raw['data']), [16], axis=1), dtype=float)
    lettertarget = np.delete(np.array(raw['data']), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                             axis=1).flatten()

    raw = arff.load(open('/home/jw/ai/arff-datasets-master/classification/haberman.arff', 'rb'))
    habermandata = np.array(np.delete(np.array(raw['data']), [3], axis=1), dtype=float)
    habermantarget = np.delete(np.array(raw['data']), [0, 1, 2], axis=1).flatten()

    raw = arff.load(open('/home/jw/ai/arff-datasets-master/classification/heart.statlog.arff', 'rb'))
    heartdata = np.array(np.delete(np.array(raw['data']), [13], axis=1), dtype=float)
    hearttarget = np.delete(np.array(raw['data']), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], axis=1).flatten()

    raw = arff.load(open('/home/jw/ai/arff-datasets-master/classification/page.blocks.arff', 'rb'))
    pagedata = np.array(np.delete(np.array(raw['data']), [10], axis=1), dtype=float)
    pagetarget = np.delete(np.array(raw['data']), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], axis=1).flatten()

    d = [iris.data, digits.data, wine.data, cancer.data, ecolidata, diabetesdata, letterdata, habermandata, heartdata,
         pagedata]
    t = [iris.target, digits.target, wine.target, cancer.target, ecolitarget, diabetestarget, lettertarget,
         habermantarget, hearttarget, pagetarget]

    return d, t
