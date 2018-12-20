from sklearn import decomposition, tree
from sklearn import datasets

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
digits = datasets.load_digits()
X = digits.data
print (X[0])
Y = digits.target


pca = decomposition.PCA(n_components=37)
pca.fit(X)
X = pca.transform(X)
print (X[0])


np.random.seed(0)
(trainData, testData, trainLabels, testLabels) = train_test_split(X,
                                                                  Y, test_size=0.25, random_state=42)
dct = tree.DecisionTreeClassifier()
dct = dct.fit(X, Y)
prediction = dct.predict(testData)
print(accuracy_score(testLabels, prediction))