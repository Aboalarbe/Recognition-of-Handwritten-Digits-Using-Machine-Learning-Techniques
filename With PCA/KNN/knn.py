from sklearn import decomposition
from sklearn import datasets

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# knn
knn = KNeighborsClassifier()
knn.fit(trainData, trainLabels)
# predect using k values(1,3,5,7,9)
for i in range(1,10,2):
 KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=i, p=2,
           weights='uniform')
 prediction=knn.predict(testData)

 print(accuracy_score(testLabels, prediction))