## Import Libraries
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
## Load and split the dataset
digits=datasets.load_digits()
features = digits.data
# print (features)
labels = digits.target
# print(labels)

# split the data to 60% training and 40% testing
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=.4)
print('Training samples is : ',len(x_train))
print('Testing samples is : ', len((x_test)))
##When K=1
knn = KNeighborsClassifier(n_neighbors=1)
clf = knn.fit(x_train,y_train)
predictions = knn.predict(x_test)
print('Training ......')
print ('Accuracy is : ',accuracy_score(y_test,predictions))
## When K=3
knn = KNeighborsClassifier(n_neighbors=3)
clf = knn.fit(x_train,y_train)
predictions = knn.predict(x_test)
print('Training ......')
print ('Accuracy is : ',accuracy_score(y_test,predictions))
## When K=5
knn = KNeighborsClassifier(n_neighbors=5)
clf = knn.fit(x_train,y_train)
predictions = knn.predict(x_test)
print('Training ......')
print ('Accuracy is : ',accuracy_score(y_test,predictions))
## When K=7
knn = KNeighborsClassifier(n_neighbors=7)
clf = knn.fit(x_train,y_train)
predictions = knn.predict(x_test)
print('Training ......')
print ('Accuracy is : ',accuracy_score(y_test,predictions))
## When K=9
knn = KNeighborsClassifier(n_neighbors=9)
clf = knn.fit(x_train,y_train)
predictions = knn.predict(x_test)
print('Training ......')
print ('Accuracy is : ',accuracy_score(y_test,predictions))