## Import Libraries
from sklearn import datasets
from sklearn.cross_validation import train_test_split\n",
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO
import pydotplus as pd
## Load and Split Dataset
load digits dataset from sklearn
digits =datasets.load_digits()
features = digits.data
print (features)
labels = digits.target
print(labels)
    
# split the data to 60% training and 40% testing
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=.4)\n",
print('Training samples is : ',len(x_train))
print('Testing samples is : ', len((x_test)))
## Training and Testing
DTx = tree.DecisionTreeClassifier()
clf = DTx.fit(x_train,y_train)
predictions = DTx.predict(x_test)
print('Training ......')
print ('Accuracy is : ',accuracy_score(y_test,predictions))