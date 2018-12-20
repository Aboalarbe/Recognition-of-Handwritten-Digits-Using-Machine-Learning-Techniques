## Import Libraries 
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn_evaluation import plot
## Load and Split Dataset
digits=datasets.load_digits()
features = digits.data
# print (features)
labels = digits.target
# print(labels)

# split the data to 60% training and 40% testing
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=.4)
print('Training samples is : ',len(x_train))
print('Testing samples is : ', len((x_test)))
## Training and Testing
ANN=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,), random_state=3)
clf=ANN.fit(features, labels)
predictions=clf.predict(x_test)

print('Training ......')
print ('Accuracy is : ',accuracy_score(y_test,predictions))
## Plot the Confusing Matrix
plot.confusion_matrix(y_test, predictions)
plt.show()
