#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#importing dataset
dataset=pd.read_csv("iris.csv")




#seperating independent variables from dependent
X=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values





#Converting Labels into integers(Encoding)
from sklearn.preprocessing import LabelBinarizer
le=LabelBinarizer()
y=le.fit_transform(y)
#dividing the dataset into train and test set




from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)




#Creating a classifier object that will train the model based on taining data
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)



#Now we will predict using test sets
y_pred=classifier.predict(X_test)




from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
acc = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))

