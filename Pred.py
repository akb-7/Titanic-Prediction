# -*- coding: utf-8 -*-
"""
Created on Sun May  3 10:14:53 2020

@author: Aakash Babu
"""
# to remove the warning in my code
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

# Reading the train and testing data

train_data = pd.read_csv("D:\\Studies\\Machine Learning\\Titanic Prediction\\data\\train.csv")

test_data = pd.read_csv("D:\\Studies\\Machine Learning\\Titanic Prediction\\data\\test.csv")

check=pd.read_csv("D:\\Studies\\Machine Learning\\Titanic Prediction\\data\\gender_submission.csv")


# calculating the null values
def print_null():	
	print("\nTRAIN")
	print(train_data.isnull().sum())
	
	print("\nTEST")
	print(test_data.isnull().sum())

def print_shape():
	print("Train:",train_data.shape)
	print("\nTest:",test_data.shape)
	
def replacenull_train_embarked():
	train_data['Embarked']=np.where((train_data.Pclass==1),'C',train_data.Embarked)

def fare_test_null():
	test_data['Fare'].fillna((test_data['Fare'].mean()),inplace=True)

def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

# we now drop the cabin which is of no use
def drop_Cabin():
	test_data.drop(['Cabin'],axis=1)
	train_data.drop(['Cabin'],axis=1)
	
def replace_malefemale(): # 1 is male and 0 is female
	train_data['Sex']=np.where((train_data.Sex=='male'),1,train_data.Sex)
	test_data['Sex']=np.where((test_data.Sex=='male'),1,test_data.Sex)
	train_data['Sex']=np.where((train_data.Sex=='female'),0,train_data.Sex)
	test_data['Sex']=np.where((test_data.Sex=='female'),0,test_data.Sex)

	
	

cut_points = [-1,0,5,12,18,35,60,100]
#label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
label_names = [0,1,2,3,4,5,6]

train = process_age(train_data,cut_points,label_names)
test = process_age(test_data,cut_points,label_names)
	
def plot_agecategory():
	pivot = train.pivot_table(index="Age_categories",values='Survived')
	pivot.plot.bar()
	plt.show()

def model_run():	
	fare_test_null()
	drop_Cabin()
	replacenull_train_embarked()
	replace_malefemale()
#	print_null()
#	print_shape()


'''
Now we have our dataset free from the null values now we are going to
use various classifier by taking into an account of AGE, PClass ,Sex
'''
X=[]
model_run()

# Selecting the Age, pclass and sex from train and test as below
xtrain = train_data.iloc[:,[2,4,5,12]] # [2,4,5]
ytrain = train_data["Survived"]
xtest  = test_data.iloc[:,[1,3,4,11]] # [1,3,5]
ytest = check["Survived"]

print(xtest.shape)

# Logistic Regression model
classifier = LogisticRegression(random_state = 0) 
classifier.fit(xtrain, ytrain) 

y_pred = classifier.predict(xtest) 

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(ytest, y_pred) 

print ("Confusion Matrix : \n", cm) 
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(ytest, y_pred)) 

y_pred = pd.DataFrame(y_pred, columns=['predictions']).to_csv('D:\\Studies\\Machine Learning\\Titanic Prediction\\data\\prediction.csv')

'''
# ploting the graph


sex_pivot = train_data.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()

pclass_pivot = train_data.pivot_table(index = 'Pclass', values = 'Survived')
pclass_pivot.plot.bar()
plt.show()

emb_pivot = train_data.pivot_table(index = 'Embarked', values = 'Pclass')
emb_pivot.plot.bar()
plt.show()
'''