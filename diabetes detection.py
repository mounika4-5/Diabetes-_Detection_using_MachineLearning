#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np #2 perform mathematical operations on arrays
import pandas as pd #for data analysis
import seaborn as sns #statistical graphics

from sklearn.preprocessing import StandardScaler #resize the distribution of values
from sklearn.model_selection import train_test_split #measure the accuracy of the model 
from sklearn import svm #fit the data u provide, returning a "best fit" hyperplane that devides/categorizes ur data
from sklearn.metrics import accuracy_score #measure model performance
from sklearn.model_selection import cross_val_score 


# In[2]:


#load the dataset 2 pandas data frame for manupulating the data
raw_diabetes_data = pd.read_csv(r"C:\Users\mouni\OneDrive\Desktop\diabetesdetection.csv")

#now v hv 2 replace null values with null string otherwise it will show errors
#v will store this in variable claaed "mail_data"
diabetes_dataset = raw_diabetes_data.where((pd.notnull(raw_diabetes_data)), '')

#lets check the shape of the dataset
diabetes_dataset.shape


# In[3]:


diabetes_dataset.describe(include = "all")


# In[4]:


#counts no of observations per category
sns.countplot(diabetes_dataset['Outcome'])


# In[5]:


#checking no.of diabetes & non-diabetes
#v can c how many examples r there for class 1 & 0
print(diabetes_dataset['Outcome'].value_counts()) 


# In[6]:


#v r just finding the mean values of diabetic & non diabetic
#the mean value of non diabetic is less thn compared 2 diabetic
#this difference is very imp for us & this is how our ML Algo can find the difference b/w / it can predict b/w diabetic & non diabetic
diabetes_dataset.groupby('Outcome').mean()


# In[7]:


#splitting the variables
#assigning data(Pregnancies, Glucose, ....., Age) as X
#v r gonna drop the outcome column 
#as v r droping the column v need 2 mention axis = 1
X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)

#assigning labels(0 & 1) as Y
Y = diabetes_dataset['Outcome']


# In[8]:


print(X) #printing the data(Pregnancies, Glucose, ....., Age)
print("-----------------------------------------------------------------------------------------------")
print(Y) #printing the labels(0 & 1)


# In[9]:


#DATA STANDARDIZATION
#if there is a difference in the range of all these values
#it will b difficult for our ml model 2 make sm predictions
#in  ost cases v try 2 standardize the data in a particular range & that helps our ml 2 make better predictions
#v r loading the StandardScaler & fitting the data 2 the variable "x" 
#v r fitting all these inconsistent data wyt out StandardScaler function 
#now v need transform this data 
#based on that Standardization v r transforming all the data 2 the common range
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
     


# In[10]:


print(X) #print this data in the standardized data
print("-----------------------------------------------------------------------------------------------")
print(Y) #printing the labels(0 & 1)

#as v can c here all these values here r in the range of 0 & 1
#so this will help our model 2 make better predictions 
#coz all the values r almost in the similar range


# In[11]:


#spliting the dataset in2 Training & Testing

#test size --> 2 specify the percentage of test data needed ==> 0.2 ==> 20%

#random state --> specific split of data each value of random_state splits the data differently, v can put any state v want
#v need 2 specify the same random_state everytym if v want 2 split the data the same way everytym

#stratifying it based on the y, so that the data is split in the crt way
#stratify --> for crt distribution of data as of the original data(2 split the data correctly as of the original data)
#if i dont mention stratify = y, the distribution of 0 & 1 can b very different in the training data & testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)
     


# In[12]:


#lets c how many examples r there for each cases
#checking dimensions of data
print(X.shape, X_train.shape, X_test.shape)


# In[13]:


#lets c how many examples r there for each cases
#checking dimensions of labels
print(Y.shape, Y_train.shape, Y_test.shape)


# In[14]:


#SKLEARN APPLYING MACHINE LEARNING ALGORITHM
#training the support vector Machine Classifier
#loading the SVM 2 the variable "classifier"
#training the SVM Model with Training Data
#v r fitting the data x_train, y_train 2 the model which is the svm model, so the model is trained with the data
#linear kernel SVM is used whn the data is Linearly separable(separated using single line)
#it is used whn there r large no of features in particular dataset
#train a linear SVM classifier on the training data
classifier = svm.SVC(kernel = 'linear').fit(X_train, Y_train)
     


# In[15]:


#prediction on train_data(PRDECTING SEEN DATA)
X_train_prediction = classifier.predict(X_train)
X_train_prediction


# In[16]:


#prediction on test_data(PREDECTING UNSEEN DATA)
X_test_prediction = classifier.predict(X_test)
X_test_prediction


# In[17]:


#v r finding the accuracy_score on the training data 2 check how the model performs on traing data 
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

#v r finding the accuracy_score on the testing data 2 check how the model performs on testing data 
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[18]:


#print the accuracy_score on training data
print('Accuracy score of the training data : ', training_data_accuracy)

#print the accuracy_score on testing data
print('Accuracy score of the test data : ', test_data_accuracy)


# In[19]:


#mean accuracy (accuracy score)
#measuring the accuracy of the model against the training data 
classifier.score(X_train, Y_train)
     


# In[20]:


#mean accuracy (accuracy score)
#measuring the accuracy of the model against the test data 
classifier.score(X_test, Y_test)


# In[21]:


#cross validation
#it is used to protect against overfitting in a predictive model, 
#particularly in a case where the amount of data may be limited. In cross-validation, 
#you make a fixed number of folds (or partitions) of the data, run the analysis on each fold, and then average the overall error estimate.
#cv = 5 ==> partition the data in2 4 Training & 1 Testing Data parts
print(cross_val_score(classifier, X, Y, cv = 5))
     


# In[27]:


input_data = (5,166,72,19,175,25.8,0.0,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[23]:


import pickle #keeps track of the objects it has already serialized ==> allows saving model in very little tym
     


# In[24]:


#save the model trained in the file "trained_model.sav" to a new file called "diabetes_trained_model.pkl"
filename = 'diabetes_trained.sav'
pickle.dump(classifier, open(filename, 'wb'))


# In[25]:


#loading the saved model
loaded_model = pickle.load(open('diabetes_trained_model.sav', 'rb'))


# In[26]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:




