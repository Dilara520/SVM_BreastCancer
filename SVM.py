import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Breast_cancer_data.csv')
df.head()
df.tail()
df.shape()
df.info()

df['diagnosis'].value_counts()

df['diagnosis'].value_counts().plot( kind="bar", color=["salmon", "lightblue"])

plt.style.use("default")
plt.style.use("seaborn-whitegrid")

X=df.iloc[:,:-1].values

y=df['diagnosis'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.20)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = svm.SVC(kernel='linear',C=10)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))

matrix = confusion_matrix(y_test, y_pred)

#SVM Parameters Optimization
#find best hyper parameters
param_grid = [
  {'C': [1, 5, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf','sigmoid']},
 ]
grid=GridSearchCV(model,param_grid=param_grid, cv=10, n_jobs=-1)
grid.fit(X_train,y_train)
Y_pred=grid.predict(X_test)

print(accuracy_score(y_test, Y_pred))

matrix = confusion_matrix(y_test, Y_pred)

grid.best_params_()
grid.best_estimator_()

#in this case the last model improvement did not yield the percentage of accuracy. However, we were succeed to decrease an error type II.