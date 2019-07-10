import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

data = pd.read_csv("train.csv")

xtrain = data.drop('subscribed',axis = 1)

xtrain = pd.get_dummies(xtrain)

ytrain = data['subscribed'] 

logreg = LogisticRegression()

logreg.fit(xtrain,ytrain)

test = pd.read_csv("test.csv")

test = pd.get_dummies(test)

pred = logreg.predict(test)

test['subscribed'] = pred

test.to_csv('new.csv')
