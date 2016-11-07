import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression


df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forcase_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forcast_out = int(math.ceil(0.01 * len(df)))

df['lable'] = df[forcase_col].shift(-forcast_out)
df.dropna(inplace=True)
# print(df.head())

x = np.array(df.drop(['lable'], 1))
y = np.array(df['lable'])

x = preprocessing.scale(x)

# x = x[:-forcast_out + 1]
# df.dropna(inplace=True)
y = np.array(df['lable'])

# print(len(x), len(y))

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = LinearRegression(n_jobs=10)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

print("Forcast Out: {}days,  Accuracy: {}%".format(forcast_out, accuracy * 100))
