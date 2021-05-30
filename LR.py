from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

from pycaret.classification import setup
from pycaret.classification import compare_models

df=pd.read_csv("btc_bars_4h.csv")
#df1=pd.read_csv('btc_bars_1d.csv', usecols=["close"])
df = df.loc[:, df.columns != 'newvalue']
#df = df.loc[:, df.columns != 'close']
df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d').dt.strftime("%Y%m%d")
#df['date'] = df['date'].str.replace('-','')
#df['date'] = df['date'].str.replace(':','')
#df['date'] = df['date'].str.replace(' 12','')
#df['date'] = df['date'].str.replace(' 04','')
#df['date'] = df['date'].str.replace(' 8','')
#df['date'] = df['date'].str.replace('00','')
#df['date'] = df['date'].str.replace(' 06','')
#df['date'] = df['date'].str.replace(' 1','')
#df['date'] = df['date'].str.replace(' 6','')
#df['date'] = df['date'].str.replace(' 20','')
#df['date'] = df['date'].str.replace(' 08','')

df1=pd.read_csv('btc_bars_4h.csv', usecols=["open","newvalue"])
df1['testingvalue'] = df1.newvalue.shift(+1)
df1 = df1.iloc[1:-1]
df = df.iloc[1:-1]
df1['open'] = np.where(df1['open'], df1['testingvalue'], df1['open'])
#df['date'] = df['date'][:-11]
del df['open']
del df1['newvalue']
del df1['testingvalue']
df.fillna(0, inplace=True)
print(df.shape)
print(df1.shape)
print(df.head())
print(df1.head())

#print(df.head())
#df= df.iloc[1:-1]
#df1.newvalue = df1.newvalue.shift(+1)
#df1 = df1.iloc[1:-1]

x_train, x_test, y_train, y_test = train_test_split(df, df1, test_size=0.4, random_state=0)
LR1 = LogisticRegression(solver='newton-cg', max_iter=40, tol=0.0001)

LR1.fit(x_train, y_train)
predictions = LR1.predict(x_test)
score1 = LR1.score(x_train, y_train)
score = LR1.score(x_test, y_test)
cm = metrics.confusion_matrix(y_test, predictions)
TP = cm[0][0]
FN = cm[0][1]
FP = cm[1][0]
TN = cm[1][1]
print(cm)
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1 = (2 * Precision * Recall) / (Precision + Recall)

print("Accuracy Training: ", (score1*100))
print("Accuracy Prediction: ", (score*100))
print("precision: ", round(Precision*100), "%")
print("recall :", round(Recall*100), "%")
print("F1 Score: ", F1)

#df123=df=pd.read_csv("btc_bars_1d.csv")
#grid = setup(data=df, target=df1, html=False, silent=True, verbose=False)
#best = compare_models()
#print(best)




#plt.figure(figsize=(9,9))
#sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label')
#all_sample_title = 'Accuracy Score: {0}'.format(score)
#plt.title(all_sample_title, size = 15)
