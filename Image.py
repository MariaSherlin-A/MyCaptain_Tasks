import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
%matplotlib inline

data = pd.read_csv('Data.csv')
data.head()

a = data.iloc[3, 1:].values
a = a.reshape(28, 28).astype('uint8')
plt.imshow(a)

df_x = data.iloc[:, 1:]
df_y = data.iloc[:, 0]

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

y_train.head()

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

s = y_test.values
count = 0
for i in range(len(y_pred)):
    if y_pred[i] == s[i]:
        count += 1

print("Correct predictions:", count)
print("Total predictions:", len(y_pred))