import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense

p_dir = os.getcwd()
os.chdir('C://Users//tony3//OneDrive//Documents//Machine Learning A-Z//Part 8 - Deep Learning//Section 39 - '
         'Artificial Neural Networks (ANN)')
p_dir_new = os.getcwd()

data = pd.read_csv('Churn_Modelling.csv')
y = data['Exited']
X = data.iloc[:, 3:-1]
# TODO: have a dog

# X['Age'] = X['Age'].fillna(np.mean(X['Age']))
# X['Salary'] = X['Salary'].fillna(np.mean(X['Salary']))

# X.info()
# ohe = OneHotEncoder()
# x = ohe.fit_transform(X.iloc[:, [0]]).toarray()
preprocessor = make_column_transformer((OneHotEncoder(drop='first'), ['Geography', 'Gender']), remainder='passthrough')
X_1 = preprocessor.fit_transform(X)
# TODO: add a flower
# X_2 = pd.get_dummies(X).values  # Get the same results as one-hot-encoder except here in the last 5 columns
# le = LabelEncoder()
# y_1 = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.2, random_state=0)
sc_X = StandardScaler()
X_train_1 = sc_X.fit_transform(X_train)
X_test_1 = sc_X.transform(X_test)
print()



classifier = Sequential()
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)
y_pred = classifier.predict(X_test)
y_pred_1 = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test, y_pred)
