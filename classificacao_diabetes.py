import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

directory = "H:\Barbosa\Downloads\diabetes.csv"
directory = directory.replace('\\', r'/')
df_diabetes = pd.read_csv(directory)
df_diabetes.describe()

standardscaler = StandardScaler()


X = df_diabetes.iloc[:, 0:8].values
y = df_diabetes.iloc[:, 8].values
X = standardscaler.fit_transform(X)

print(X)
print(y)

X_train, Y_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=400, activation='relu', input_shape=(8,)))
model.add(tf.keras.layers.Dense(units=400, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
epoch_hist = model.fit(X_train, y_train, epochs = 500)

#predict = model.predict(X_train)
#predict = (predict >= 0.5)
#Gravida - Glucose - Pressao sanguinea - Gordura na pele insulina - IMC - DiabetesPedigreeFunction - Idade
predict = [[ 6, 200, 60, 43, 0, 25.6, 0.300, 50 ]]
predict_standard = standardscaler.fit_transform(predict)
final = model.predict(predict_standard)

print(( final >= 0.5 ))