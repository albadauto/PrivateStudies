import pandas as pd
import os as os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
import tensorflow as tf

#Subindo base de dados
directory = r"H:\Barbosa\Downloads\trip_advisor\tripadvisor_hotel_reviews.csv"
directory = directory.replace('\\', '/')
df_reviews = pd.read_csv(directory)

#Fazendo tratamento dos dados
df_reviews_isPositive = [ (df_reviews['Rating'] >= 3) ]
df_reviews.insert(2, 'isPositive', df_reviews_isPositive[0].replace({True: 1, False: 0}), True)

positive = df_reviews[df_reviews['isPositive'] == 1]
negative = df_reviews[df_reviews['isPositive'] == 0]

vectorizer = CountVectorizer()
review_vectorizer = vectorizer.fit_transform(df_reviews['Review'])

reviews = pd.DataFrame(review_vectorizer.toarray())
if not os.path.exists('csv_pronto.csv'):
    df_reviews = pd.concat([df_reviews, reviews], axis=1)
    df_reviews.to_csv('csv_pronto.csv')

df_csv_pronto = pd.read_csv('csv_pronto.csv')

#Treinando os modelos
X = reviews
y = df_reviews['isPositive']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 5)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=400, activation='relu', input_shape=(52923,)))
model.add(tf.keras.layers.Dense(units=400, activation='relu'))
model.add(tf.keras.layers.Dense(units=400, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer='Adam', loss='binary_crossentropy')
epoch_hist = model.fit(X_train, y_train, epochs=2)

#Testando a rede neural
frase = ['I Love You Renata! Forever']
euAmoRenata = vectorizer.transform(frase)
euAmoRenata = euAmoRenata.toarray()
predict_renata = model.predict(euAmoRenata)

print(r"A frase é positiva (Te amo muito muito muito Renata <3)" if predict_renata >= 0.5 else r"A frase é negativa")

#Testando a rede neural com a base de treinamento
predict_train = model.predict(X_train)
print((predict_train >= 5))









