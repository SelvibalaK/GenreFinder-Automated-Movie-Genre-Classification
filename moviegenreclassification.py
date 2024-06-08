# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:10:24 2024

@author: Selvibala
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random
train_data = pd.read_csv(r'C:\Users\Selvibala\Downloads\Genre Classification Dataset\train_data.txt', sep=':::', names=['Title', 'Genre', 'Description'], engine='python')
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data["Description"])
y_train = train_data["Genre"]
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
test_data = pd.read_csv(r'C:\Users\Selvibala\Downloads\Genre Classification Dataset\test_data.txt', sep=':::', names=['Title', 'Description'], engine='python')
random_index = random.randint(0, len(test_data) - 1)
selected_plot_summary = test_data.loc[random_index, "Description"]
print(f"Selected Plot Summary:\n{selected_plot_summary}\n")
X_selected_tfidf = tfidf_vectorizer.transform([selected_plot_summary])
predicted_genre = model.predict(X_selected_tfidf)[0]
print(f"Predicted Genre: {predicted_genre}")
