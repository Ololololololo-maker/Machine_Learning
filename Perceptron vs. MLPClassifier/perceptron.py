#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install scikit-learn numpy matplotlib


# In[4]:


# Wytrenowanie modelu na podstawie zbioru danych "wine" pobranych z scikit-learn z zastosowaniem Perceptronu. 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# wczytanie danych

wine = load_wine()

#wykorzystanie tylko 2 cech

X1 = wine.data[:, [0, 1]]
y1 = (wine.target == 0).astype(int)

# podział na dane testowe i trenowane

X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.3, random_state=42, stratify=y1)

# Tworzę pipeline 
pipe = make_pipeline(
    StandardScaler(),
    Perceptron(max_iter=1000, random_state=42)
)
pipe.fit(X_train, y_train)

# Pobieranie skalera i perceptronu z potoku

scaler = pipe.named_steps["standardscaler"]
perc = pipe.named_steps["perceptron"]

# Waga i bias 
w = perc.coef_[0]
b = perc.intercept_[0]

# zakres osi 
x0_min, x0_max = X1[:, 0].min() - 0.5, X1[:, 0].max() + 0.5
x_vals = np.linspace(x0_min, x0_max, 400)

# zmieniam na skalę standaryzowaną 
x_scaled = (x_vals - scaler.mean_[0]) / scaler.scale_[0]

# obliczam odpowiadające punkty i cofam skalowanie

x1_scaled = -(b + w[0] * x_scaled) / w[1]
y_vals = x1_scaled * scaler.scale_[1] + scaler.mean_[1] # -> powrót do oryginalnej skali

# rysunek
plt.figure(figsize=(10,7))

# punkty treningowe

plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            marker='o', edgecolor='k', label='klasa 0 treningowa')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
            marker = 'x', color = 'b', label='klasa ≠ 0 treningowa')

# punkty testowe 

plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
            marker='v', facecolors='none', edgecolor = 'r', alpha=0.7, label='klasa 0 testowa')
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
            marker='^', color = 'g', alpha=0.7, label='klasa ≠ 0 testowa')

plt.plot(x_vals, y_vals, "k--", linewidth=2, label="Granica decyzyjna")

# ustawienie limitów osi Y na podstawie danych.
plt.ylim(X1[:, 1].min() - 1, X1[:, 1].max() + 1)

plt.xlabel('Alcohol (stężenie)')
plt.ylabel('Malic Acid (kwas jabłkowy)')
plt.title('Klasyfikacja wina: Perceptron (Skalowany)')
plt.legend(loc='upper right', framealpha=0.9) # przeniesienie legendy i dodanie tła
plt.grid(True, alpha=0.3) # dodanie siatki na wykresie


# In[ ]:



