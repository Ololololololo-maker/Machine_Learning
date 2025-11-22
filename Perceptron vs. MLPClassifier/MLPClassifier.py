#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!pip install numpy matplotlib scikit-learn


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# przygotowanie danych
wine = load_wine()

# wykorzystanie tylko 2 cech
X1 = wine.data[:, [0, 1]]
y1 = (wine.target == 0).astype(int) # Klasa 0 vs Reszta

X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.3, random_state=42, stratify=y1)

# trenowanie
pipe = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=[10], max_iter=1000, random_state=42)
)
pipe.fit(X_train, y_train)

# rysowanie granicy decyzyjnej

# tworzenie siatki punktów (Meshgrid) obejmującą cały wykres
# dodaje margines +/- 0.5, żeby punkty nie dotykały krawędzi
x_min, x_max = X1[:, 0].min() - 0.5, X1[:, 0].max() + 0.5
y_min, y_max = X1[:, 1].min() - 0.5, X1[:, 1].max() + 0.5

# rozdzielczość siatki
h = 0.02 

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# predykcja dla całej siatki
# spłaszczam siatkę (ravel) i łączę w macierz [N, 2] (c_)
grid_points = np.c_[xx.ravel(), yy.ravel()]

# używam całego 'pipe' do predykcji. 
# Pipeline SAM przeskaluje te punkty (StandardScaler) i przepuści przez MLP.
Z = pipe.predict(grid_points)

# przywracam kształt siatki, aby pasował do wykresu
Z = Z.reshape(xx.shape)

# wizaulizacja
plt.figure(figsize=(10, 8))

# rysuje tło (obszary decyzyjne)
# cmap: kolory tła to jasnoniebieski i jasnoczerwony
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))

# rysuje krawędź graniczną
plt.contour(xx, yy, Z, colors='k', linewidths=0.5)

# punkty treningowe

plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            marker='o', color='blue', edgecolor='k', label='Klasa 0 (Trening)')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
            marker='x', color='red', label='Inna klasa (Trening)')

# punkty testowe
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
            marker='v', facecolors='none', edgecolors='blue', s=80, label='Klasa 0 (Test)')
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
            marker='^', facecolors='none', edgecolors='red', s=80, label='Inna klasa (Test)')

plt.xlabel('Alcohol')
plt.ylabel('Malic Acid')
plt.title('Nieliniowa granica decyzyjna MLPClassifier')
plt.legend(loc='upper right')
plt.show()


# In[ ]:



