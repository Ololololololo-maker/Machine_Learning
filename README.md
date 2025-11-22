# Perceptron vs MLPClassifier - Analiza i Porównanie

## 📋 Spis treści
- [Wprowadzenie](#wprowadzenie)
- [Analiza plików](#analiza-plików)
  - [perceptron.py](#perceptronpy)
  - [MLPClassifier.py](#mlpclassifierpy)
- [Porównanie](#porównanie)
- [Kluczowe różnice](#kluczowe-różnice)
- [Wspólne elementy](#wspólne-elementy)
- [Podsumowanie](#podsumowanie)

---

## 📖 Wprowadzenie

To repozytorium zawiera dwa skrypty demonstrujące różnicę między liniową a nieliniową klasyfikacją w uczeniu maszynowym. Oba wykorzystują zbiór danych **Wine** z biblioteki scikit-learn do klasyfikacji binarnej (klasa 0 vs pozostałe klasy).

---

## 📊 Analiza plików

### **perceptron.py**

**Cel:** Klasyfikacja binarna wina przy użyciu prostego Perceptronu

#### Kluczowe elementy:
- **Algorytm:** Perceptron - liniowy klasyfikator
- **Dane:** Dataset "wine" - 2 cechy (Alcohol, Malic Acid)
- **Preprocessing:** StandardScaler
- **Wizualizacja:** Liniowa granica decyzyjna obliczona ręcznie z wag modelu
- **Parametry:**
  - `max_iter=1000`
  - `random_state=42`

#### Szczegóły techniczne:
- Ręczne obliczanie granicy decyzyjnej z wag (`w`) i biasu (`b`)
- Transformacja do oryginalnej skali po standaryzacji
- Różne markery dla punktów treningowych (`o`, `x`) i testowych (`v`, `^`)
- Wykres z linią przerywaną reprezentującą granicę decyzyjną

#### Kod kluczowy:
```python
# Waga i bias
w = perc.coef_[0]
b = perc.intercept_[0]

# obliczam odpowiadające punkty i cofam skalowanie
x1_scaled = -(b + w[0] * x_scaled) / w[1]
y_vals = x1_scaled * scaler.scale_[1] + scaler.mean_[1]
```

---

### **MLPClassifier.py**

**Cel:** Ta sama klasyfikacja, ale z użyciem wielowarstwowej sieci neuronowej

#### Kluczowe elementy:
- **Algorytm:** MLPClassifier - nieliniowa sieć neuronowa
- **Architektura:** 1 warstwa ukryta z 10 neuronami
- **Dane:** Identyczny zbiór (wine, 2 cechy)
- **Preprocessing:** StandardScaler
- **Wizualizacja:** Nieliniowa granica decyzyjna przez meshgrid
- **Parametry:**
  - `hidden_layer_sizes=[10]`
  - `max_iter=1000`
  - `random_state=42`

#### Szczegóły techniczne:
- Generowanie gęstej siatki punktów (meshgrid) z rozdzielczością `h=0.02`
- Predykcja dla ~100,000 punktów do wizualizacji obszarów decyzyjnych
- Użycie `contourf` dla kolorowego tła i `contour` dla krawędzi granicy
- Kolorowe obszary pokazujące regiony decyzyjne

#### Kod kluczowy:
```python
# tworzenie siatki punktów
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# predykcja dla całej siatki
Z = pipe.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# wizualizacja obszarów decyzyjnych
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
```

---

## 🔄 Porównanie

| Aspekt | Perceptron | MLPClassifier |
|--------|-----------|---------------|
| **Typ modelu** | Liniowy | Nieliniowy (sieć neuronowa) |
| **Granica decyzyjna** | Prosta linia | Krzywa nieliniowa |
| **Złożoność** | Prosty (tylko wagi + bias) | Bardziej złożony (10 neuronów ukrytych) |
| **Obliczenia granicy** | Analityczne (z równania) | Numeryczne (predykcja na siatce) |
| **Wizualizacja** | Linia przerywana | Obszary kolorowe + kontur |
| **Elastyczność** | Niska - tylko separacja liniowa | Wysoka - aproksymacja nieliniowych wzorców |
| **Kod wizualizacji** | ~30 linii (ręczne obliczenia) | ~25 linii (meshgrid) |
| **Wydajność treningowa** | Bardzo szybka | Wolniejsza (optymalizacja wielu wag) |
| **Zużycie pamięci** | Minimalne | Większe (przechowuje wagi ukryte) |
| **Interpretowalność** | Wysoka | Niska (czarna skrzynka) |

---

## 💡 Kluczowe różnice

### 1. Podejście do wizualizacji
- **Perceptron:** Matematyczne wyprowadzenie linii z wag modelu
  - Wykorzystuje równanie: `w₀x₀ + w₁x₁ + b = 0`
  - Przekształca je do postaci `x₁ = f(x₀)`
  - Cofa standaryzację do oryginalnej skali

- **MLP:** Brute-force przez predykcję tysięcy punktów
  - Tworzy siatkę punktów pokrywającą cały obszar
  - Predykuje klasę dla każdego punktu
  - Rysuje kontury między regionami

### 2. Zdolność modelowania
- **Perceptron:**
  - Działa tylko dla danych **liniowo separowalnych**
  - Nie może rozwiązać problemu XOR
  - Ograniczony do prostych granic decyzyjnych

- **MLP:**
  - Może uczyć się **nieliniowych wzorców**
  - Aproksymuje dowolne funkcje (Universal Approximation Theorem)
  - Radzi sobie z XOR, koncentrycznymi okręgami, etc.

### 3. Interpretowalność
- **Perceptron:**
  - Łatwa interpretacja: `wagi = ważność cech`
  - Można bezpośrednio zobaczyć, która cecha ma większy wpływ

- **MLP:**
  - Trudna interpretacja (czarna skrzynka)
  - Wagi ukryte nie mają bezpośredniej interpretacji
  - Wymaga technik jak SHAP, LIME do zrozumienia

### 4. Przypadki użycia
- **Perceptron:**
  - Idealne dla prostych, liniowych problemów
  - Szybki prototyping
  - Baseline model

- **MLP:**
  - Lepsze dla złożonych, nieliniowych danych
  - Gdy potrzebujemy większej dokładności
  - Problemy wymagające aproksymacji nieliniowych funkcji

---

## ✅ Wspólne elementy

Oba skrypty dzielą następujące elementy:

1. **Identyczne dane wejściowe**
   - Wine dataset z scikit-learn
   - 2 cechy: Alcohol (indeks 0) i Malic Acid (indeks 1)
   - Klasyfikacja binarna: klasa 0 vs reszta

2. **Ten sam preprocessing**
   - StandardScaler do normalizacji danych
   - Pipeline do automatyzacji przepływu

3. **Podział train/test**
   - 70% trening, 30% test
   - `random_state=42` dla odtwarzalności
   - Stratyfikacja (`stratify=y1`) dla zachowania proporcji klas

4. **Podobna struktura kodu**
   - Czytelne komentarze w języku polskim
   - Logiczne sekcje: import → dane → trening → wizualizacja
   - Wykorzystanie matplotlib do wykresów

5. **Wizualizacja**
   - Różne markery dla danych treningowych i testowych
   - Legenda i etykiety osi
   - Siatka na wykresie dla lepszej czytelności

---

## 🎯 Podsumowanie

### Perceptron (1958)
Algorytm historyczny autorstwa Franka Rosenblatta. Prosty, szybki i efektywny dla liniowo separowalnych danych. Świetny jako punkt wyjścia do nauki ML.

**Zalety:**
- ✅ Bardzo szybki trening
- ✅ Łatwa interpretacja
- ✅ Niskie wymagania pamięciowe
- ✅ Idealny dla prostych problemów

**Wady:**
- ❌ Ograniczony do separacji liniowej
- ❌ Nie radzi sobie z danymi nieliniowymi
- ❌ Nie gwarantuje zbieżności dla danych nieliniowo separowalnych

### MLPClassifier (Modern)
Nowoczesna wielowarstwowa sieć neuronowa. Bardziej elastyczna i potężna, kosztem większej złożoności obliczeniowej.

**Zalety:**
- ✅ Aproksymuje nieliniowe funkcje
- ✅ Wysoka elastyczność
- ✅ Dobra dla złożonych problemów
- ✅ Silne podstawy teoretyczne

**Wady:**
- ❌ Wolniejszy trening
- ❌ Trudna interpretacja
- ❌ Więcej hiperparametrów do tuningu
- ❌ Ryzyko overfittingu

---

## 🚀 Jak uruchomić

### Wymagania
```bash
pip install numpy matplotlib scikit-learn
```

### Uruchomienie
```bash
# Perceptron
python "Perceptron vs. MLPClassifier/perceptron.py"

# MLPClassifier
python "Perceptron vs. MLPClassifier/MLPClassifier.py"
```

---

## 📚 Dodatkowe zasoby

- [Scikit-learn: Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
- [Scikit-learn: MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Wine Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset)

---

**Autor:** Machine Learning Repository
**Data utworzenia:** 2025-11-22
**Licencja:** Open Source
