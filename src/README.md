# Data Pipeline – Keystroke Dynamics

Ten moduł implementuje deterministyczny pipeline przetwarzania danych keystroke dynamics.

## Kontrakt danych (Data Contract)

### 1. Surowy rekord zdarzenia

Każdy rekord w danych wejściowych reprezentuje pojedyncze zdarzenie klawisza i zawiera następujące pola:

- UserId – identyfikator użytkownika
- SampleNumber – numer próbki dla danego użytkownika
- KeyPressed – naciśnięty klawisz
- PressTime – czas wciśnięcia klawisza
- ReleaseTime – czas puszczenia klawisza

---

### 2. Definicja próbki

Jedna próbka jest zdefiniowana jako:

(UserId, SampleNumber) → wszystkie zdarzenia klawiszy dla jednego wpisania tekstu

Oznacza to, że:
- jeden użytkownik może mieć wiele próbek
- każda próbka zawiera pełną sekwencję naciśnięć klawiszy

---

### 3. Przetwarzanie danych (pipeline)

Pipeline składa się z następujących etapów:

1. Wczytanie danych z bazy danych (`load_keystrokes`)
2. Czyszczenie danych (`clean_data`)
3. Sortowanie danych (`sort_data`)
4. Budowa cech (`build_features`)

Pipeline jest:
- deterministyczny (zawsze daje ten sam wynik dla tych samych danych)
- bez efektów ubocznych (nie modyfikuje danych wejściowych)

---

### 4. Czyszczenie danych

W etapie `clean_data` stosowane są następujące reguły:

- usuwanie brakujących wartości (`dropna`)
- odrzucenie rekordów, gdzie:
  ReleaseTime < PressTime
- obliczenie czasu przytrzymania klawisza:
  hold_time = ReleaseTime - PressTime
- usunięcie anomalii:
  hold_time >= 1000 (przyjęto jako nienaturalnie długie naciśnięcie)

---

### 5. Sortowanie danych

Dane są sortowane według:

UserId → SampleNumber → PressTime

Zapewnia to:
- powtarzalność wyników
- poprawne grupowanie zdarzeń w próbki

---

### 6. Feature engineering

W etapie `build_features`:

- dane są grupowane po:
  UserId, SampleNumber, KeyPressed
- dla każdego klawisza obliczany jest średni czas przytrzymania (`hold_time`)
- dane są przekształcane tak, że:
  - jeden wiersz = jedna próbka
  - kolumny = klawisze

Przykładowa struktura:

UserId | SampleNumber | hold_a | hold_b | hold_c | ...

Brakujące wartości (klawisze niewystępujące w próbce) są uzupełniane zerami.

---

### 7. Kolumny wynikowe

#### Kolumny obowiązkowe (dla przyszłego klasyfikatora):
- UserId – etykieta klasy
- hold_* – cechy numeryczne

#### Kolumny pomocnicze:
- SampleNumber – identyfikator próbki (nie używany w modelu)

---

### 8. Właściwości pipeline

Pipeline:
- jest deterministyczny
- nie posiada efektów ubocznych
- zawsze zwraca spójną macierz cech
- przygotowuje dane do dalszej analizy (np. kNN)

---

### 9. Zakres

W obecnym etapie:
- nie implementujemy klasyfikatora (kNN)
- nie wykonujemy walidacji (leave-one-out)
- nie wykonujemy wizualizacji