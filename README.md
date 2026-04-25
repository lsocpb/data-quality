# ⌨️ Projekt #1: Analiza Jakości Danych

Repozytorium zawiera kod oraz środowisko eksperymentalne do analizy jakości dwóch typów danych:

1. **Szeregi czasowe / dane tabelaryczne:** Dynamika pisania na klawiaturze  
2. **Obrazy:** Zdjęcia twarzy (analiza szumu, ocena BRISQUE, segmentacja)

Ze względu na ochronę prywatności, same zbiory danych (zdjęcia i próbki klawiaturowe) **nie znajdują się** w tym repozytorium.

---

## 🚀 Jak uruchomić projekt lokalnie?

Projekt wykorzystuje wirtualne środowiska, aby zapewnić spójność wersji bibliotek u każdego członka zespołu.

---

## 1️⃣ Pobranie repozytorium

```bash
git clone https://github.com/lsocpb/data-quality.git
cd data-quality
```

---

## 2️⃣ Instalacja zależności

Projekt zarządza zależnościami za pomocą nowoczesnego narzędzia `uv`, ale wspiera również klasycznego `pip`.

Wybierz jedną z poniższych opcji:

---

### 👉 Opcja A: Używam `uv` (Zalecane)

Narzędzie `uv` automatycznie:

- utworzy środowisko w folderze `.venv`
- pobierze wszystkie pakiety na podstawie pliku `uv.lock`

```bash
uv sync
```

---

### 👉 Opcja B: Używam `pip` (Klasyczne podejście)

Jeśli nie masz zainstalowanego `uv`, utwórz środowisko ręcznie i zainstaluj paczki z pliku `requirements.txt`.

#### 🪟 Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### 🍎🐧 macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Konfiguracja edytora kodu

Aby edytor widział zainstalowane biblioteki (`pandas`, `scikit-learn`, `cv2`), musisz wskazać środowisko `.venv`.

---

## 🟦 Visual Studio Code

1. Otwórz folder projektu  
2. `Ctrl + Shift + P` (lub `Cmd + Shift + P` na Macu)  
3. Wybierz: **Python: Select Interpreter**  
4. Wskaż interpreter z folderu `.venv` (np. `./.venv/Scripts/python.exe`)  
5. W notebookach `.ipynb` upewnij się, że wybrany jest ten sam kernel  

---

## 🟩 PyCharm

1. Otwórz projekt  
2. `File -> Settings` (Mac: `PyCharm -> Settings`)  
3. `Project: projekt-jakosc-danych -> Python Interpreter`  
4. **Add Interpreter -> Add Local Interpreter**  
5. Wybierz **Existing environment**  
6. Wskaż plik `python.exe` z folderu `.venv`  
7. Kliknij **OK**

---

## ⌨️ Workflow keystroke dynamics

Repozytorium zawiera kompletny pipeline badawczy dla projektu **keystroke dynamics**:

1. pobranie surowych zdarzeń z bazy,
2. czyszczenie i budowa macierzy cech,
3. klasyfikacja kNN z metrykami `Euclidean`, `Chebyshev` i `Bray-Curtis`,
4. ewaluacja `leave-one-out`,
5. wygenerowanie artefaktów badawczych oraz wykresu `accuracy vs k`,
6. identyfikacja i weryfikacja z modyfikowalnym progiem.

### Wymagane dane wejściowe

W katalogu głównym repozytorium albo w `notebooks/.env` musi znajdować się:

```env
DATABASE_URL=postgresql://...
```

Pipeline korzysta z tabeli `"Keystrokes"` zapisanej przez aplikację frontendową i API.

### Szybki smoke test bez bazy

Ten test sprawdza:
- feature engineering,
- kNN,
- leave-one-out,
- rekomendację progu,
- identyfikację,
- weryfikację.

```bash
.venv\Scripts\python.exe main.py --task keystrokes-smoke
```

### Budowa macierzy cech i ewaluacja leave-one-out

```bash
.venv\Scripts\python.exe main.py --task keystrokes-loo --k-values 1,3,5
```

Wyniki trafiają do katalogu `score\`:

- `keystrokes_features.csv`
- `keystrokes_leave_one_out_iterations.csv`
- `keystrokes_leave_one_out_summary.csv`
- `keystrokes_leave_one_out_accuracy_curve.csv`
- `keystrokes_leave_one_out_best_by_metric.csv`
- `keystrokes_leave_one_out_best_overall.csv`
- `keystrokes_leave_one_out_accuracy_vs_k.png`

### Identyfikacja użytkownika

Identyfikacja rozpoznaje najbardziej podobnego użytkownika i może odrzucić decyzję, jeśli score przekroczy próg.

```bash
.venv\Scripts\python.exe main.py --task keystrokes-identify --metric bray_curtis --k 1 --sample-index 0
```

Jeśli nie podasz `--threshold`, skrypt sam wyznaczy rekomendowany próg z wyników `leave-one-out` dla wybranych `k` i metryki. Własny próg można wymusić np.:

```bash
.venv\Scripts\python.exe main.py --task keystrokes-identify --metric bray_curtis --k 1 --sample-index 0 --threshold 0.09
```

### Weryfikacja deklarowanej tożsamości

Weryfikacja sprawdza, czy próbka jest zgodna z zadeklarowanym użytkownikiem.

```bash
.venv\Scripts\python.exe main.py --task keystrokes-verify --metric bray_curtis --k 1 --sample-index 0 --claimed-user Jan
```

Jeśli `--claimed-user` nie zostanie podany, skrypt użyje `UserId` wybranej próbki.

### Uwagi metodologiczne

- Domyślny rekomendowany próg jest wyznaczany z `leave-one-out` i ma ograniczać błędne akceptacje.
- `keystrokes-identify` i `keystrokes-verify` są trybami demonstracyjnymi dla już zebranych próbek z bazy.
- Główną oceną jakości systemu pozostaje eksperyment `leave-one-out` oraz wykres `accuracy vs k`.
