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
cd projekt-jakosc-danych
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
