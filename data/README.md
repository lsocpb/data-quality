# 📊 Dane Projektowe (Data)

Ze względów bezpieczeństwa oraz ochrony prywatności (RODO), w tym repozytorium **nie są przechowywane** żadne surowe pliki z danymi osobowymi, zdjęciami twarzy, ani wyeksportowanymi próbkami biometrycznymi. Folder ten jest ignorowany przez system kontroli wersji (`.gitignore`).

Dane dotyczące dynamiki pisania na klawiaturze (Keystroke Dynamics) są zbierane przez dedykowaną aplikację webową (React) i zapisywane na bieżąco w chmurowej bazie danych **PostgreSQL (Neon.tech)**.

Poniżej znajduje się instrukcja, jak w bezpieczny sposób pobrać te dane bezpośrednio do środowiska analitycznego (Jupyter Notebook).

---

## 🛠️ Jak połączyć się z bazą danych w Jupyter Notebook?

Zamiast ręcznie eksportować pliki `.csv`, nasz projekt łączy się z bazą danych "w locie", co gwarantuje, że analizujemy zawsze najświeższe dane.

### Krok 1: Instalacja wymaganych bibliotek

Upewnij się, że w Twoim wirtualnym środowisku (np. przez `uv` lub `pip`) zainstalowane są pakiety do obsługi bazy i zmiennych środowiskowych:

```bash
uv add pandas sqlalchemy psycopg2-binary python-dotenv
```

### Krok 2: Konfiguracja zmiennych środowiskowych (Bezpieczeństwo!)

**Nigdy nie wpisuj hasła do bazy bezpośrednio w notatniku.**

1. Utwórz w głównym katalogu projektu plik o nazwie `.env` (plik ten jest ignorowany przez Gita).
2. Wklej do niego swój ciąg połączeniowy z bazy Neon.tech (pamiętaj, aby prefiks `postgres://` zamienić na `postgresql://`):

```env
DATABASE_URL=postgresql://<USER>:<PASSWORD>@ep-twoja-baza.eu-central-1.aws.neon.tech/neondb?sslmode=require
```

### Krok 3: Pobranie danych w Pythonie

W swoim pliku `.ipynb` (np. w folderze `notebooks/`) użyj poniższego kodu, aby pobrać całą tabelę z próbkami do obiektu `pandas.DataFrame`.

```python
import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# 1. Wczytanie bezpiecznych zmiennych z pliku .env
load_dotenv()
db_url = os.getenv("DATABASE_URL")

if not db_url:
    raise ValueError("Nie znaleziono DATABASE_URL. Sprawdź plik .env!")

# 2. Utworzenie połączenia z bazą danych
print("Łączenie z bazą PostgreSQL...")
engine = create_engine(db_url)

# 3. Wykonanie zapytania SQL i zapis do DataFrame
query = 'SELECT * FROM "Keystrokes"'
df = pd.read_sql(query, engine)

print(f"✅ Sukces! Pobrano {len(df)} rekordów.")

# 4. Podgląd pierwszych próbek
display(df.head())
```

Od tego momentu masz w zmiennej `df` surowe dane (ID użytkownika, wciśnięty klawisz, czas wciśnięcia i puszczenia), gotowe do agregacji cech (Dwell Time, Flight Time) oraz zaawansowanej analizy z użyciem algorytmów PCA i k-Means.
