import cv2
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

from multimodal.login_module import validate_user_identifier
from multimodal.camera_module import keep_camera_frame_temporarily
from multimodal.typing_module import DEFAULT_TEXT, TypingRecorder


class AuthDataCollectorApp:
    """
    UI do pobierania danych użytkownika.

    Aktualny zakres:
    1. Pobranie identyfikatora użytkownika.
    2. Pobranie zdjęcia twarzy z kamery i trzymanie go TYLKO w pamięci.
    3. Pobranie próbki dynamiki pisania i zapisanie jej jako wektor cech do CSV.

    Ten plik NIE wykonuje jeszcze weryfikacji użytkownika.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("System pobierania danych biometrycznych")
        self.root.geometry("840x640")
        self.root.configure(bg="#f4f6f8")

        # Dane identyfikacyjne
        self.user_id = ""

        # Zdjęcie twarzy przechowywane tymczasowo w pamięci.
        # Typ: numpy.ndarray w formacie OpenCV BGR.
        #
        # WAŻNE DLA ZESPOŁU:
        # To pole należy później przekazać do modułu rozpoznawania twarzy.
        # Nie zapisujemy tego zdjęcia do pliku.
        #
        # Przykład przyszłego użycia:
        #
        # result = face_verification_module.verify_user_face(
        #     captured_frame=self.face_frame,
        #     database_dir="faces"
        # )
        #
        # Folder referencyjny powinien mieć strukturę:
        #
        # faces/
        # ├── Majkel/
        # │   ├── 1.jpg
        # │   ├── 2.jpg
        # │   └── 3.jpg
        # ├── Anna/
        # │   ├── 1.jpg
        # │   └── 2.jpg
        #
        # Nazwa folderu powinna odpowiadać nazwie użytkownika w bazie.
        self.face_frame = None

        # Kamera i aktualna klatka
        self.camera = None
        self.current_frame = None

        # Dane dynamiki pisania
        self.sample_number = 1
        self.typing_recorder = TypingRecorder()
        self.typing_csv_path = ""

        self.show_login_screen()

    def clear(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def header(self, title, subtitle):
        Label(
            self.root,
            text=title,
            font=("Arial", 24, "bold"),
            bg="#f4f6f8",
            fg="#111827"
        ).pack(pady=(30, 8))

        Label(
            self.root,
            text=subtitle,
            font=("Arial", 12),
            bg="#f4f6f8",
            fg="#6b7280"
        ).pack(pady=(0, 20))

    def card(self):
        frame = Frame(self.root, bg="white", padx=35, pady=30)
        frame.pack(pady=10)
        return frame

    def button(self, parent, text, command, color):
        return Button(
            parent,
            text=text,
            command=command,
            font=("Arial", 13, "bold"),
            bg=color,
            fg="white",
            activebackground=color,
            activeforeground="white",
            bd=0,
            padx=25,
            pady=10,
            cursor="hand2"
        )

    # ============================================================
    # KROK 1 — LOGIN / IDENTYFIKATOR
    # ============================================================

    def show_login_screen(self):
        self.clear()

        self.header(
            "Krok 1: Identyfikacja użytkownika",
            "Podaj login użytkownika"
        )

        frame = self.card()

        Label(
            frame,
            text="Identyfikator użytkownika",
            font=("Arial", 12, "bold"),
            bg="white",
            fg="#374151"
        ).pack(anchor="w")

        self.login_entry = Entry(
            frame,
            font=("Arial", 15),
            width=38,
            relief="solid",
            bd=1
        )
        self.login_entry.pack(pady=14, ipady=6)
        self.login_entry.focus()

        self.button(
            frame,
            "Dalej",
            self.save_login,
            "#2563eb"
        ).pack(pady=12)

    def save_login(self):
        try:
            self.user_id = validate_user_identifier(self.login_entry.get())
            self.show_camera_screen()
        except ValueError as error:
            messagebox.showerror("Błąd", str(error))

    # ============================================================
    # KROK 2 — KAMERA / ZDJĘCIE TWARZY
    # ============================================================

    def show_camera_screen(self):
        self.clear()

        self.header(
            "Krok 2: Zdjęcie twarzy",
            "Ustaw twarz w kadrze i wykonaj zdjęcie"
        )

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            messagebox.showerror("Błąd", "Nie można uruchomić kamery.")
            self.show_login_screen()
            return

        self.video_label = Label(self.root, bg="#111827")
        self.video_label.pack(pady=10)

        self.button(
            self.root,
            "Zrób zdjęcie",
            self.capture_image,
            "#16a34a"
        ).pack(pady=15)

        self.update_camera_frame()

    def update_camera_frame(self):
        if self.camera is None:
            return

        ret, frame = self.camera.read()

        if ret:
            self.current_frame = frame

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (560, 360))

            image = Image.fromarray(frame_rgb)
            image_tk = ImageTk.PhotoImage(image=image)

            self.video_label.image_tk = image_tk
            self.video_label.configure(image=image_tk)

        self.root.after(20, self.update_camera_frame)

    def capture_image(self):
        """
        Pobiera aktualną klatkę z kamery i zapisuje ją tylko w pamięci.

        WAŻNE:
        - Nie zapisujemy zdjęcia do pliku.
        - self.face_frame zawiera obraz jako numpy.ndarray.
        - Obraz jest w formacie OpenCV BGR.
        - To właśnie self.face_frame należy później przekazać
          do modułu porównującego twarz z bazą faces/.

        Potencjalny przyszły moduł:
            face_verification_module.py

        Potencjalna funkcja:
            verify_user_face(captured_frame, database_dir="faces")

        Przykład:
            face_result = verify_user_face(
                captured_frame=self.face_frame,
                database_dir="faces"
            )

        Wynik tej funkcji może wyglądać np. tak:
            {
                "recognized": True,
                "matched_user": "Majkel",
                "confidence": 0.91
            }
        """

        try:
            self.face_frame = keep_camera_frame_temporarily(
                self.current_frame
            )

            if self.camera:
                self.camera.release()
                self.camera = None

            self.show_typing_screen()

        except Exception as error:
            messagebox.showerror("Błąd", str(error))

    # ============================================================
    # KROK 3 — DYNAMIKA PISANIA
    # ============================================================

    def show_typing_screen(self):
        self.clear()

        self.typing_recorder = TypingRecorder()

        self.header(
            "Krok 3: Dynamika pisania",
            "Przepisz tekst. System mierzy czasy trzymania klawiszy"
        )

        frame = self.card()

        Label(
            frame,
            text="Tekst do przepisania:",
            font=("Arial", 12, "bold"),
            bg="white",
            fg="#374151"
        ).pack(anchor="w")

        Label(
            frame,
            text=DEFAULT_TEXT,
            font=("Arial", 14),
            wraplength=660,
            bg="#eef2ff",
            fg="#1e3a8a",
            padx=18,
            pady=14,
            justify="left"
        ).pack(pady=12, fill="x")

        self.text_entry = Text(
            frame,
            font=("Arial", 15),
            width=60,
            height=4,
            relief="solid",
            bd=1,
            wrap="word"
        )
        self.text_entry.pack(pady=12)
        self.text_entry.focus()

        # Rejestrujemy moment naciśnięcia i puszczenia klawisza.
        #
        # WAŻNE DLA ZESPOŁU:
        # Te eventy nie są bezpośrednio zapisywane jako surowe dane.
        # TypingRecorder przelicza je na cechy hold_<klawisz>,
        # czyli średni czas trzymania konkretnego klawisza w milisekundach.
        self.text_entry.bind("<KeyPress>", self.on_key_press)
        self.text_entry.bind("<KeyRelease>", self.on_key_release)

        self.button(
            frame,
            "Zapisz próbkę",
            self.finish,
            "#7c3aed"
        ).pack(pady=15)

    def normalize_key(self, event):
        """
        Normalizuje nazwę klawisza do formatu zgodnego z CSV.

        Przykłady:
        - spacja       -> " "
        - litera a     -> "a"
        - litera ł     -> "ł"
        - Backspace    -> "Backspace"
        - Shift        -> "Shift"
        - Enter        -> "Enter"

        Dzięki temu kolumny w CSV mają format:
        hold_a, hold_b, hold_ł, hold_Backspace, hold_Shift itd.
        """

        if event.keysym == "space":
            return " "

        if event.char and len(event.char) == 1:
            return event.char

        return event.keysym

    def on_key_press(self, event):
        key = self.normalize_key(event)
        self.typing_recorder.key_pressed(key)

    def on_key_release(self, event):
        key = self.normalize_key(event)
        self.typing_recorder.key_released(key)

    def finish(self):
        """
        Zapisuje próbkę dynamiki pisania do CSV.

        WAŻNE DLA ZESPOŁU:
        Plik wynikowy typing_features.csv zawiera gotowy wektor cech
        dla jednej próbki użytkownika.

        Format:
            UserId,SampleNumber,hold_a,hold_b,hold_c,...

        Przykład:
            Majkel,1,105.3,109.4,107.0,...

        Te dane można potem wykorzystać bezpośrednio w module ML:

            import pandas as pd

            df = pd.read_csv("typing_features.csv")

            X = df.drop(columns=["UserId", "SampleNumber"])
            y = df["UserId"]

        Weryfikacja użytkownika może działać np. tak:
        - pobieramy nową próbkę pisania z UI,
        - tworzymy z niej wektor cech,
        - porównujemy go z próbkami zapisanymi wcześniej w CSV,
        - używamy np.:
            - odległości Euklidesa,
            - odległości Czebyszewa,
            - k-NN,
            - SVM,
            - Random Forest,
            - modelu trenowanego na cechach hold_<klawisz>.

        Ten UI zapisuje dane, ale jeszcze nie wykonuje klasyfikacji.
        """

        try:
            self.typing_csv_path = self.typing_recorder.save_feature_row_to_csv(
                user_id=self.user_id,
                sample_number=self.sample_number,
                output_path="typing_features.csv"
            )

            self.show_summary_screen()

        except Exception as error:
            messagebox.showerror("Błąd", str(error))

    # ============================================================
    # PODSUMOWANIE
    # ============================================================

    def show_summary_screen(self):
        self.clear()

        self.header(
            "Dane zostały pobrane",
            "System nie wykonuje jeszcze rozpoznawania użytkownika"
        )

        frame = self.card()

        summary = f"""
Użytkownik:
{self.user_id}

Numer próbki:
{self.sample_number}

Zdjęcie twarzy:
pobrane tymczasowo do pamięci jako self.face_frame

Format zdjęcia:
numpy.ndarray, OpenCV BGR

Cechy dynamiki pisania zapisane w:
{self.typing_csv_path}

Format danych klawiatury:
UserId, SampleNumber, hold_<klawisz>

Przyszła weryfikacja powinna korzystać z:
- self.face_frame
- typing_features.csv
- folderu faces/
"""

        Label(
            frame,
            text=summary,
            font=("Arial", 12),
            bg="white",
            fg="#111827",
            justify="left"
        ).pack(anchor="w")

        self.button(
            self.root,
            "Zamknij",
            self.close_app,
            "#111827"
        ).pack(pady=15)

    def close_app(self):
        if self.camera:
            self.camera.release()

        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = AuthDataCollectorApp(root)
    root.mainloop()