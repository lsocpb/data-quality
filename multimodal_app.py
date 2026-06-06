import logging
import os
import threading
import time
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2
import pandas as pd
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from multimodal.login_module import validate_user_identifier
from multimodal.camera_module import keep_camera_frame_temporarily
from multimodal.typing_module import DEFAULT_TEXT, TypingRecorder

from src.face_eigenfaces import (
    DEFAULT_THRESHOLD as FACE_THRESHOLD,
    load_faces,
    train_eigenfaces,
    verify_face,
)
from src.biometric_fusion import BiometricFusionResult, fuse_and
from src.keystroke_identity import verify_claimed_identity
from src.keystroke_pipeline import get_engine, run_pipeline

FACES_DIR = Path(__file__).parent / "faces"
KEYSTROKE_THRESHOLD = 150.0


class AuthDataCollectorApp:
    """
    UI do pobierania danych i weryfikacji tożsamości użytkownika.

    Kroki:
    1. Pobranie identyfikatora użytkownika.
    2. Pobranie zdjęcia twarzy z kamery.
    3. Pobranie próbki dynamiki pisania.
    4. Weryfikacja biometryczna: Eigenfaces + dynamika klawiatury (fuzja AND).
    """

    def __init__(self, root):
        self.root = root
        self.root.title("System weryfikacji biometrycznej")
        self.root.geometry("840x640")
        self.root.configure(bg="#f4f6f8")

        self.user_id = ""
        self.face_frame = None
        self.camera = None
        self.current_frame = None
        self._camera_active = False
        self.sample_number = 1
        self.typing_recorder = TypingRecorder()
        self.typing_csv_path = ""

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
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
            fg="#111827",
        ).pack(pady=(30, 8))
        Label(
            self.root,
            text=subtitle,
            font=("Arial", 12),
            bg="#f4f6f8",
            fg="#6b7280",
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
            cursor="hand2",
        )

    # ============================================================
    # KROK 1 — LOGIN / IDENTYFIKATOR
    # ============================================================

    def show_login_screen(self):
        self.clear()
        self.header("Krok 1: Identyfikacja użytkownika", "Podaj login użytkownika")
        frame = self.card()

        Label(frame, text="Identyfikator użytkownika",
              font=("Arial", 12, "bold"), bg="white", fg="#374151").pack(anchor="w")

        self.login_entry = Entry(frame, font=("Arial", 15), width=38, relief="solid", bd=1)
        self.login_entry.pack(pady=14, ipady=6)
        self.login_entry.focus()

        self.button(frame, "Dalej", self.save_login, "#2563eb").pack(pady=12)

    def save_login(self):
        try:
            self.user_id = validate_user_identifier(self.login_entry.get())
            self.show_camera_screen()
        except ValueError as error:
            messagebox.showerror("Błąd", str(error))

    # ============================================================
    # KROK 2 — KAMERA / ZDJĘCIE TWARZY
    # ============================================================

    def _release_camera(self):
        self._camera_active = False
        if self.camera:
            self.camera.release()
            self.camera = None

    def _open_camera(self):
        """Próbuje otworzyć kamerę przez MSMF i DSHOW, zwraca obiekt lub None."""
        for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(0, backend)
            if not cap.isOpened():
                cap.release()
                continue
            time.sleep(0.5)  # czas na inicjalizację
            for _ in range(10):
                ret, _ = cap.read()
                if ret:
                    return cap
                time.sleep(0.05)
            cap.release()
        return None

    def show_camera_screen(self):
        self.clear()
        self._chosen_image_path = None
        self.header("Krok 2: Zdjęcie twarzy", "Zrób zdjęcie kamerą lub wybierz plik")

        # --- sekcja kamery ---
        cam_frame = Frame(self.root, bg="#f4f6f8")
        cam_frame.pack()

        self.camera = self._open_camera()
        camera_ok = self.camera is not None

        if camera_ok:
            self.video_label = Label(cam_frame, bg="#111827")
            self.video_label.pack(pady=6)
            self.button(cam_frame, "Zrób zdjęcie kamerą", self.capture_image, "#16a34a").pack(pady=6)
            self._camera_active = True
            self.update_camera_frame()
        else:
            self._release_camera()
            Label(cam_frame, text="Kamera niedostępna",
                  font=("Arial", 11), bg="#f4f6f8", fg="#9ca3af").pack(pady=6)

        # --- separator ---
        Label(self.root, text="— lub —", font=("Arial", 11),
              bg="#f4f6f8", fg="#9ca3af").pack(pady=4)

        # --- sekcja pliku (zawsze widoczna) ---
        file_frame = Frame(self.root, bg="#f4f6f8")
        file_frame.pack()

        self._file_path_label = Label(file_frame, text="Nie wybrano pliku",
                                      font=("Arial", 11), bg="#f4f6f8", fg="#9ca3af")
        self._file_path_label.pack(pady=4)

        btn_row = Frame(file_frame, bg="#f4f6f8")
        btn_row.pack()
        self.button(btn_row, "Wybierz zdjęcie z pliku…", self._pick_image_file, "#2563eb").pack(side="left", padx=6)
        self.button(btn_row, "Dalej →", self._confirm_file_image, "#16a34a").pack(side="left", padx=6)

    def _pick_image_file(self):
        path = filedialog.askopenfilename(
            title="Wybierz zdjęcie twarzy",
            initialdir=str(FACES_DIR),
            filetypes=[("Obrazy", "*.jpg *.jpeg *.png"), ("Wszystkie pliki", "*.*")],
        )
        if path:
            self._chosen_image_path = path
            self._file_path_label.config(text=Path(path).name, fg="#111827")

    def _confirm_file_image(self):
        path = getattr(self, "_chosen_image_path", None)
        if not path:
            messagebox.showwarning("Brak pliku", "Wybierz najpierw plik ze zdjęciem.")
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Błąd", f"Nie można wczytać pliku:\n{path}")
            return
        self._release_camera()
        self.face_frame = img
        self.show_typing_screen()

    def update_camera_frame(self):
        if self.camera is None or not self._camera_active:
            return
        try:
            ret, frame = self.camera.read()
            if ret:
                self.current_frame = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (560, 360))
                image = Image.fromarray(frame_rgb)
                image_tk = ImageTk.PhotoImage(image=image)
                self.video_label.image_tk = image_tk
                self.video_label.configure(image=image_tk)
        except Exception:
            pass
        if self._camera_active:
            self.root.after(20, self.update_camera_frame)

    def capture_image(self):
        try:
            self.face_frame = keep_camera_frame_temporarily(self.current_frame)
            self._release_camera()
            self.show_typing_screen()
        except Exception as error:
            messagebox.showerror("Błąd", str(error))

    # ============================================================
    # KROK 3 — DYNAMIKA PISANIA
    # ============================================================

    def show_typing_screen(self):
        self.clear()
        self.typing_recorder = TypingRecorder()
        self.header("Krok 3: Dynamika pisania",
                    "Przepisz tekst. System mierzy czasy trzymania klawiszy")

        frame = self.card()

        Label(frame, text="Tekst do przepisania:",
              font=("Arial", 12, "bold"), bg="white", fg="#374151").pack(anchor="w")

        Label(frame, text=DEFAULT_TEXT, font=("Arial", 14), wraplength=660,
              bg="#eef2ff", fg="#1e3a8a", padx=18, pady=14, justify="left").pack(pady=12, fill="x")

        self.text_entry = Text(frame, font=("Arial", 15), width=60, height=4,
                               relief="solid", bd=1, wrap="word")
        self.text_entry.pack(pady=12)
        self.text_entry.focus()

        self.text_entry.bind("<KeyPress>", self.on_key_press)
        self.text_entry.bind("<KeyRelease>", self.on_key_release)

        self.button(frame, "Weryfikuj tożsamość", self.finish, "#7c3aed").pack(pady=15)

    def normalize_key(self, event):
        if event.keysym == "space":
            return " "
        if event.char and len(event.char) == 1:
            return event.char
        return event.keysym

    def on_key_press(self, event):
        self.typing_recorder.key_pressed(self.normalize_key(event))

    def on_key_release(self, event):
        self.typing_recorder.key_released(self.normalize_key(event))

    def finish(self):
        try:
            self.typing_csv_path = self.typing_recorder.save_feature_row_to_csv(
                user_id=self.user_id,
                sample_number=self.sample_number,
                output_path="typing_features.csv",
            )
            self.show_loading_screen()
            threading.Thread(target=self._run_verification, daemon=True).start()
        except Exception as error:
            messagebox.showerror("Błąd", str(error))

    # ============================================================
    # KROK 4 — WERYFIKACJA BIOMETRYCZNA
    # ============================================================

    def show_loading_screen(self):
        self.clear()
        self.header("Krok 4: Weryfikacja biometryczna", "Trwa analiza danych…")
        frame = self.card()

        Label(frame, text="Ładowanie danych z bazy i modelu twarzy.\nProszę czekać…",
              font=("Arial", 13), bg="white", fg="#374151", justify="center").pack(pady=20)

        self._spinner_label = Label(frame, text="⏳", font=("Arial", 32), bg="white")
        self._spinner_label.pack()

    def _run_verification(self):
        try:
            result = self._verify_biometric()
            self.root.after(0, lambda: self.show_verification_result(result))
        except Exception as error:
            self.root.after(0, lambda e=error: self.show_verification_result(None, error=e))

    def _verify_biometric(self) -> BiometricFusionResult:
        # --- Weryfikacja klawiaturowa ---
        engine = get_engine()
        training_features = run_pipeline(engine)

        row_dict = self.typing_recorder.build_feature_row(self.user_id, self.sample_number)
        sample_row = pd.Series(row_dict)

        # Wyrównanie kolumn do danych treningowych
        for col in training_features.columns:
            if col not in sample_row.index:
                sample_row[col] = 0.0

        keystroke_result = verify_claimed_identity(
            training_features,
            sample_row,
            claimed_user=self.user_id,
            k=3,
            metric="euclidean",
            threshold=KEYSTROKE_THRESHOLD,
        )

        # --- Weryfikacja twarzy (Eigenfaces) ---
        images, labels, label_map = load_faces(FACES_DIR)
        recognizer = train_eigenfaces(images, labels)
        face_result = verify_face(
            recognizer,
            label_map,
            self.face_frame,
            claimed_user=self.user_id,
            threshold=FACE_THRESHOLD,
        )

        return fuse_and(face_result, keystroke_result)

    def show_verification_result(
        self,
        result: BiometricFusionResult | None,
        error: Exception | None = None,
    ):
        self.clear()

        if error is not None:
            self.header("Błąd weryfikacji", "Nie udało się przeprowadzić analizy")
            frame = self.card()
            Label(frame, text=str(error), font=("Arial", 11), bg="white",
                  fg="#dc2626", wraplength=680, justify="left").pack(pady=10)
            self.button(self.root, "Spróbuj ponownie", self.show_login_screen, "#2563eb").pack(pady=15)
            return

        if result.accepted:
            verdict = "TOŻSAMOŚĆ POTWIERDZONA"
            verdict_color = "#16a34a"
            bg_color = "#f0fdf4"
        else:
            verdict = "TOŻSAMOŚĆ ODRZUCONA"
            verdict_color = "#dc2626"
            bg_color = "#fff1f2"

        self.header("Krok 4: Wynik weryfikacji", f"Użytkownik: {result.claimed_user}")

        verdict_frame = Frame(self.root, bg=bg_color, padx=40, pady=20)
        verdict_frame.pack(pady=10, fill="x", padx=60)

        Label(verdict_frame, text=verdict, font=("Arial", 22, "bold"),
              bg=bg_color, fg=verdict_color).pack()

        details_frame = self.card()

        rows = [
            ("Twarz (Eigenfaces)", result.face_matched,
             f"pewność: {result.face_confidence:.0f}  (próg: {FACE_THRESHOLD:.0f})"),
            ("Dynamika klawiatury (KNN)", result.keystroke_matched,
             f"odległość: {result.keystroke_score:.1f} ms  (próg: {KEYSTROKE_THRESHOLD:.0f} ms)"),
            ("Strategia fuzji", None, result.fusion_strategy),
        ]

        for label_text, matched, detail in rows:
            row = Frame(details_frame, bg="white")
            row.pack(fill="x", pady=4)

            Label(row, text=label_text, font=("Arial", 12, "bold"),
                  bg="white", fg="#374151", width=30, anchor="w").pack(side="left")

            if matched is not None:
                status_text = "TAK" if matched else "NIE"
                status_color = "#16a34a" if matched else "#dc2626"
                Label(row, text=status_text, font=("Arial", 12, "bold"),
                      bg="white", fg=status_color, width=6).pack(side="left")

            Label(row, text=detail, font=("Arial", 11),
                  bg="white", fg="#6b7280").pack(side="left", padx=8)

        self.button(self.root, "Zamknij", self.close_app, "#111827").pack(pady=20)

    # ============================================================

    def close_app(self):
        self._release_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = AuthDataCollectorApp(root)
    root.mainloop()
