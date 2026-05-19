import time
import csv
from pathlib import Path
from collections import defaultdict


DEFAULT_TEXT = "Młody sprytny bramkarz chwycił piłkę faulując wprost przed linią pola karnego."


SUPPORTED_KEYS = [
    " ", ",", ".", "/", "0", "3", "8", ":", ";", "=",
    "A", "AltGraph", "ArrowLeft", "ArrowRight", "B", "Backspace",
    "C", "CapsLock", "Control", "E", "Enter", "Insert", "K",
    "NumLock", "O", "P", "Q", "R", "S", "Shift", "T",
    "Unidentified", "W", "Y", "Z", "\\", "]",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
    "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
    "v", "w", "x", "y", "z",
    "à", "ó", "ą", "ć", "ę", "ł", "ń", "ś", "ź", "ż"
]


class TypingRecorder:
    def __init__(self):
        self.press_times = {}
        self.hold_times = defaultdict(list)

    def key_pressed(self, key: str):
        if key not in self.press_times:
            self.press_times[key] = time.time()

    def key_released(self, key: str):
        if key not in self.press_times:
            return

        press_time = self.press_times.pop(key)
        release_time = time.time()

        hold_time_ms = (release_time - press_time) * 1000

        if hold_time_ms > 0:
            self.hold_times[key].append(hold_time_ms)

    def build_feature_row(self, user_id: str, sample_number: int) -> dict:
        row = {
            "UserId": user_id,
            "SampleNumber": sample_number
        }

        for key in SUPPORTED_KEYS:
            column_name = f"hold_{key}"

            values = self.hold_times.get(key, [])

            if values:
                row[column_name] = sum(values) / len(values)
            else:
                row[column_name] = 0.0

        return row

    def save_feature_row_to_csv(
        self,
        user_id: str,
        sample_number: int,
        output_path: str = "typing_features.csv"
    ) -> str:
        output_file = Path(output_path)

        row = self.build_feature_row(user_id, sample_number)

        file_exists = output_file.exists()

        with open(output_file, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

        return str(output_file)

    def reset(self):
        self.press_times.clear()
        self.hold_times.clear()