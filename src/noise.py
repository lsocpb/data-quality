"""import cv2
import numpy as np


class NoiseEvaluator:
    def compute_noise(self, image_path: str) -> float:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")

        image = image.astype(np.float64)

        # Maska Immerkaera
        kernel = np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ])

        # Splot
        filtered = cv2.filter2D(image, cv2.CV_64F, kernel)
        filtered_center = filtered[1:-1, 1:-1]

        # Obliczenie sigma (poziom szumu)
        h, w = filtered_center.shape
        sigma = np.sqrt(np.pi / 2) * (1 / (6 * (w - 2) * (h - 2))) * np.sum(np.abs(filtered_center))

        # Zamiana na %
        noise_percent = (sigma / 255.0) * 100

        return float(noise_percent)"""

import cv2
import numpy as np


class NoiseEvaluator:
    def compute_noise(self, image_path: str) -> float:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")

        image = image.astype(np.float64)

        # ========================
        # METODA 1: IMMERKAER
        # ========================
        kernel = np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ])

        filtered = cv2.filter2D(image, -1, kernel)

        h, w = image.shape
        sigma = np.sqrt(np.pi / 2) * (1 / (6 * (w - 2) * (h - 2))) * np.sum(np.abs(filtered))

        immerkaer_noise = (sigma / 255.0) * 100

        # ========================
        # METODA 2: BLUR DIFFERENCE
        # ========================
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        noise = cv2.absdiff(image, blurred)

        blur_noise = (np.mean(noise) / 255.0) * 100

        # ========================
        # POŁĄCZENIE
        # ========================
        final_noise = (immerkaer_noise + blur_noise) / 2
        print(f"  debug: immerkaer={immerkaer_noise:.2f}, blur={blur_noise:.2f}")

        return float(final_noise)