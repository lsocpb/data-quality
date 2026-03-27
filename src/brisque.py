import cv2


class BrisqueEvaluator:
    def __init__(self, model_path: str, range_path: str):
        self.model_path = model_path
        self.range_path = range_path

        if not hasattr(cv2, "quality"):
            raise ImportError(
                "OpenCV nie ma modułu quality. Zainstaluj: pip install opencv-contrib-python"
            )

    def compute_score(self, image_path: str) -> float:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Nie można wczytać obrazu: {image_path}")

        try:
            score = cv2.quality.QualityBRISQUE_compute(
                image, self.model_path, self.range_path
            )
        except AttributeError:
            score = cv2.quality.QualityBRISQUE.compute(
                image, self.model_path, self.range_path
            )

        if isinstance(score, (tuple, list)):
            return float(score[0])

        return float(score)