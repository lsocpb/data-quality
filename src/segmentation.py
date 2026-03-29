import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Nie mozna wczytac obrazu: {image_path}")
    return image


def load_binary_mask(mask_path: str) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Nie mozna wczytac maski: {mask_path}")
    return (mask > 127).astype(np.uint8)


def resize_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_shape
    if mask.shape == (target_height, target_width):
        return (mask > 0).astype(np.uint8)

    resized = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return ((mask > 0).astype(np.uint8) * 255)


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    overlay = image.copy()
    binary_mask = mask > 0
    overlay[binary_mask] = (
        overlay[binary_mask] * (1.0 - alpha) + np.array(color) * alpha
    ).astype(np.uint8)
    return overlay


def postprocess_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    binary_mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return binary_mask


def segment_grabcut(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    rect = (
        int(width * 0.05),
        int(height * 0.05),
        int(width * 0.90),
        int(height * 0.90),
    )

    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    result = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        1,
        0,
    ).astype(np.uint8)
    return postprocess_mask(result)


def segment_otsu(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(mask == 255) > 0.7:
        mask = cv2.bitwise_not(mask)

    return postprocess_mask((mask > 0).astype(np.uint8))


def segment_watershed(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(thresh == 255) > 0.7:
        thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image.copy(), markers)
    mask = np.where(markers > 1, 1, 0).astype(np.uint8)
    return postprocess_mask(mask)


def segmentation_methods():
    return {
        "grabcut": segment_grabcut,
        "otsu": segment_otsu,
        "watershed": segment_watershed,
    }


def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return float(intersection / union) if union > 0 else 1.0


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    intersection = np.logical_and(pred, true).sum()
    denom = pred.sum() + true.sum()
    return float((2 * intersection) / denom) if denom > 0 else 1.0
