from collections import defaultdict
from pathlib import Path
import argparse
import csv
import statistics

import cv2
import matplotlib.pyplot as plt

from src.brisque import BrisqueEvaluator
from src.noise import NoiseEvaluator
from src.segmentation import (
    create_overlay,
    dice_score,
    iou_score,
    load_binary_mask,
    load_image,
    mask_to_uint8,
    resize_mask_to_shape,
    segmentation_methods,
)


ROOT_DIR = Path("faces")
MANUAL_MASKS_DIR = Path("manual_masks")
SCORE_DIR = Path("score")

MODEL_PATH = "model/brisque_model_live.yml"
RANGE_PATH = "model/brisque_range_live.yml"

OUTPUT_CSV = "wyniki_brisque.csv"
NOISE_CSV = "wyniki_noise.csv"
SUMMARY_CSV = "podsumowanie.csv"
RANKING_CSV = "ranking_osob.csv"

CHART_MEAN_PATH = "wykres_sredni_brisque.png"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_MASK_EXTENSIONS = [".png", ".jpg", ".jpeg"]

BRISQUE_RESULTS_CSV = SCORE_DIR / "wyniki_brisque.csv"
BRISQUE_SUMMARY_CSV = SCORE_DIR / "podsumowanie_brisque.csv"
BRISQUE_RANKING_CSV = SCORE_DIR / "ranking_brisque.csv"
BRISQUE_CHART_PATH = SCORE_DIR / "wykres_sredni_brisque.png"

SEGMENTATION_RESULTS_CSV = SCORE_DIR / "wyniki_segmentacji.csv"
SEGMENTATION_SUMMARY_CSV = SCORE_DIR / "podsumowanie_segmentacji.csv"
SEGMENTATION_PERSON_SUMMARY_CSV = SCORE_DIR / "podsumowanie_segmentacji_osoby.csv"
MISSING_MASKS_CSV = SCORE_DIR / "brakujace_maski_reczne.csv"
SEGMENTATION_MASKS_DIR = SCORE_DIR / "maski"
SEGMENTATION_OVERLAYS_DIR = SCORE_DIR / "overlaye"


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def list_person_dirs(root_dir: Path, person_filter: str | None = None):
    person_dirs = []
    for path in sorted(root_dir.iterdir()):
        if not path.is_dir():
            continue
        has_images = any(
            child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
            for child in path.iterdir()
        )
        if has_images:
            person_dirs.append(path)

    if person_filter:
        person_dirs = [path for path in person_dirs if path.name == person_filter]

    return person_dirs


def list_images(person_dir: Path, limit: int | None = None):
    images = sorted(
        [
            path
            for path in person_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )
    if limit is not None:
        images = images[:limit]
    return images


def find_manual_mask_path(person_name: str, image_stem: str) -> Path | None:
    for extension in SUPPORTED_MASK_EXTENSIONS:
        candidate = MANUAL_MASKS_DIR / person_name / f"{image_stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def save_csv(rows, path: Path):
    if not rows:
        return

    ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=rows[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def summarize_brisque(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[row["osoba"]].append(row["brisque"])

    summary = []
    for osoba, scores in sorted(grouped.items()):
        summary.append(
            {
                "osoba": osoba,
                "liczba_zdjec": len(scores),
                "srednia_brisque": round(statistics.mean(scores), 4),
                "mediana_brisque": round(statistics.median(scores), 4),
                "min_brisque": round(min(scores), 4),
                "max_brisque": round(max(scores), 4),
            }
        )
    return summary


def create_brisque_ranking(summary_rows):
    ranking = sorted(summary_rows, key=lambda item: item["srednia_brisque"])
    ranking_rows = []
    for idx, row in enumerate(ranking, start=1):
        ranking_rows.append(
            {
                "miejsce": idx,
                "osoba": row["osoba"],
                "liczba_zdjec": row["liczba_zdjec"],
                "srednia_brisque": row["srednia_brisque"],
                "mediana_brisque": row["mediana_brisque"],
                "min_brisque": row["min_brisque"],
                "max_brisque": row["max_brisque"],
            }
        )
    return ranking_rows


def plot_mean_brisque(summary_rows, output_path: Path):
    ranking = sorted(summary_rows, key=lambda item: item["srednia_brisque"])
    persons = [row["osoba"] for row in ranking]
    means = [row["srednia_brisque"] for row in ranking]

    ensure_parent_dir(output_path)
    plt.figure(figsize=(12, 6))
    plt.bar(persons, means)
    plt.xlabel("Osoba")
    plt.ylabel("Sredni BRISQUE")
    plt.title("Sredni wynik BRISQUE dla kazdej osoby")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_brisque_ranking(ranking_rows):
    print("\n=== RANKING BRISQUE ===")
    for row in ranking_rows:
        print(
            f'{row["miejsce"]:>2}. {row["osoba"]:<15} '
            f'srednia BRISQUE: {row["srednia_brisque"]:.4f}'
        )


def run_brisque(person_filter: str | None = None, limit: int | None = None):
    evaluator = BrisqueEvaluator(MODEL_PATH, RANGE_PATH)
    noise_evaluator = NoiseEvaluator()
    results = []
    noise_results = []

    for person_dir in list_person_dirs(ROOT_DIR, person_filter):
        print(f"\nOsoba: {person_dir.name}")
        for image_path in list_images(person_dir, limit):
            try:
                score = evaluator.compute_score(str(image_path))
                print(f"  {image_path.name}: {score:.4f}")
                results.append(
                    {
                        "osoba": person_dir.name,
                        "zdjecie": image_path.name,
                        "sciezka": str(image_path.resolve()),
                        "brisque": round(score, 4),
                    }
                )
            except Exception as exc:
                print(f"  Blad: {image_path.name} -> {exc}")
                brisque_score = evaluator.compute_score(str(img))
                noise_score = noise_evaluator.compute_noise(str(img))
                print(
                    f"  {img.name}: BRISQUE={brisque_score:.4f}, "
                    f"NOISE={noise_score:.2f}%, "
                )

                results.append({
                    "osoba": person_dir.name,
                    "zdjecie": img.name,
                    "sciezka": str(img.resolve()),
                    "brisque": round(brisque_score, 4),
                    "noise_percent": round(noise_score, 2),
                })
                noise_results.append({
                    "osoba": person_dir.name,
                    "zdjecie": img.name,
                    "sciezka": str(img.resolve()),
                    "noise_percent": round(noise_score, 2),
                })

            except Exception as e:
                print(f"  Błąd: {img.name} -> {e}")

    if not results:
        print("Brak poprawnie przetworzonych zdjec.")
        return

    save_csv(results, BRISQUE_RESULTS_CSV)
    summary_rows = summarize_brisque(results)
    save_csv(summary_rows, BRISQUE_SUMMARY_CSV)
    ranking_rows = create_brisque_ranking(summary_rows)
    save_csv(ranking_rows, BRISQUE_RANKING_CSV)
    plot_mean_brisque(summary_rows, BRISQUE_CHART_PATH)

    print_brisque_ranking(ranking_rows)
    print("\nZakonczono.")
    print(f"Wyniki BRISQUE: {BRISQUE_RESULTS_CSV}")
    print(f"Podsumowanie BRISQUE: {BRISQUE_SUMMARY_CSV}")
    print(f"Ranking BRISQUE: {BRISQUE_RANKING_CSV}")
    print(f"Wykres BRISQUE: {BRISQUE_CHART_PATH}")


def save_mask(mask, output_path: Path):
    ensure_parent_dir(output_path)
    cv2.imwrite(str(output_path), mask_to_uint8(mask))


def save_overlay(image, mask, output_path: Path):
    ensure_parent_dir(output_path)
    cv2.imwrite(str(output_path), create_overlay(image, mask))


def summarize_segmentation(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["metoda"]].append(row)

    summary = []
    for method_name, items in sorted(grouped.items()):
        ious = [row["iou"] for row in items]
        dices = [row["dice"] for row in items]
        summary.append(
            {
                "metoda": method_name,
                "liczba_porownan": len(items),
                "srednie_iou": round(statistics.mean(ious), 4),
                "mediana_iou": round(statistics.median(ious), 4),
                "min_iou": round(min(ious), 4),
                "max_iou": round(max(ious), 4),
                "srednie_dice": round(statistics.mean(dices), 4),
                "mediana_dice": round(statistics.median(dices), 4),
                "min_dice": round(min(dices), 4),
                "max_dice": round(max(dices), 4),
            }
        )
    return summary


def summarize_segmentation_per_person(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["osoba"], row["metoda"])].append(row)

    summary = []
    for (osoba, method_name), items in sorted(grouped.items()):
        ious = [row["iou"] for row in items]
        dices = [row["dice"] for row in items]
        summary.append(
            {
                "osoba": osoba,
                "metoda": method_name,
                "liczba_porownan": len(items),
                "srednie_iou": round(statistics.mean(ious), 4),
                "srednie_dice": round(statistics.mean(dices), 4),
            }
        )
    return summary


def run_segmentation(person_filter: str | None = None, limit: int | None = None):
    compared_rows = []
    missing_mask_rows = []
    methods = segmentation_methods()

    for person_dir in list_person_dirs(ROOT_DIR, person_filter):
        print(f"\nOsoba: {person_dir.name}")

        for image_path in list_images(person_dir, limit):
            manual_mask_path = find_manual_mask_path(person_dir.name, image_path.stem)
            if manual_mask_path is None:
                missing_mask_rows.append(
                    {
                        "osoba": person_dir.name,
                        "zdjecie": image_path.name,
                        "oczekiwana_maska": str(
                            (MANUAL_MASKS_DIR / person_dir.name / f"{image_path.stem}.png").resolve()
                        ),
                    }
                )
                print(f"  {image_path.name}: brak maski recznej")
                continue
    save_csv(results, OUTPUT_CSV)
    save_csv(noise_results, NOISE_CSV)

            try:
                image = load_image(str(image_path))
                true_mask = load_binary_mask(str(manual_mask_path))
                true_mask = resize_mask_to_shape(true_mask, image.shape[:2])
            except Exception as exc:
                print(f"  Blad odczytu obrazu lub maski: {image_path.name} -> {exc}")
                continue

            for method_name, method_fn in methods.items():
                try:
                    pred_mask = method_fn(image)
                    pred_mask = resize_mask_to_shape(pred_mask, image.shape[:2])
                except Exception as exc:
                    print(f"  {image_path.name} | {method_name}: blad segmentacji -> {exc}")
                    continue

                output_mask_path = (
                    SEGMENTATION_MASKS_DIR / method_name / person_dir.name / f"{image_path.stem}.png"
                )
                output_overlay_path = (
                    SEGMENTATION_OVERLAYS_DIR
                    / method_name
                    / person_dir.name
                    / f"{image_path.stem}_overlay.png"
                )
                save_mask(pred_mask, output_mask_path)
                save_overlay(image, pred_mask, output_overlay_path)

                iou = iou_score(pred_mask, true_mask)
                dice = dice_score(pred_mask, true_mask)
                compared_rows.append(
                    {
                        "osoba": person_dir.name,
                        "zdjecie": image_path.name,
                        "metoda": method_name,
                        "sciezka_obrazu": str(image_path.resolve()),
                        "maska_reczna": str(manual_mask_path.resolve()),
                        "maska_auto": str(output_mask_path.resolve()),
                        "overlay": str(output_overlay_path.resolve()),
                        "iou": round(iou, 4),
                        "dice": round(dice, 4),
                    }
                )
                print(
                    f"  {image_path.name} | {method_name}: "
                    f"IoU={iou:.4f}, Dice={dice:.4f}"
                )

    if missing_mask_rows:
        save_csv(missing_mask_rows, MISSING_MASKS_CSV)

    print("\nZakończono.")
    print(f"Wyniki zdjęć: {OUTPUT_CSV}")
    print(f"Podsumowanie osób: {SUMMARY_CSV}")
    print(f"Ranking osób: {RANKING_CSV}")
    print(f"Wykres średnich: {CHART_MEAN_PATH}")
    print(f"Wyniki szumu: {NOISE_CSV}")
    if not compared_rows:
        print("\nNie policzono metryk, bo nie znaleziono dopasowanych masek recznych.")
        if missing_mask_rows:
            print(f"Lista brakujacych masek: {MISSING_MASKS_CSV}")
        return

    save_csv(compared_rows, SEGMENTATION_RESULTS_CSV)
    method_summary_rows = summarize_segmentation(compared_rows)
    save_csv(method_summary_rows, SEGMENTATION_SUMMARY_CSV)
    person_summary_rows = summarize_segmentation_per_person(compared_rows)
    save_csv(person_summary_rows, SEGMENTATION_PERSON_SUMMARY_CSV)

    print("\n=== PODSUMOWANIE SEGMENTACJI ===")
    for row in method_summary_rows:
        print(
            f'{row["metoda"]:<10} | '
            f'srednie IoU={row["srednie_iou"]:.4f} | '
            f'srednie Dice={row["srednie_dice"]:.4f}'
        )

    print("\nZakonczono.")
    print(f"Wyniki segmentacji: {SEGMENTATION_RESULTS_CSV}")
    print(f"Podsumowanie segmentacji: {SEGMENTATION_SUMMARY_CSV}")
    print(f"Podsumowanie segmentacji per osoba: {SEGMENTATION_PERSON_SUMMARY_CSV}")
    if missing_mask_rows:
        print(f"Brakujace maski reczne: {MISSING_MASKS_CSV}")
    print(f"Maski automatyczne: {SEGMENTATION_MASKS_DIR}")
    print(f"Overlaye: {SEGMENTATION_OVERLAYS_DIR}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["brisque", "segmentation"],
        default="segmentation",
    )
    parser.add_argument(
        "--person",
        help="Opcjonalnie ogranicz przetwarzanie do jednej osoby, np. Bartek",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Opcjonalnie ogranicz liczbe zdjec na osobe",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task == "brisque":
        run_brisque(person_filter=args.person, limit=args.limit)
    else:
        run_segmentation(person_filter=args.person, limit=args.limit)


if __name__ == "__main__":
    main()
