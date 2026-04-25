from collections import defaultdict
from pathlib import Path
import argparse
import csv
import statistics
import os

import cv2
import matplotlib.pyplot as plt

from src.noise import NoiseEvaluator

from src.brisque import BrisqueEvaluator
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
from src.keystroke_pipeline import get_engine, run_pipeline, test_pipeline
from src.keystroke_leave_one_out import evaluate_leave_one_out
from src.keystroke_research_artifacts import (
    build_keystroke_research_artifacts,
    plot_accuracy_vs_k,
)
from src.keystroke_smoke_tests import smoke_test_keystroke_pipeline_and_knn
from src.keystroke_knn import available_metrics, classify_sample
from src.keystroke_identity import (
    identify_sample,
    suggest_identification_threshold,
    verify_claimed_identity,
)

ROOT_DIR = Path("faces")
MANUAL_MASKS_DIR = Path("manual_masks")
SCORE_DIR = Path("score")

MODEL_PATH = "model/brisque_model_live.yml"
RANGE_PATH = "model/brisque_range_live.yml"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_MASK_EXTENSIONS = [".png", ".jpg", ".jpeg"]

BRISQUE_RESULTS_CSV = SCORE_DIR / "wyniki_brisque.csv"
BRISQUE_SUMMARY_CSV = SCORE_DIR / "podsumowanie_brisque.csv"
BRISQUE_RANKING_CSV = SCORE_DIR / "ranking_brisque.csv"
BRISQUE_CHART_PATH = SCORE_DIR / "wykres_sredni_brisque.png"

NOISE_RESULTS_CSV = SCORE_DIR / "wyniki_noise.csv"

SEGMENTATION_RESULTS_CSV = SCORE_DIR / "wyniki_segmentacji.csv"
SEGMENTATION_SUMMARY_CSV = SCORE_DIR / "podsumowanie_segmentacji.csv"
SEGMENTATION_PERSON_SUMMARY_CSV = SCORE_DIR / "podsumowanie_segmentacji_osoby.csv"
MISSING_MASKS_CSV = SCORE_DIR / "brakujace_maski_reczne.csv"
SEGMENTATION_MASKS_DIR = SCORE_DIR / "maski"
SEGMENTATION_OVERLAYS_DIR = SCORE_DIR / "overlaye"

KEYSTROKES_FEATURES_CSV = SCORE_DIR / "keystrokes_features.csv"
KEYSTROKES_LOO_ITERATIONS_CSV = SCORE_DIR / "keystrokes_leave_one_out_iterations.csv"
KEYSTROKES_LOO_SUMMARY_CSV = SCORE_DIR / "keystrokes_leave_one_out_summary.csv"
KEYSTROKES_LOO_ACCURACY_CURVE_CSV = SCORE_DIR / "keystrokes_leave_one_out_accuracy_curve.csv"
KEYSTROKES_LOO_BEST_BY_METRIC_CSV = SCORE_DIR / "keystrokes_leave_one_out_best_by_metric.csv"
KEYSTROKES_LOO_BEST_OVERALL_CSV = SCORE_DIR / "keystrokes_leave_one_out_best_overall.csv"
KEYSTROKES_LOO_ACCURACY_PLOT = SCORE_DIR / "keystrokes_leave_one_out_accuracy_vs_k.png"
DEFAULT_LOO_K_VALUES = [1, 3, 5]
DEFAULT_SAMPLE_INDEX = 48


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


def parse_k_values(raw_value: str | None) -> list[int]:
    if not raw_value:
        return DEFAULT_LOO_K_VALUES.copy()

    values = []
    for chunk in raw_value.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        k = int(stripped)
        if k <= 0:
            raise ValueError("All leave-one-out k values must be positive")
        if k not in values:
            values.append(k)

    if not values:
        raise ValueError("At least one leave-one-out k value is required")

    return values


def pick_sample(df, sample_index: int | None):
    if df.empty:
        raise ValueError("Cannot classify an empty feature set")

    if sample_index is None:
        sample_index = DEFAULT_SAMPLE_INDEX

    resolved_index = min(max(int(sample_index), 0), len(df) - 1)
    return resolved_index, df.iloc[resolved_index]


def resolve_threshold(df, *, threshold: float | None, metric: str, k: int):
    if threshold is not None:
        return float(threshold), None

    evaluation = evaluate_leave_one_out(df, ks=[k], metrics=[metric])
    recommendation = suggest_identification_threshold(
        evaluation.iterations,
        metric=metric,
        k=k,
    )
    return recommendation.threshold, recommendation


def print_threshold_recommendation(recommendation):
    if recommendation is None:
        return

    print("\n=== RECOMMENDED IDENTIFICATION THRESHOLD ===")
    print(
        f"metric={recommendation.metric} "
        f"k={recommendation.k} "
        f"threshold={recommendation.threshold:.6f} "
        f"decision_accuracy={recommendation.decision_accuracy:.4f} "
        f"FAR={recommendation.false_accept_rate:.4f} "
        f"FRR={recommendation.false_reject_rate:.4f} "
        f"strategy={recommendation.strategy}"
    )


def print_keystroke_loo_summary(summary_df, *, title: str = "KEYSTROKES LEAVE-ONE-OUT SUMMARY"):
    print(f"\n=== {title} ===")
    for _, row in summary_df.iterrows():
        print(
            f'metric={row["metric"]:<12} '
            f'k={int(row["k"]):<2} '
            f'accuracy={row["accuracy"]:.4f} '
            f'precision_macro={row["precision_macro"]:.4f} '
            f'recall_macro={row["recall_macro"]:.4f} '
            f'f1_macro={row["f1_macro"]:.4f}'
        )


def print_keystroke_research_overview(best_overall_df, best_by_metric_df):
    if not best_overall_df.empty:
        best_row = best_overall_df.iloc[0]
        print("\n=== KEYSTROKES BEST OVERALL ===")
        print(
            f'metric={best_row["metric"]} '
            f'k={int(best_row["k"])} '
            f'accuracy={best_row["accuracy"]:.4f} '
            f'f1_macro={best_row["f1_macro"]:.4f}'
        )

    print("\n=== KEYSTROKES BEST BY METRIC ===")
    for _, row in best_by_metric_df.iterrows():
        print(
            f'metric={row["metric"]:<12} '
            f'best_k={int(row["k"]):<2} '
            f'accuracy={row["accuracy"]:.4f} '
            f'f1_macro={row["f1_macro"]:.4f}'
        )


def run_keystrokes_leave_one_out(df, *, k_values: list[int]):
    evaluation = evaluate_leave_one_out(
        df,
        ks=k_values,
        metrics=available_metrics(),
    )
    artifacts = build_keystroke_research_artifacts(evaluation)

    os.makedirs("score", exist_ok=True)
    artifacts.raw_results.to_csv(KEYSTROKES_LOO_ITERATIONS_CSV, index=False)
    artifacts.aggregated_results.to_csv(KEYSTROKES_LOO_SUMMARY_CSV, index=False)
    artifacts.accuracy_curve.to_csv(KEYSTROKES_LOO_ACCURACY_CURVE_CSV, index=False)
    artifacts.best_by_metric.to_csv(KEYSTROKES_LOO_BEST_BY_METRIC_CSV, index=False)
    artifacts.best_overall.to_csv(KEYSTROKES_LOO_BEST_OVERALL_CSV, index=False)
    plot_accuracy_vs_k(artifacts.accuracy_curve, KEYSTROKES_LOO_ACCURACY_PLOT)

    print_keystroke_loo_summary(artifacts.aggregated_results)
    print_keystroke_research_overview(artifacts.best_overall, artifacts.best_by_metric)
    print(f"\nZapisano wyniki surowe LOO do: {KEYSTROKES_LOO_ITERATIONS_CSV}")
    print(f"Zapisano wyniki zagregowane LOO do: {KEYSTROKES_LOO_SUMMARY_CSV}")
    print(f"Zapisano accuracy curve do: {KEYSTROKES_LOO_ACCURACY_CURVE_CSV}")
    print(f"Zapisano najlepsze konfiguracje per metryka do: {KEYSTROKES_LOO_BEST_BY_METRIC_CSV}")
    print(f"Zapisano najlepsza konfiguracje globalna do: {KEYSTROKES_LOO_BEST_OVERALL_CSV}")
    print(f"Zapisano wykres accuracy vs k do: {KEYSTROKES_LOO_ACCURACY_PLOT}")

    return evaluation


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
    results = []

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

def run_noise(person_filter: str | None = None, limit: int | None = None):
    noise_evaluator = NoiseEvaluator()
    noise_results = []

    for person_dir in list_person_dirs(ROOT_DIR, person_filter):
        print(f"\nOsoba: {person_dir.name}")

        for image_path in list_images(person_dir, limit):
            try:
                noise_score = noise_evaluator.compute_noise(str(image_path))

                print(f"  {image_path.name}: NOISE={noise_score:.2f}%")

                noise_results.append({
                    "osoba": person_dir.name,
                    "zdjecie": image_path.name,
                    "sciezka": str(image_path.resolve()),
                    "noise_percent": round(noise_score, 2),
                })

            except Exception as e:
                print(f"  Błąd: {image_path.name} -> {e}")

    if not noise_results:
        print("Brak danych.")
        return

    save_csv(noise_results, NOISE_RESULTS_CSV)

    print("\nZakończono.")
    print(f"Wyniki szumu: {NOISE_RESULTS_CSV}")


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
        choices=[
            "brisque",
            "segmentation",
            "noise",
            "keystrokes",
            "keystrokes-loo",
            "keystrokes-identify",
            "keystrokes-verify",
            "keystrokes-smoke",
        ],
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
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Liczba sąsiadów dla KNN (tylko dla keystrokes)",
    )

    parser.add_argument(
        "--metric",
        choices=["euclidean", "chebyshev", "bray_curtis"],
        default="euclidean",
        help="Metryka odległości dla KNN",
    )

    parser.add_argument(
        "--k-values",
        help="Lista wartosci k dla leave-one-out, np. 1,3,5",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Opcjonalny prog identyfikacji/weryfikacji. Gdy brak, zostanie zaproponowany z leave-one-out.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        help="Indeks probki do identyfikacji lub weryfikacji.",
    )
    parser.add_argument(
        "--claimed-user",
        help="Deklarowany uzytkownik do weryfikacji. Gdy brak, zostanie uzyty UserId probki.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.task == "brisque":
        run_brisque(person_filter=args.person, limit=args.limit)
    elif args.task == "noise":
        run_noise(person_filter=args.person, limit=args.limit)
    elif args.task == "keystrokes-smoke":
        smoke_test_keystroke_pipeline_and_knn()
    elif args.task in {"keystrokes", "keystrokes-loo", "keystrokes-identify", "keystrokes-verify"}:
        engine = get_engine()
        df = run_pipeline(engine)
        test_pipeline(engine)
        print("\n=== KEYSTROKES FEATURES ===")
        print(df.head())
        print(f"\nLiczba próbek: {len(df)}")

        os.makedirs("score", exist_ok=True)
        df.to_csv(KEYSTROKES_FEATURES_CSV, index=False)
        print(f"\nZapisano cechy do: {KEYSTROKES_FEATURES_CSV}")

        if args.task == "keystrokes" and len(df) > 1:
            sample_index, sample = pick_sample(df, args.sample_index)
            prediction = classify_sample(
                df,
                sample,
                k=args.k,
                metric=args.metric,
                exclude_same_sample=True,
            )
            print("\n=== KNN SAMPLE CLASSIFICATION ===")
            print(f"Metryka: {args.metric}, k: {args.k}")
            print(f"Indeks próbki: {sample_index}")
            print(f'Próbka: UserId={sample["UserId"]}, SampleNumber={sample["SampleNumber"]}')
            print(f"Przewidziany użytkownik: {prediction.predicted_user}")
            print(f"Score: {prediction.score:.6f}")
            loo_k_values = parse_k_values(args.k_values)
            run_keystrokes_leave_one_out(df, k_values=loo_k_values)

        elif args.task == "keystrokes-identify" and len(df) > 1:
            sample_index, sample = pick_sample(df, args.sample_index)
            threshold, recommendation = resolve_threshold(
                df,
                threshold=args.threshold,
                metric=args.metric,
                k=args.k,
            )
            decision = identify_sample(
                df,
                sample,
                k=args.k,
                metric=args.metric,
                threshold=threshold,
                exclude_same_sample=True,
            )
            print_threshold_recommendation(recommendation)
            print("\n=== KEYSTROKES IDENTIFICATION ===")
            print(f"Metryka: {args.metric}, k: {args.k}, threshold: {threshold:.6f}")
            print(f"Indeks próbki: {sample_index}")
            print(f'Próbka: UserId={sample["UserId"]}, SampleNumber={sample["SampleNumber"]}')
            print(f"Najbliższy użytkownik: {decision.predicted_user}")
            print(f"Score: {decision.score:.6f}")
            print(
                f"Decyzja: {'odrzucono próbkę' if decision.rejected else f'zaakceptowano użytkownika {decision.accepted_user}'}"
            )

        elif args.task == "keystrokes-verify" and len(df) > 1:
            sample_index, sample = pick_sample(df, args.sample_index)
            threshold, recommendation = resolve_threshold(
                df,
                threshold=args.threshold,
                metric=args.metric,
                k=args.k,
            )
            claimed_user = args.claimed_user or sample["UserId"]
            decision = verify_claimed_identity(
                df,
                sample,
                claimed_user=claimed_user,
                k=args.k,
                metric=args.metric,
                threshold=threshold,
                exclude_same_sample=True,
            )
            print_threshold_recommendation(recommendation)
            print("\n=== KEYSTROKES VERIFICATION ===")
            print(f"Metryka: {args.metric}, k: {args.k}, threshold: {threshold:.6f}")
            print(f"Indeks próbki: {sample_index}")
            print(f'Próbka: UserId={sample["UserId"]}, SampleNumber={sample["SampleNumber"]}')
            print(f"Deklarowany użytkownik: {decision.claimed_user}")
            print(f"Najbliższy użytkownik z kNN: {decision.predicted_user}")
            print(f"Score deklarowanego użytkownika: {decision.score:.6f}")
            print(f"Decyzja: {'zgodny' if decision.matched else 'niezgodny'}")

        elif args.task == "keystrokes-loo":
            loo_k_values = parse_k_values(args.k_values)
            run_keystrokes_leave_one_out(df, k_values=loo_k_values)
    else:
        run_segmentation(person_filter=args.person, limit=args.limit)


if __name__ == "__main__":
    main()
