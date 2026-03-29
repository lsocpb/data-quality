from pathlib import Path
import csv
import statistics

import matplotlib.pyplot as plt

from src.brisque import BrisqueEvaluator
from src.noise import NoiseEvaluator


ROOT_DIR = "faces"
MODEL_PATH = "model/brisque_model_live.yml"
RANGE_PATH = "model/brisque_range_live.yml"

OUTPUT_CSV = "wyniki_brisque.csv"
NOISE_CSV = "wyniki_noise.csv"
SUMMARY_CSV = "podsumowanie.csv"
RANKING_CSV = "ranking_osob.csv"

CHART_MEAN_PATH = "wykres_sredni_brisque.png"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_persons(root_dir: str):
    root = Path(root_dir)
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_images(person_dir: Path):
    return sorted([
        p for p in person_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])


def save_csv(rows, path: str):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def summarize(results):
    grouped = {}

    for r in results:
        grouped.setdefault(r["osoba"], []).append(r["brisque"])

    summary = []
    for osoba, scores in sorted(grouped.items()):
        summary.append({
            "osoba": osoba,
            "liczba_zdjec": len(scores),
            "srednia_brisque": round(statistics.mean(scores), 4),
            "mediana_brisque": round(statistics.median(scores), 4),
            "min_brisque": round(min(scores), 4),
            "max_brisque": round(max(scores), 4),
        })

    return summary


def create_ranking(summary_rows):
    ranking = sorted(summary_rows, key=lambda x: x["srednia_brisque"])

    ranking_rows = []
    for idx, row in enumerate(ranking, start=1):
        ranking_rows.append({
            "miejsce": idx,
            "osoba": row["osoba"],
            "liczba_zdjec": row["liczba_zdjec"],
            "srednia_brisque": row["srednia_brisque"],
            "mediana_brisque": row["mediana_brisque"],
            "min_brisque": row["min_brisque"],
            "max_brisque": row["max_brisque"],
        })

    return ranking_rows


def plot_mean_brisque(summary_rows, output_path: str):
    ranking = sorted(summary_rows, key=lambda x: x["srednia_brisque"])
    persons = [row["osoba"] for row in ranking]
    means = [row["srednia_brisque"] for row in ranking]

    plt.figure(figsize=(12, 6))
    plt.bar(persons, means)
    plt.xlabel("Osoba")
    plt.ylabel("Średni BRISQUE")
    plt.title("Średni wynik BRISQUE dla każdej osoby")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_ranking(ranking_rows):
    print("\n=== RANKING OSÓB ===")
    for row in ranking_rows:
        print(
            f'{row["miejsce"]:>2}. {row["osoba"]:<15} '
            f'średnia BRISQUE: {row["srednia_brisque"]:.4f}'
        )


def main():
    evaluator = BrisqueEvaluator(MODEL_PATH, RANGE_PATH)
    noise_evaluator = NoiseEvaluator()
    results = []
    noise_results = []

    for person_dir in list_persons(ROOT_DIR):
        print(f"\nOsoba: {person_dir.name}")
        images = list_images(person_dir)

        for img in images:
            try:
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
        print("Brak poprawnie przetworzonych zdjęć.")
        return

    save_csv(results, OUTPUT_CSV)
    save_csv(noise_results, NOISE_CSV)

    summary_rows = summarize(results)
    save_csv(summary_rows, SUMMARY_CSV)

    ranking_rows = create_ranking(summary_rows)
    save_csv(ranking_rows, RANKING_CSV)

    plot_mean_brisque(summary_rows, CHART_MEAN_PATH)


    print_ranking(ranking_rows)

    print("\nZakończono.")
    print(f"Wyniki zdjęć: {OUTPUT_CSV}")
    print(f"Podsumowanie osób: {SUMMARY_CSV}")
    print(f"Ranking osób: {RANKING_CSV}")
    print(f"Wykres średnich: {CHART_MEAN_PATH}")
    print(f"Wyniki szumu: {NOISE_CSV}")


if __name__ == "__main__":
    main()