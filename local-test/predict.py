"""
BM2-Net Local Image Measurement Prediction CLI.

Run inference using best_model.pt on your local photos or silhouette masks
to obtain complete body measurements (14 key anthropometric metrics).

Examples:
    # Basic usage with photos or masks:
    python local-test/predict.py --frontal my_front.jpg --lateral my_side.jpg --height 178 --weight 75

    # Usage with imperial units:
    python local-test/predict.py --frontal front.png --lateral side.png --height "5ft 10" --weight 165lbs

    # Interactive mode (prompts for inputs):
    python local-test/predict.py
"""

import argparse
import csv
import datetime
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import (
    MEASUREMENT_META,
    MeasurementPredictor,
    parse_height,
    parse_weight,
    render_report_image,
)


def print_banner():
    banner = r"""
========================================================================
    ____  __  ___ ___         _   __     __ 
   / __ )/  |/  /|__ \       / | / /__  / /_
  / __  / /|_/ / __/ /______/  |/ / _ \/ __/
 / /_/ / /  / / / __//_____/ /|  /  __/ /_  
/_____/_/  /_/ /____/     /_/ |_/\___/\__/  
      Body Measurement Estimation from Dual-View Silhouettes
========================================================================
"""
    print(banner)


def print_table(result: dict):
    """Print an aesthetic ASCII table of results."""
    meta = result["metadata"]
    measurements = result["measurements"]

    print("\n" + "=" * 76)
    print(f"  SUBJECT PHYSICAL PARAMETERS")
    print(f"  Height: {meta['height_cm']} cm ({meta['height_ft_in']})  |  Weight: {meta['weight_kg']} kg ({meta['weight_lbs']} lbs)  |  BMI: {meta['bmi']}")
    print(f"  Model:  {meta['checkpoint']}  |  Device: {meta['device']}")
    print("=" * 76)
    print(f" {'#':<3} | {'Measurement':<26} | {'Metric (cm)':<15} | {'Imperial (in)':<15} | {'Type'}")
    print("-" * 76)

    for idx, (name, item) in enumerate(measurements.items(), start=1):
        print(
            f" {idx:<3} | {item['label']:<26} | {item['cm']:>8.2f} cm      | {item['inches']:>8.2f} in      | {item['type']}"
        )

    print("=" * 76 + "\n")


def save_results(result: dict, output_dir: Path, generate_vis: bool = True):
    """Save JSON, CSV, silhouette masks, and visual report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save JSON
    json_path = output_dir / f"measurements_{timestamp}.json"
    latest_json_path = output_dir / "measurements_latest.json"

    export_dict = {
        "metadata": result["metadata"],
        "measurements": {
            k: {
                "label": v["label"],
                "value_cm": v["cm"],
                "value_inches": v["inches"],
                "type": v["type"],
                "description": v["desc"],
            }
            for k, v in result["measurements"].items()
        },
    }

    with open(json_path, "w") as f:
        json.dump(export_dict, f, indent=2)
    with open(latest_json_path, "w") as f:
        json.dump(export_dict, f, indent=2)

    # 2. Save CSV
    csv_path = output_dir / f"measurements_{timestamp}.csv"
    latest_csv_path = output_dir / "measurements_latest.csv"

    rows = []
    for k, v in result["measurements"].items():
        rows.append({
            "Key": k,
            "Measurement": v["label"],
            "Value_cm": v["cm"],
            "Value_inches": v["inches"],
            "Type": v["type"],
            "Description": v["desc"],
        })

    for p in [csv_path, latest_csv_path]:
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Key", "Measurement", "Value_cm", "Value_inches", "Type", "Description"])
            writer.writeheader()
            writer.writerows(rows)

    # 3. Save silhouettes
    sil_dir = output_dir / "silhouettes"
    sil_dir.mkdir(parents=True, exist_ok=True)
    f_sil_path = sil_dir / f"frontal_{timestamp}.png"
    l_sil_path = sil_dir / f"lateral_{timestamp}.png"
    result["silhouettes"]["frontal"].save(f_sil_path)
    result["silhouettes"]["lateral"].save(l_sil_path)

    # 4. Save Visual Report
    report_path = None
    if generate_vis:
        report_path = output_dir / f"report_{timestamp}.png"
        render_report_image(result, report_path)
        latest_report_path = output_dir / "report_latest.png"
        render_report_image(result, latest_report_path)

    print("[SUCCESS] Results saved successfully:")
    print(f"  • JSON Report:   {json_path.resolve()}")
    print(f"  • CSV Table:     {csv_path.resolve()}")
    print(f"  • Silhouettes:   {sil_dir.resolve()}")
    if report_path:
        print(f"  • Visual Card:   {report_path.resolve()}")

    return json_path, csv_path, report_path


def prompt_user_input():
    """Interactive prompt for inputs if not supplied via CLI."""
    print("Please provide the required inputs below (drag & drop image files into terminal):\n")

    while True:
        f_in = input("1. Frontal photo / mask path: ").strip().strip('"').strip("'")
        if f_in and Path(f_in).exists():
            break
        print("   File not found. Please enter a valid path.")

    while True:
        l_in = input("2. Lateral (side) photo / mask path: ").strip().strip('"').strip("'")
        if l_in and Path(l_in).exists():
            break
        print("   File not found. Please enter a valid path.")

    while True:
        h_in = input("3. Height (e.g. 175 cm or 5ft 10): ").strip()
        try:
            parse_height(h_in)
            break
        except ValueError as e:
            print(f"   {e}. Please try again.")

    while True:
        w_in = input("4. Weight (e.g. 72 kg or 158 lbs): ").strip()
        try:
            parse_weight(w_in)
            break
        except ValueError as e:
            print(f"   {e}. Please try again.")

    return f_in, l_in, h_in, w_in


def main():
    parser = argparse.ArgumentParser(
        description="BM2-Net Local Image Measurement Predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-f", "--frontal", type=str, default=None, help="Path to frontal image or silhouette mask")
    parser.add_argument("-l", "--lateral", type=str, default=None, help="Path to lateral (side) image or silhouette mask")
    parser.add_argument("-H", "--height", type=str, default=None, help="Subject height (e.g. 175, 175cm, '5ft 10', 70in)")
    parser.add_argument("-W", "--weight", type=str, default=None, help="Subject weight (e.g. 70, 70kg, 154lbs)")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("-o", "--output-dir", type=str, default="local-test/output", help="Directory to save output files")
    parser.add_argument("--no-vis", action="store_true", help="Disable generating the visual report infographic")
    parser.add_argument("--device", type=str, default=None, help="Inference device: 'cuda' or 'cpu'")

    args = parser.parse_args()
    print_banner()

    frontal_path = args.frontal
    lateral_path = args.lateral
    height_val = args.height
    weight_val = args.weight

    # Check if arguments need interactive prompt
    if not (frontal_path and lateral_path and height_val and weight_val):
        if sys.stdin.isatty():
            frontal_path, lateral_path, height_val, weight_val = prompt_user_input()
        else:
            parser.error("Missing required arguments. Run with --frontal, --lateral, --height, and --weight.")

    print(f"Initializing BM2-Net Predictor...")
    predictor = MeasurementPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print(f"Loaded checkpoint '{predictor.checkpoint_path.name}' (Validation MAE: {predictor.best_mae:.3f} cm)")
    print(f"Running inference on {predictor.device}...")

    # Run inference
    result = predictor.predict(
        frontal_input=frontal_path,
        lateral_input=lateral_path,
        height=height_val,
        weight=weight_val,
    )

    # Display results
    print_table(result)

    # Save output artifacts
    save_results(
        result=result,
        output_dir=Path(args.output_dir),
        generate_vis=not args.no_vis,
    )


if __name__ == "__main__":
    main()
