"""
Quick sample run demonstration for BM2-Net local inference.

Uses the sample images in `local-test/input/` and runs the prediction
pipeline, generating visual card, CSV, and JSON outputs.
"""

import sys
from pathlib import Path

# Add project root and local-test to path
LOCAL_TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LOCAL_TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(LOCAL_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_TEST_DIR))

from inference import MeasurementPredictor
from predict import print_banner, print_table, save_results


def run_sample():
    print_banner()
    print("Running quick sample verification with best_model.pt...\n")

    input_dir = LOCAL_TEST_DIR / "input"
    frontal_path = input_dir / "sample_frontal.png"
    lateral_path = input_dir / "sample_lateral.png"

    if not frontal_path.exists() or not lateral_path.exists():
        print(f"Sample images missing in {input_dir}.")
        return

    # Sample subject metadata:
    height = "160.9 cm"
    weight = "61.95 kg"

    print(f"Sample Input Frontal: {frontal_path.name}")
    print(f"Sample Input Lateral: {lateral_path.name}")
    print(f"Sample Height:        {height}")
    print(f"Sample Weight:        {weight}\n")

    predictor = MeasurementPredictor()
    result = predictor.predict(
        frontal_input=frontal_path,
        lateral_input=lateral_path,
        height=height,
        weight=weight,
    )

    print_table(result)

    out_dir = LOCAL_TEST_DIR / "output"
    save_results(result, output_dir=out_dir, generate_vis=True)

    print("\n[SUCCESS] Sample run completed! Check 'local-test/output/' for generated artifacts.")


if __name__ == "__main__":
    run_sample()
