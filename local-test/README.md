# BM2-Net Local Image Testing & Body Measurement Pipeline

Use this pipeline to estimate your full body measurements (14 key anthropometric metrics) from your local photos or silhouette masks using your trained `best_model.pt`.

---

## 🚀 Quick Verification

Run a sample test out-of-the-box using the bundled sample masks:

```bash
python local-test/sample_run.py
```

This will run the model on `local-test/input/sample_frontal.png` & `sample_lateral.png`, print the measurement table, and generate your visual report card in `local-test/output/`.

---

## 📸 Step 1: Taking Your Photos for Best Accuracy

The BM2-Net architecture relies on dual silhouettes (frontal + lateral views) together with your actual height and weight. For best results:

| View | Posture | Tips |
| :--- | :--- | :--- |
| **Frontal** | Stand facing the camera straight on in an **"A-pose"** (arms slightly separated from torso, feet shoulder-width apart). | Camera at waist/chest height, roughly 2.5–3 meters away. |
| **Lateral (Side)** | Turn **90° sideways**, head facing forward, arms resting naturally along sides. | Maintain the same camera distance and level. |

> **Clothing Tip:** Wear fitted activewear, swimwear, or snug-fitting clothing so your body contour is clearly defined.

---

## 🏃 Step 2: Running the Measurement Predictor

### Option A: Command-Line Flags (Recommended)

```bash
# Using metric units (cm and kg):
python local-test/predict.py --frontal path/to/front.jpg --lateral path/to/side.jpg --height 175 --weight 70

# Using imperial units (feet/inches and lbs):
python local-test/predict.py --frontal path/to/front.jpg --lateral path/to/side.jpg --height "5ft 10" --weight 165lbs
```

### Option B: Interactive Mode

Simply run without arguments and follow the prompts:

```bash
python local-test/predict.py
```

You can drag-and-drop your image files directly into the terminal when prompted!

---

## 📐 Supported Units

- **Height:**
  - Centimeters: `178`, `178cm`
  - Feet & Inches: `"5ft 10"`, `"5'10\""`, `"5ft 9.5in"`
  - Total Inches: `70in`
  - Meters: `1.78m`
- **Weight:**
  - Kilograms: `75`, `75kg`
  - Pounds: `165lbs`, `165 lb`

---

## 🎭 How Silhouettes Are Handled

1. **Auto-Segmentation from Photos:**
   If you pass normal RGB photos (e.g. `.jpg`, `.png`, `.avif`), the script will attempt to automatically isolate your silhouette using `rembg`.
   > *Note:* If `rembg` is not installed, install it with `pip install rembg` or provide silhouette masks directly.

2. **Transparent PNGs:**
   If your image has a transparent background (e.g., from iOS "Lift Subject", Photoshop, or remove.bg), the alpha channel is automatically converted to the exact silhouette mask.

3. **Pre-extracted Silhouette Masks:**
   If you already have binary masks, pass them directly. The pipeline automatically checks corner pixels to determine polarity:
   - BodyM format: **Foreground (Person) = White (255)**, **Background = Black (0)**.
   - If an inverted mask is detected (white background, black person), the script **automatically inverts it** for you so the model receives the correct representation.

---

## 📊 What You Get: 14 Measurements

Every prediction calculates:

| Measurement | Anthropometric Description |
| :--- | :--- |
| **Chest Circumference** | Around fullest part of chest / bust |
| **Waist Circumference** | Around narrowest waist indentation |
| **Hip Circumference** | Around widest point of hips / buttocks |
| **Shoulder Breadth** | Distance across back between acromion points |
| **Shoulder to Crotch** | Vertical torso length |
| **Arm Length** | Shoulder tip to wrist bone |
| **Leg Length** | Inseam / outseam leg length |
| **Bicep Circumference** | Maximum upper arm girth |
| **Forearm Circumference** | Maximum forearm girth |
| **Wrist Circumference** | Around wrist crease |
| **Thigh Circumference** | Upper thigh girth below gluteal fold |
| **Calf Circumference** | Maximum calf girth |
| **Ankle Circumference** | Minimum ankle girth |
| **Body Height** | Total standing height estimate |

---

## 📁 Generated Outputs (`local-test/output/`)

After each run, the following files are produced:

1. **`report_latest.png` / `report_<timestamp>.png`:** A dark-themed infographic card containing your dual-view silhouettes, subject stats (Height, Weight, BMI), and a formatted table of all 14 measurements in both metric (cm) and imperial (inches).
2. **`measurements_latest.json`:** Structured JSON containing all numerical values and metadata.
3. **`measurements_latest.csv`:** Tabular CSV for spreadsheet tools (Excel, Google Sheets).
4. **`silhouettes/`:** The normalized silhouette masks used for model inference.

---

## ⚙️ Advanced CLI Options

| Argument | Shorthand | Default | Description |
| :--- | :--- | :--- | :--- |
| `--frontal` | `-f` | None | Path to frontal photo or mask |
| `--lateral` | `-l` | None | Path to lateral photo or mask |
| `--height` | `-H` | None | Height string (cm, ft'in, in) |
| `--weight` | `-W` | None | Weight string (kg, lbs) |
| `--checkpoint`| `-c` | `train/checkpoints/best_model.pt` | Path to custom checkpoint |
| `--output-dir`| `-o` | `local-test/output` | Output directory for results |
| `--no-vis` | | False | Skip generating the PNG infographic |
| `--device` | | Auto (cuda/cpu) | Target compute device |
