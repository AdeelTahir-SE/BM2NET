"""
BM2-Net Local Inference Engine.

Handles silhouette preprocessing, height/weight normalization,
forward pass with best_model.pt, denormalization to cm/inches,
and generation of visual measurement reports.
"""

import datetime
import math
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.config import TrainConfig
from train.model import build_model
from train.utils import (
    denormalize_measurements,
    load_checkpoint,
    load_hwg_stats,
    load_measurement_stats,
    load_pixel_stats,
)

# Human-friendly descriptions for each measurement
MEASUREMENT_META = {
    "height": {"label": "Body Height", "type": "Linear", "desc": "Total standing height"},
    "chest": {"label": "Chest Circumference", "type": "Circumference", "desc": "Around fullest part of chest/bust"},
    "waist": {"label": "Waist Circumference", "type": "Circumference", "desc": "Around narrowest waist point"},
    "hip": {"label": "Hip Circumference", "type": "Circumference", "desc": "Around widest point of hips/buttocks"},
    "shoulder-breadth": {"label": "Shoulder Breadth", "type": "Linear", "desc": "Distance between acromion points"},
    "shoulder-to-crotch": {"label": "Shoulder to Crotch", "type": "Linear", "desc": "Vertical torso length"},
    "arm-length": {"label": "Arm Length", "type": "Linear", "desc": "From shoulder tip to wrist bone"},
    "leg-length": {"label": "Leg Length", "type": "Linear", "desc": "From hip/crotch to ankle base"},
    "bicep": {"label": "Bicep Circumference", "type": "Circumference", "desc": "Around maximum arm circumference"},
    "forearm": {"label": "Forearm Circumference", "type": "Circumference", "desc": "Around maximum forearm girth"},
    "wrist": {"label": "Wrist Circumference", "type": "Circumference", "desc": "Around wrist crease"},
    "thigh": {"label": "Thigh Circumference", "type": "Circumference", "desc": "Around upper thigh below gluteal fold"},
    "calf": {"label": "Calf Circumference", "type": "Circumference", "desc": "Around maximum calf bulge"},
    "ankle": {"label": "Ankle Circumference", "type": "Circumference", "desc": "Around smallest ankle girth above malleolus"},
}


def parse_height(value: Union[str, float, int]) -> float:
    """
    Parse height string into centimeters (float).
    Supports:
        - 175, 175.5, '175cm', '1.75m'
        - '5ft 9', "5'9\"", '5ft 9in', '5\'10', '68in'
    """
    if isinstance(value, (int, float)):
        return float(value)

    val = str(value).strip().lower()

    # If it's a pure numeric string, it is directly centimeters
    try:
        return float(val)
    except ValueError:
        pass

    # Strip cm if present (e.g. '175cm', '175.5 cm')
    if val.endswith("cm"):
        try:
            return float(val[:-2].strip())
        except ValueError:
            pass

    # Meter string like '1.75m'
    if val.endswith("m") and not val.endswith("cm"):
        try:
            return round(float(val[:-1].strip()) * 100.0, 2)
        except ValueError:
            pass

    # Inch string like '70in' or '70"'
    if val.endswith("in") or val.endswith('"'):
        num_part = val.rstrip('in"').strip()
        try:
            return round(float(num_part) * 2.54, 2)
        except ValueError:
            pass

    # Feet and inches like 5'9", 5ft 9, 5' 9", 5ft9in, 5'10
    ft_in_pattern = r"""^(\d+)\s*(?:ft|')\s*(?:(\d+(?:\.\d+)?)\s*(?:in|")?)?$"""
    ft_in_match = re.match(ft_in_pattern, val)
    if ft_in_match:
        ft = float(ft_in_match.group(1))
        inches = float(ft_in_match.group(2) or 0)
        total_inches = ft * 12.0 + inches
        return round(total_inches * 2.54, 2)

    raise ValueError(f"Cannot parse height from '{value}'. Expected e.g. 175, 175cm, or 5ft 9.")


def parse_weight(value: Union[str, float, int]) -> float:
    """
    Parse weight string into kilograms (float).
    Supports:
        - 70, 70.5, '70kg'
        - '154lbs', '154 lb', '154 pounds'
    """
    if isinstance(value, (int, float)):
        return float(value)

    val = str(value).strip().lower()

    # Pounds
    for suffix in ["lbs", "lb", "pounds", "pound"]:
        if val.endswith(suffix):
            num_str = val[:-len(suffix)].strip()
            return round(float(num_str) * 0.45359237, 2)

    # Strip kg
    if val.endswith("kg"):
        val = val[:-2].strip()

    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Cannot parse weight from '{value}'. Expected e.g. 70, 70kg, or 154lbs.")


def process_image_to_silhouette(
    image_input: Union[str, Path, Image.Image],
    view_name: str = "view",
) -> Tuple[Image.Image, bool]:
    """
    Load an image and guarantee a proper BodyM silhouette format:
        Foreground (Person) = 255 (White)
        Background          = 0 (Black)

    Returns:
        (silhouette_image_L_mode, was_segmented_with_rembg)
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.exists():
            raise FileNotFoundError(f"Input image not found: {img_path}")
        img = Image.open(img_path)
    else:
        img = image_input

    was_segmented = False

    # Check if image has an alpha channel that already separates subject
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        alpha = np.array(img.convert("RGBA"))[:, :, 3]
        if np.any(alpha < 250) and np.any(alpha > 10):
            # Useful alpha channel present
            sil = np.where(alpha > 128, 255, 0).astype(np.uint8)
            return Image.fromarray(sil, mode="L"), False

    # Check if already a grayscale silhouette or photo
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)
    unique_vals = np.unique(arr)

    # If it has only a few discrete levels (e.g. [0, 255]), it's already a silhouette mask
    is_already_mask = len(unique_vals) <= 10

    if not is_already_mask:
        # It's an RGB/photo: attempt automatic background removal
        try:
            from rembg import new_session, remove
            session = new_session("u2net_human_seg")
            mask = np.array(remove(img.convert("RGB"), session=session, only_mask=True))
            arr = np.where(mask > 128, 255, 0).astype(np.uint8)
            was_segmented = True
        except ImportError:
            # rembg not installed: use thresholding fallback with notice
            print(
                f"[{view_name}] 'rembg' not installed. Treating image as direct silhouette mask. "
                "For automatic photo segmentation, install rembg (`pip install rembg`)."
            )
            arr = np.where(arr > 128, 255, 0).astype(np.uint8)

    # Check polarity: in BodyM, corners are background (0) and person is 255.
    # If the corners are predominantly white (255), invert it so background is 0.
    h, w = arr.shape
    corner_samples = [
        arr[:5, :5],
        arr[:5, -5:],
        arr[-5:, :5],
        arr[-5:, -5:],
    ]
    corner_mean = np.mean([np.mean(c) for c in corner_samples])
    if corner_mean > 127:
        arr = 255 - arr

    return Image.fromarray(arr, mode="L"), was_segmented


class MeasurementPredictor:
    """
    Inference wrapper for BM2-Net body measurement prediction.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        data_root: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.project_root = PROJECT_ROOT

        # Locate checkpoint
        if checkpoint_path is None:
            checkpoint_path = self.project_root / "train" / "checkpoints" / "best_model.pt"
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        # Locate data_root for normalization stats
        if data_root is None:
            data_root = self.project_root / "processed-bodym"
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data statistics directory not found at: {self.data_root}")

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Config & Stats
        self.config = TrainConfig(
            project_root=str(self.project_root),
            data_root=str(self.data_root),
            device=str(self.device),
        )
        self.pixel_mean, self.pixel_std = load_pixel_stats(self.data_root)
        self.hwg_stats = load_hwg_stats(self.data_root)
        self.measurement_stats = load_measurement_stats(self.data_root)

        # Build & load model
        self.model = build_model(self.config, pretrained=False)
        load_info = load_checkpoint(self.checkpoint_path, self.model)
        self.model.to(self.device)
        self.model.eval()

        self.epoch = load_info.get("epoch", 0)
        self.best_mae = load_info.get("best_mae", 0.0)

    def prepare_input_tensor(
        self,
        frontal_sil: Image.Image,
        lateral_sil: Image.Image,
        height_cm: float,
        weight_kg: float,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Construct the composite 3×640×960 input tensor for BM2-Net:
            Ch0: [frontal (640, 480) | lateral (640, 480)] normalized
            Ch1: height broadcast (640, 960) normalized
            Ch2: weight broadcast (640, 960) normalized

        Returns:
            (tensor_1x3x640x960, composite_sil_array_640x960)
        """
        # Resize to standard model dimensions (480w x 640h)
        f_resized = frontal_sil.resize((self.config.img_width, self.config.img_height), Image.BILINEAR)
        l_resized = lateral_sil.resize((self.config.img_width, self.config.img_height), Image.BILINEAR)

        f_raw = np.array(f_resized, dtype=np.float32) / 255.0
        l_raw = np.array(l_resized, dtype=np.float32) / 255.0

        # Normalize pixels using training statistics
        f_norm = (f_raw - self.pixel_mean) / self.pixel_std
        l_norm = (l_raw - self.pixel_mean) / self.pixel_std

        # Channel 0: Horizontal concatenation
        ch0 = np.concatenate([f_norm, l_norm], axis=1)  # (640, 960)

        # Channel 1: Height constant image
        h_norm = (height_cm - self.hwg_stats["height_cm"]["mean"]) / self.hwg_stats["height_cm"]["std"]
        ch1 = np.full((self.config.input_height, self.config.input_width), h_norm, dtype=np.float32)

        # Channel 2: Weight constant image
        w_norm = (weight_kg - self.hwg_stats["weight_kg"]["mean"]) / self.hwg_stats["weight_kg"]["std"]
        ch2 = np.full((self.config.input_height, self.config.input_width), w_norm, dtype=np.float32)

        composite_tensor = torch.from_numpy(np.stack([ch0, ch1, ch2], axis=0)).unsqueeze(0).to(self.device)
        composite_vis = np.concatenate([np.array(f_resized), np.array(l_resized)], axis=1)

        return composite_tensor, composite_vis

    @torch.no_grad()
    def predict(
        self,
        frontal_input: Union[str, Path, Image.Image],
        lateral_input: Union[str, Path, Image.Image],
        height: Union[str, float, int],
        weight: Union[str, float, int],
    ) -> Dict:
        """
        Run full prediction pipeline.

        Returns:
            Dictionary with predicted measurements (cm & inches), subject metadata,
            and intermediate silhouette images.
        """
        height_cm = parse_height(height)
        weight_kg = parse_weight(weight)

        # Process silhouettes
        frontal_sil, f_segmented = process_image_to_silhouette(frontal_input, view_name="Frontal")
        lateral_sil, l_segmented = process_image_to_silhouette(lateral_input, view_name="Lateral")

        # Build tensor
        tensor, composite_vis = self.prepare_input_tensor(frontal_sil, lateral_sil, height_cm, weight_kg)

        # Model forward pass
        raw_pred = self.model(tensor)

        # Denormalize measurements to cm
        pred_cm = denormalize_measurements(raw_pred, self.measurement_stats, self.config.measurement_names)
        pred_cm = pred_cm.squeeze(0).cpu().numpy()

        # Build structured results
        measurements = {}
        for idx, name in enumerate(self.config.measurement_names):
            val_cm = float(pred_cm[idx])
            val_in = val_cm / 2.54
            meta = MEASUREMENT_META.get(name, {"label": name.replace("-", " ").title(), "type": "Linear", "desc": ""})
            measurements[name] = {
                "name": name,
                "label": meta["label"],
                "type": meta["type"],
                "desc": meta["desc"],
                "cm": round(val_cm, 2),
                "inches": round(val_in, 2),
            }

        # Calculate BMI
        height_m = height_cm / 100.0
        bmi = round(weight_kg / (height_m * height_m), 1)

        return {
            "metadata": {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "height_cm": height_cm,
                "height_ft_in": f"{int(height_cm / 2.54 // 12)}'{round(height_cm / 2.54 % 12, 1)}\"",
                "weight_kg": weight_kg,
                "weight_lbs": round(weight_kg * 2.20462, 1),
                "bmi": bmi,
                "checkpoint": str(self.checkpoint_path.name),
                "device": str(self.device),
            },
            "measurements": measurements,
            "silhouettes": {
                "frontal": frontal_sil,
                "lateral": lateral_sil,
                "composite": Image.fromarray(composite_vis, mode="L"),
            },
        }


def render_report_image(
    result: Dict,
    save_path: Union[str, Path],
    frontal_orig: Optional[Image.Image] = None,
    lateral_orig: Optional[Image.Image] = None,
) -> Path:
    """
    Render a clean visual infographic with dual views, subject stats,
    and a measurement table card.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    meta = result["metadata"]
    measurements = result["measurements"]

    # Fonts
    font_large = None
    font_medium = None
    font_small = None
    font_bold = None

    for font_cand in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "arial.ttf"]:
        if Path(font_cand).exists():
            try:
                font_large = ImageFont.truetype(font_cand, 26)
                font_medium = ImageFont.truetype(font_cand, 18)
                font_small = ImageFont.truetype(font_cand, 14)
                font_bold = ImageFont.truetype(font_cand, 15)
                break
            except Exception:
                pass

    if font_large is None:
        font_large = font_medium = font_small = font_bold = ImageFont.load_default()

    # Dimensions
    canvas_w = 1200
    canvas_h = 750
    card = Image.new("RGB", (canvas_w, canvas_h), color=(18, 24, 38))  # Dark sleek theme
    draw = ImageDraw.Draw(card)

    # ── Header ─────────────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (canvas_w, 75)], fill=(28, 38, 58))
    draw.text((30, 16), "BM2-Net Body Measurement Prediction", fill=(255, 255, 255), font=font_large)
    timestamp_str = f"Date: {meta['timestamp']} | Checkpoint: {meta['checkpoint']}"
    draw.text((30, 48), timestamp_str, fill=(160, 180, 205), font=font_small)

    # ── Left Column: Silhouette & Dual-View Previews ──────────────────────
    preview_box_x = 30
    preview_box_y = 95
    box_w = 500
    box_h = 620

    # Draw panel container
    draw.rectangle(
        [(preview_box_x, preview_box_y), (preview_box_x + box_w, preview_box_y + box_h)],
        fill=(24, 32, 48),
        outline=(45, 60, 85),
        width=2,
    )
    draw.text((preview_box_x + 16, preview_box_y + 12), "Input Silhouettes (Frontal & Lateral)", fill=(220, 230, 245), font=font_medium)

    # Resize composite view to fit in box
    comp_pil = result["silhouettes"]["composite"].convert("RGB")
    target_comp_w = 460
    target_comp_h = int(target_comp_w * (640 / 960))  # maintain aspect
    comp_resized = comp_pil.resize((target_comp_w, target_comp_h), Image.BILINEAR)

    img_pos_x = preview_box_x + (box_w - target_comp_w) // 2
    img_pos_y = preview_box_y + 55
    card.paste(comp_resized, (img_pos_x, img_pos_y))

    # Add labels below images
    half_w = target_comp_w // 2
    draw.text((img_pos_x + half_w // 2 - 35, img_pos_y + target_comp_h + 8), "Frontal View", fill=(130, 175, 225), font=font_bold)
    draw.text((img_pos_x + half_w + half_w // 2 - 35, img_pos_y + target_comp_h + 8), "Lateral View", fill=(130, 175, 225), font=font_bold)

    # Stats banner in left box
    stats_y = img_pos_y + target_comp_h + 45
    draw.rectangle([(img_pos_x, stats_y), (img_pos_x + target_comp_w, stats_y + 175)], fill=(32, 42, 64), outline=(50, 70, 100))
    draw.text((img_pos_x + 16, stats_y + 12), "Subject Physical Parameters", fill=(240, 245, 255), font=font_bold)

    stat_lines = [
        f"Height:  {meta['height_cm']} cm  ({meta['height_ft_in']})",
        f"Weight:  {meta['weight_kg']} kg  ({meta['weight_lbs']} lbs)",
        f"Body Mass Index (BMI):  {meta['bmi']}",
        f"Inference Device:  {meta['device']}",
    ]
    for idx, line in enumerate(stat_lines):
        draw.text((img_pos_x + 16, stats_y + 40 + idx * 28), line, fill=(185, 205, 230), font=font_small)

    # ── Right Column: Measurements Table ───────────────────────────────────
    table_x = 555
    table_y = 95
    table_w = 615
    table_h = 620

    draw.rectangle([(table_x, table_y), (table_x + table_w, table_y + table_h)], fill=(24, 32, 48), outline=(45, 60, 85), width=2)
    draw.text((table_x + 18, table_y + 12), "Predicted Body Measurements (14 Keypoints)", fill=(220, 230, 245), font=font_medium)

    # Table Header
    header_y = table_y + 45
    draw.rectangle([(table_x + 10, header_y), (table_x + table_w - 10, header_y + 32)], fill=(36, 48, 72))
    draw.text((table_x + 25, header_y + 7), "Measurement", fill=(255, 255, 255), font=font_bold)
    draw.text((table_x + 310, header_y + 7), "Metric (cm)", fill=(255, 255, 255), font=font_bold)
    draw.text((table_x + 470, header_y + 7), "Imperial (in)", fill=(255, 255, 255), font=font_bold)

    # Rows
    row_y = header_y + 36
    row_h = 36
    for idx, (name, item) in enumerate(measurements.items()):
        bg_col = (28, 38, 56) if idx % 2 == 0 else (22, 30, 44)
        draw.rectangle([(table_x + 10, row_y), (table_x + table_w - 10, row_y + row_h - 2)], fill=bg_col)

        # Measurement Label
        draw.text((table_x + 25, row_y + 8), item["label"], fill=(230, 235, 245), font=font_small)

        # CM value
        cm_str = f"{item['cm']:.2f} cm"
        draw.text((table_x + 310, row_y + 8), cm_str, fill=(75, 210, 160), font=font_bold)

        # Inches value
        in_str = f"{item['inches']:.2f} in"
        draw.text((table_x + 470, row_y + 8), in_str, fill=(110, 190, 255), font=font_small)

        row_y += row_h

    # Save
    card.save(save_path)
    return save_path
