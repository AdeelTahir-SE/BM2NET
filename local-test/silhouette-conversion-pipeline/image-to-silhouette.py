import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

def make_silhouette(in_path: str | Path, out_path: str | Path, thresh: int = 128, invert: bool = False):
    """
    Extract person silhouette from an input image.

    BodyM Convention:
        Person / Foreground = 255 (white)
        Background          = 0 (black)
    """
    try:
        from rembg import remove, new_session
    except ImportError:
        print("Error: 'rembg' is not installed in the current environment.")
        print("Please install it with: pip install rembg")
        sys.exit(1)

    session = new_session("u2net_human_seg")
    img = Image.open(in_path).convert("RGB")

    # Get person mask (white = person, 255)
    mask = np.array(remove(img, session=session, only_mask=True))

    if invert:
        # Person = 0, background = 255
        sil = np.where(mask > thresh, 0, 255).astype(np.uint8)
    else:
        # Standard BodyM format: Person = 255, background = 0
        sil = np.where(mask > thresh, 255, 0).astype(np.uint8)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sil, mode="L").save(out_p)
    print(f"Saved silhouette to: {out_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an image to a body silhouette mask")
    parser.add_argument("input", type=str, help="Path to input photo")
    parser.add_argument("output", type=str, nargs="?", default="silhouette.png", help="Path to output silhouette PNG")
    parser.add_argument("--thresh", type=int, default=128, help="Threshold for binarization (0-255)")
    parser.add_argument("--invert", action="store_true", help="Invert output (person=black, bg=white)")

    args = parser.parse_args()
    make_silhouette(args.input, args.output, thresh=args.thresh, invert=args.invert)
