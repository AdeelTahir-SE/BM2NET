import sys
import numpy as np
from PIL import Image
from rembg import remove, new_session

def make_silhouette(in_path, out_path, thresh=128):
    session = new_session("u2net_human_seg")
    img = Image.open(in_path).convert("RGB")

    # Get person mask (white = person)
    mask = np.array(remove(img, session=session, only_mask=True))

    # Person = black (0), background = white (255)
    sil = np.where(mask > thresh, 0, 255).astype(np.uint8)
    Image.fromarray(sil, mode="L").save(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "silhouette.png"
    make_silhouette(inp, out)