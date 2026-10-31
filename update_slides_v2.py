"""Insert real OceanEmbed screenshots into SIH presentation Slide 3 - v2."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
from pptx import Presentation
from pptx.util import Inches, Emu
from PIL import Image
import io as bio

BASE = os.path.dirname(os.path.abspath(__file__))
SS_DIR = os.path.join(BASE, "screenshots")

PPTX_SRC = os.path.join(BASE, "SIH2026-OceanEmbed-Final (3).pptx")
PPTX_OUT = os.path.join(BASE, "SIH2026-OceanEmbed-Final-With-Prototype-Screenshots.pptx")


def insert_screenshots():
    """Insert screenshots into Slide 3 using the full rectangle area."""
    prs = Presentation(PPTX_SRC)
    slide3 = prs.slides[2]

    # From inspection:
    # Rounded Rectangle 17410 (background for all 4 screenshots):
    #   pos=(4.37in, 1.34in), size=(8.87in x 1.91in)
    # Labels are at y=1.94in with [Insert screenshot] text
    # The 4 columns are centered at x = 4.70, 6.97, 9.24, 11.51
    #
    # Each column is about 2.13in wide
    # Available height: from 1.34in to 3.25in = 1.91in
    # Labels take bottom ~0.60in, so screenshots go from 1.36in to ~2.55in = 1.19in

    screenshots = [
        (os.path.join(SS_DIR, "01_ocean_map.png"),           4.70, "Ocean Map"),
        (os.path.join(SS_DIR, "02_subsurface_profile.png"),  6.97, "Subsurface Profile"),
        (os.path.join(SS_DIR, "03_cyclone_intelligence.png"),9.24, "Cyclone Intelligence"),
        (os.path.join(SS_DIR, "04_copilot.png"),            11.51, "Ocean Copilot"),
    ]

    # Place screenshots filling the full rectangle height
    # Rectangle: y=1.34in to y=3.25in = 1.91in tall
    # Use full height minus small margins
    ss_top = Inches(1.35)
    ss_height = Inches(1.88)  # Full rectangle height minus margins
    col_width = Inches(2.10)

    for path, label_x, label_name in screenshots:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue

        img = Image.open(path)
        w, h = img.size
        print(f"  {label_name}: source {w}x{h}px")

        # Center in column
        img_left = Inches(label_x - 1.05)
        target_w = Emu(int(col_width))
        target_h = Emu(int(ss_height))

        # Crop to fill the target area
        target_ratio = col_width / ss_height
        img_ratio = w / h

        if img_ratio > target_ratio:
            new_w = int(h * target_ratio)
            left_crop = (w - new_w) // 2
            img = img.crop((left_crop, 0, left_crop + new_w, h))
        else:
            new_h = int(w / target_ratio)
            top_crop = (h - new_h) // 2
            img = img.crop((0, top_crop, w, top_crop + new_h))

        buf = bio.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        pic = slide3.shapes.add_picture(buf, img_left, ss_top, width=target_w, height=target_h)
        print(f"  -> Inserted at ({img_left/914400:.2f}in, {ss_top/914400:.2f}in) size {target_w/914400:.2f}x{target_h/914400:.2f}in")

    prs.save(PPTX_OUT)
    print(f"\nSaved: {PPTX_OUT}")

    # Verify
    prs2 = Presentation(PPTX_OUT)
    print(f"\nVerification:")
    print(f"  Total slides: {len(prs2.slides)}")
    for i, slide in enumerate(prs2.slides):
        title = slide.shapes.title.text if slide.shapes.title else "No title"
        imgs = sum(1 for s in slide.shapes if s.shape_type == 13)
        marker = " <-- MODIFIED" if i == 2 else ""
        print(f"  Slide {i+1}: '{title[:50]}' - {imgs} images{marker}")


if __name__ == "__main__":
    print("Inserting screenshots into Slide 3...")
    insert_screenshots()
    print("\nDone!")
