"""Insert real OceanEmbed screenshots into SIH presentation Slide 3."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.util import Inches, Emu
from PIL import Image
import io as bio

BASE = os.path.dirname(os.path.abspath(__file__))
SS_DIR = os.path.join(BASE, "screenshots")
os.makedirs(SS_DIR, exist_ok=True)

PPTX_SRC = os.path.join(BASE, "SIH2026-OceanEmbed-Final (3).pptx")
PPTX_OUT = os.path.join(BASE, "SIH2026-OceanEmbed-Final-With-Prototype-Screenshots.pptx")

URL = "http://localhost:5001"


def take_screenshots():
    """Take 4 screenshots using Playwright."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        # Screenshot 1: Ocean Map
        print("1. Ocean Map...")
        page.goto(URL)
        page.wait_for_load_state("networkidle")
        page.evaluate("go('observations')")
        page.wait_for_timeout(3000)
        page.evaluate("if(typeof allObs!=='undefined'&&allObs.length>5)selectObs(allObs[5].lat+','+allObs[5].lon)")
        page.wait_for_timeout(2500)
        p1 = os.path.join(SS_DIR, "01_ocean_map.png")
        page.screenshot(path=p1, full_page=False)
        print(f"   Saved: {p1}")

        # Screenshot 2: Subsurface Profile
        print("2. Subsurface Profile...")
        page.evaluate("go('prediction')")
        page.wait_for_timeout(3000)
        p2 = os.path.join(SS_DIR, "02_subsurface_profile.png")
        page.screenshot(path=p2, full_page=False)
        print(f"   Saved: {p2}")

        # Screenshot 3: Cyclone Intelligence (scrolled to show results)
        print("3. Cyclone Intelligence...")
        page.evaluate("go('cyclone')")
        page.wait_for_timeout(2000)
        page.evaluate("loadDemoTrack()")
        page.wait_for_timeout(1000)
        page.evaluate("analyzeCycloneTrack()")
        page.wait_for_timeout(6000)
        page.evaluate("window.scrollTo(0,350)")
        page.wait_for_timeout(500)
        p3 = os.path.join(SS_DIR, "03_cyclone_intelligence.png")
        page.screenshot(path=p3, full_page=False)
        print(f"   Saved: {p3}")

        # Screenshot 4: Copilot (overview + open copilot panel)
        print("4. Copilot...")
        page.evaluate("window.scrollTo(0,0)")
        page.evaluate("go('overview')")
        page.wait_for_timeout(2000)
        page.evaluate("if(copilotOpen)toggleCopilot()")
        page.wait_for_timeout(500)
        page.evaluate("toggleCopilot()")
        page.wait_for_timeout(800)
        p4 = os.path.join(SS_DIR, "04_copilot.png")
        page.screenshot(path=p4, full_page=False)
        print(f"   Saved: {p4}")

        browser.close()
    return [p1, p2, p3, p4]


def insert_screenshots(screenshot_paths):
    """Insert screenshots into Slide 3 at the correct positions."""
    prs = Presentation(PPTX_SRC)
    slide3 = prs.slides[2]

    # From inspection, the 4 screenshot label positions are:
    # Label 0 (Ocean Map):          x=4.70in, y=1.94in
    # Label 1 (Subsurface Profile): x=6.97in, y=1.94in
    # Label 2 (Cyclone Intelligence): x=9.24in, y=1.94in
    # Label 3 (Ocean Copilot):      x=11.51in, y=1.94in
    #
    # The background rectangle (Rounded Rectangle 17410) is at:
    #   x=4.37in, y=1.34in, size=8.87x1.91in
    #
    # Screenshots should go ABOVE the labels, inside this rectangle.
    # Label positions give us the center x of each column.
    # Each column is ~2.13in wide.

    label_positions = [
        (4.70, "Ocean Map"),
        (6.97, "Subsurface Profile"),
        (9.24, "Cyclone Intelligence"),
        (11.51, "Ocean Copilot"),
    ]

    # Screenshot area: above the labels, inside the rounded rectangle
    # Top of rectangle: 1.34in, labels start at ~1.94in
    # Available height for image: from 1.36in to ~1.92in (0.56in)
    # But labels also take space. Let's put images from 1.36 to 1.90 = 0.54in
    # Width per column: ~2.10in

    ss_top = Inches(1.36)
    ss_height = Inches(0.54)
    col_width = Inches(2.10)

    for i, (path, (label_x, label_name)) in enumerate(zip(screenshot_paths, label_positions)):
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue

        # Center the image in the column
        img_left = Inches(label_x - 1.05)  # Center at label_x, width 2.10

        # Open and resize image
        img = Image.open(path)
        w, h = img.size

        # Target dimensions in EMU
        target_w = Emu(int(col_width))
        target_h = Emu(int(ss_height))

        # Resize maintaining aspect ratio to fill
        target_ratio = col_width / ss_height
        img_ratio = w / h

        if img_ratio > target_ratio:
            # Image is wider - crop width or fit height
            new_h = h
            new_w = int(h * target_ratio)
            left_crop = (w - new_w) // 2
            img = img.crop((left_crop, 0, left_crop + new_w, h))
        else:
            # Image is taller - crop or fit width
            new_w = w
            new_h = int(w / target_ratio)
            top_crop = (h - new_h) // 2
            img = img.crop((0, top_crop, w, top_crop + new_h))

        # Save to bytes
        buf = bio.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        # Insert picture
        pic = slide3.shapes.add_picture(buf, img_left, ss_top, width=target_w, height=target_h)
        print(f"  Inserted {label_name}: left={img_left/914400:.2f}in, top={ss_top/914400:.2f}in")

    prs.save(PPTX_OUT)
    print(f"\nSaved: {PPTX_OUT}")
    return PPTX_OUT


def main():
    print("=" * 60)
    print("OceanEmbed SIH Slide 3 - Screenshot Insertion")
    print("=" * 60)

    # Take screenshots
    print("\n--- Taking screenshots ---")
    paths = take_screenshots()

    # Insert into PPTX
    print("\n--- Inserting into presentation ---")
    out = insert_screenshots(paths)

    # Verify
    print("\n--- Verification ---")
    prs = Presentation(out)
    print(f"Total slides: {len(prs.slides)}")
    for i, slide in enumerate(prs.slides):
        title = slide.shapes.title.text if slide.shapes.title else "No title"
        imgs = sum(1 for s in slide.shapes if s.shape_type == 13)
        print(f"  Slide {i+1}: '{title[:50]}' - {imgs} images")

    print("\nDone!")


if __name__ == "__main__":
    main()
