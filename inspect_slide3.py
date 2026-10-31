"""Inspect Slide 3 text boxes to find screenshot placeholders."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pptx import Presentation
import os

PPTX_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SIH2026-OceanEmbed-Final (3).pptx")
prs = Presentation(PPTX_SRC)
slide3 = prs.slides[2]

print(f"Slide 3 title: {slide3.shapes.title.text}")
print(f"Slide width: {prs.slide_width/914400:.2f} in")
print(f"Slide height: {prs.slide_height/914400:.2f} in")
print()

for shape in slide3.shapes:
    stype = str(shape.shape_type)
    name = shape.name
    left = shape.left / 914400
    top = shape.top / 914400
    w = shape.width / 914400
    h = shape.height / 914400
    
    text = ""
    if hasattr(shape, "text_frame"):
        try:
            text = shape.text_frame.text.strip().replace("\n", " | ")[:80]
        except:
            pass
    
    print(f"  [{stype:30s}] {name:30s} ({left:6.2f}, {top:6.2f}) {w:5.2f}x{h:5.2f}  '{text}'")
