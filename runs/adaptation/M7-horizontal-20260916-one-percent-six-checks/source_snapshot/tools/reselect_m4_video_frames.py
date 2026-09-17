"""Inspect authentic video frames; no synthesis or retouching."""
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/paper/m4_video_sequence_reselected'
OUT.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture('C:/Users/wts28/Downloads/IMG_2594 (3).MOV')
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
indices = list(range(558, 591, 2))
sheet = Image.new('RGB', (5 * 450, 4 * 435), 'white')
draw = ImageDraw.Draw(sheet)
for n, idx in enumerate(indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    assert ok
    pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    rgb.save(OUT / f'candidate_{idx:04d}.png')
    view = rgb.crop((1550, 180, 2250, 810))
    view.thumbnail((450, 395))
    x, y = (n % 5) * 450, (n // 5) * 435
    sheet.paste(view, (x, y))
    draw.text((x + 10, y + 398), f'{idx}: t={pts - 4.202083333333333:.3f} s', font=font, fill='black')
sheet.save(OUT / 'drone_candidates.jpg', quality=95)
cap.release()
print(OUT / 'drone_candidates.jpg')
