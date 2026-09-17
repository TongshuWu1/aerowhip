"""Extract unretouched frames around the visible target response."""
from pathlib import Path
import json
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/paper/fig1_target_response'
OUT.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture('C:/Users/wts28/Downloads/IMG_2594 (3).MOV')
cap.set(cv2.CAP_PROP_POS_FRAMES, 614)
sheet = Image.new('RGB', (5*480, 3*470), 'white')
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 27)
rows = []
for idx in range(614, 643):
    ok, bgr = cap.read()
    assert ok
    if idx % 2:
        continue
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    pts = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
    im.save(OUT / f'frame_{idx:04d}_original.png')
    crop = im.crop((1120,870,1550,1260)).resize((480,435), Image.Resampling.LANCZOS)
    n = len(rows)
    x,y = n%5*480,n//5*470
    sheet.paste(crop,(x,y))
    draw.text((x+12,y+438), f'{idx}: {pts-4.202083333333333:.3f} s', fill='black', font=font)
    rows.append(dict(frame=idx, pts_s=pts, relative_time_s=pts-4.202083333333333))
cap.release()
sheet.save(OUT/'target_response_contacts.png')
(OUT/'frame_times.json').write_text(json.dumps(rows,indent=2)+'\n')
print(OUT/'target_response_contacts.png')
