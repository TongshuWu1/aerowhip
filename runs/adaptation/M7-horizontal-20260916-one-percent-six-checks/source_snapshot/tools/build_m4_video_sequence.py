"""Extract and arrange authentic video frames; do not retouch their contents."""
from pathlib import Path
import hashlib
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('C:/Users/wts28/Downloads/IMG_2594 (3).MOV')
OUT = ROOT/'output/paper/m4_video_sequence'
PDF = ROOT/'output/pdf/m4_video_sequence'
OUT.mkdir(parents=True, exist_ok=True)
PDF.mkdir(parents=True, exist_ok=True)
# Zero-based source-frame indices, selected by visual inspection of the video.
INDICES = (504, 546, 574, 602, 620, 626)
CROP = (1100, 80, 2850, 2040)
cap = cv2.VideoCapture(str(SOURCE))
assert cap.isOpened()
fps = cap.get(cv2.CAP_PROP_FPS)
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames, rows = [], []
for i, index in enumerate(INDICES):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    assert ok and round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == index+1
    pts = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    assert image.size == (3840,2160)
    raw = OUT/f'frame_{index:04d}_original.png'
    image.save(raw)
    cropped = image.crop(CROP)
    crop_path = OUT/f'frame_{index:04d}_crop.png'
    cropped.save(crop_path)
    frames.append(cropped)
    rows.append(dict(panel=i+1, source_frame_zero_based=index, source_pts_s=pts,
                     original_file=raw.name, crop_file=crop_path.name))
cap.release()
zero = rows[0]['source_pts_s']
for row in rows:
    row['relative_time_s'] = row['source_pts_s']-zero
    row['label'] = f"t = {row['relative_time_s']:.2f} s"

# Equal fixed crops and equal scale; no background cleaning, recoloring,
# contrast adjustment, cable tracing, compositing inside a frame, or synthesis.
panel, gutter, footer = 1200, 20, 125
panel_height = round(panel*(CROP[3]-CROP[1])/(CROP[2]-CROP[0]))
width = panel*len(frames)+gutter*(len(frames)-1)
strip = Image.new('RGB', (width,panel_height+footer), 'white')
draw = ImageDraw.Draw(strip)
font = ImageFont.truetype('C:/Windows/Fonts/times.ttf', 84)
for i, (frame, row) in enumerate(zip(frames,rows)):
    x = i*(panel+gutter)
    strip.paste(frame.resize((panel,panel_height), Image.Resampling.LANCZOS), (x,0))
    draw.text((x+panel/2,panel_height+footer/2), row['label'], fill='black', font=font, anchor='mm')
strip.save(OUT/'m4_whip_sequence.png', dpi=(600,600))
preview = strip.copy()
preview.thumbnail((2400,600))
preview.save(OUT/'m4_whip_sequence_preview.png')

# Preserve vector labels in the paper PDF; embedded images are original crops.
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
                     'pdf.fonttype':42, 'mathtext.fontset':'stix'})
figure_width = 7.16
figure_height = figure_width*(panel_height+footer)/width
fig = plt.figure(figsize=(figure_width,figure_height), facecolor='white')
for i, (frame,row) in enumerate(zip(frames,rows)):
    left = i*(panel+gutter)/width
    ax = fig.add_axes([left,footer/(panel_height+footer),panel/width,panel_height/(panel_height+footer)])
    ax.imshow(frame, interpolation='none')
    ax.set_axis_off()
    fig.text(left+panel/(2*width), footer/(2*(panel_height+footer)),
             f"$t = {row['relative_time_s']:.2f}$ s", ha='center', va='center', fontsize=8.5)
fig.savefig(PDF/'m4_whip_sequence.pdf', dpi=600, pad_inches=0)
plt.close(fig)

manifest = dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    source_frame_count=count,source_average_fps=fps,source_dimensions_px=[3840,2160],
    crop_xyxy=CROP,frames=rows,
    time_definition='Video presentation timestamps relative to frame 504 (a selected pre-motion frame), not synchronized to the flight command, model strike time, or flight logs.',
    image_processing='Decoded source frames, identical fixed crop, resizing for the PNG strip, white gutters and external time labels only; no content retouching, synthesis or enhancement.',
    interpretation='M4 identity supplied by the author. Last frames show the target interaction and visible target response; precise 3D contact cannot be inferred from a single view alone.',
    output_png='m4_whip_sequence.png',output_pdf=str(PDF/'m4_whip_sequence.pdf'))
(OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
for row in rows:
    print(row['source_frame_zero_based'],f"source {row['source_pts_s']:.6f} s",row['label'])
print('PNG:',OUT/'m4_whip_sequence.png')
print('PDF:',PDF/'m4_whip_sequence.pdf')
