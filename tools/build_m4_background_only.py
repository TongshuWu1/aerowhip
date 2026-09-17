"""Original video subjects; isolate background for optional masked cleanup."""
from pathlib import Path
import argparse
import json
import hashlib
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/paper/m4_video_sequence_reselected'
SOURCE = Path('C:/Users/wts28/Downloads/IMG_2594 (3).MOV')
INDICES = (504, 546, 570, 602, 620, 626)
CROP = (1100, 80, 2850, 2040)
PATCH = (2050, 0, 2850, 800)

def strip(images, rows, name):
    panel, gap, footer = 1200, 20, 125
    ph = round(panel * 1960 / 1750)
    canvas = Image.new('RGB', (panel * 6 + gap * 5, ph + footer), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/times.ttf', 84)
    for i, (im, row) in enumerate(zip(images, rows)):
        x = i * (panel + gap)
        canvas.paste(im.crop(CROP).resize((panel, ph), Image.Resampling.LANCZOS), (x, 0))
        draw.text((x + panel / 2, ph + footer / 2), row['label'], font=font, fill='black', anchor='mm')
    canvas.save(OUT / (name + '.png'), dpi=(600, 600))
    canvas.thumbnail((2400, 600))
    canvas.save(OUT / (name + '_preview.png'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--background')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(SOURCE))
    frames, rows = [], []
    for idx in INDICES:
        target = OUT / f'frame_{idx:04d}_original.png'
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        assert ok and round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == idx + 1
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        im.save(target)
        frames.append(im)
        rows.append(dict(frame_zero_based=idx, source_pts_s=pts, original_file=target.name))
    cap.release()
    for row in rows:
        row['relative_time_s'] = row['source_pts_s'] - rows[0]['source_pts_s']
        row['label'] = f"t = {row['relative_time_s']:.2f} s"
    strip(frames, rows, 'm4_reselected_original')
    frames[0].crop(PATCH).save(OUT / 'background_only_edit_target.png')
    manifest = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    crop_xyxy=CROP, patch_xyxy=PATCH, frames=rows,
                    time_definition='Video PTS relative to frame 504, not command start.',
                    selection='Frame 570 replaces frame 574 for a clearer, unobstructed drone silhouette; all six panels decoded afresh from the original video.')
    if args.background:
        bg = np.array(Image.open(args.background).convert('RGB').resize((800, 800), Image.Resampling.LANCZOS))
        # Use wall texture only below row 600; generated fabric must never
        # appear above the real, frame-specific sheet edge.
        bg[600:] = bg[599:600]
        cleaned, checks = [], []
        for im, row in zip(frames, rows):
            original = np.asarray(im)
            local = original[0:800, 2050:2850]
            # Only the net and its immediate background. The entire subject
            # area (x < 2200 or y >= 735) is protected, not merely traced edges.
            alpha = np.zeros((800, 800), np.float32)
            gray = cv2.cvtColor(local, cv2.COLOR_RGB2GRAY)
            for x in range(150, 600):
                # The white sheet boundary changes slightly between frames.
                # Find its bright leading edge without copying generated fabric.
                column = np.mean(gray[560:780, max(0, x-2):x+3], axis=1)
                starts = np.flatnonzero(np.convolve((column > 210).astype(int), np.ones(8, int), mode='valid') == 8)
                assert len(starts), (row['frame_zero_based'], x)
                bottom = int(starts[0] + 560)
                assert 600 < bottom < 735, (row['frame_zero_based'], x, bottom)
                side = min(1., (x - 150) / 16., (599 - x) / 16.)
                alpha[:bottom, x] = side
                # Transition only above the measured fabric boundary.
                for y in range(max(0, bottom-3), bottom):
                    alpha[y, x] *= (bottom-y) / 3.
            full_alpha = np.zeros(original.shape[:2], np.float32)
            full_alpha[:800, 2050:2850] = alpha
            assert not np.any(full_alpha[:, :2200])
            assert not np.any(full_alpha[735:])
            # Match the wall and sloping beam to their real boundary colors.
            # Broad low-frequency generated lighting is discarded: it causes
            # artificial seams. Only faint generated texture is retained.
            yy, xx = np.mgrid[:800, :800].astype(np.float32)
            slope = 0.045
            left_y = np.minimum(yy - slope * (xx-125), 585).clip(0, 799)
            right_y = np.minimum(yy + slope * (625-xx), 585).clip(0, 799)
            left = cv2.remap(local, np.full_like(xx, 125), left_y, cv2.INTER_LINEAR)
            right = cv2.remap(local, np.full_like(xx, 625), right_y, cv2.INTER_LINEAR)
            weight = np.clip((xx-125)/500, 0, 1)[..., None]
            lighting = left * (1-weight) + right * weight
            texture = bg.astype(np.float32) - cv2.GaussianBlur(bg.astype(np.float32), (0, 0), 12)
            # Retain texture only in the wall, not generated beam edges.
            texture[:300] = 0
            blended = np.clip(lighting + 0.12 * texture, 0, 255)
            result = original.copy()
            result[:800, 2050:2850] = np.rint(local * (1-alpha[..., None]) + blended * alpha[..., None]).astype(np.uint8)
            changed = np.any(result != original, axis=2)
            assert not np.any(changed & (full_alpha == 0))
            assert np.array_equal(result[:, :2200], original[:, :2200])
            assert np.array_equal(result[735:], original[735:])
            output = Image.fromarray(result)
            output.save(OUT / f"frame_{row['frame_zero_based']:04d}_background_cleaned.png")
            Image.fromarray(np.rint(full_alpha * 255).astype(np.uint8)).save(OUT / f"frame_{row['frame_zero_based']:04d}_background_mask.png")
            cleaned.append(output)
            checks.append(dict(frame=row['frame_zero_based'], changed_pixels=int(changed.sum()),
                               changed_pixels_outside_mask=0, protected_subject_region_pixel_identical=True))
        strip(cleaned, rows, 'm4_background_only')
        manifest['cleanup'] = dict(mode='Built-in image tool for isolated background crop only; deterministic masked compositing.',
                                   generated_patch=str(Path(args.background).resolve()),
                                   unchanged='Drone, cable, target, and all pixels outside background mask. No subject sharpening, contrast edits, tracing, or synthesis.',
                                   background_only=True, checks=checks)
    (OUT / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    main()
