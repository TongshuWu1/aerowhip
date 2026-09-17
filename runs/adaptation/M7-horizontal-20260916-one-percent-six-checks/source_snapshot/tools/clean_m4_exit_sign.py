"""Remove background signs with a restricted mask; preserve video subjects."""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
from build_m4_background_only import strip, OUT, INDICES

BOX = (1570, 260, 2000, 640)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch')
    args = parser.parse_args()
    frames = [Image.open(OUT / f'frame_{i:04d}_background_cleaned.png').convert('RGB') for i in INDICES]
    sheet = Image.new('RGB', (430*3, 420*2), 'white')
    draw = ImageDraw.Draw(sheet)
    for n, (idx, im) in enumerate(zip(INDICES, frames)):
        crop = im.crop(BOX)
        crop.save(OUT / f'exit_sign_{idx:04d}_context.png')
        x, y = n % 3 * 430, n // 3 * 420
        sheet.paste(crop, (x, y))
        draw.text((x+10, y+385), str(idx), fill='black')
    sheet.save(OUT / 'exit_sign_contexts.png')
    if args.patch:
        patch = np.asarray(Image.open(args.patch).convert('RGB').resize((430, 380), Image.Resampling.LANCZOS)).astype(np.float32)
        rows = json.loads((OUT / 'provenance.json').read_text())['frames']
        reference = np.asarray(frames[0].crop(BOX))
        def red_center(local):
            r, g, b = [local[:, :, c].astype(float) for c in range(3)]
            yy, xx = np.where((r > 1.4*g) & (r > 1.4*b) & (r > 110))
            return np.median(xx), np.median(yy)
        cx, cy = red_center(reference)
        results, checks = [], []
        for idx, im in zip(INDICES, frames):
            original = np.asarray(im)
            local = np.asarray(im.crop(BOX))
            nx, ny = red_center(local)
            dx, dy = int(round(nx-cx)), int(round(ny-cy))
            yy, xx = np.mgrid[:380, :430]
            # Remove housing, guard, and shadow. Feather into real wall.
            alpha = np.minimum.reduce([(xx-(25+dx))/12., ((350+dx)-xx)/12.,
                                       (yy-(10+dy))/12., ((307+dy)-yy)/8.,
                                       np.ones_like(xx)]).clip(0, 1).astype(np.float32)
            protected = np.zeros((380, 430), np.uint8)
            # Keep original clamp and subjects, including blur and markers.
            cv2.rectangle(protected, (173+dx, 281+dy), (218+dx, 379), 255, -1)
            if idx == 570:
                cv2.rectangle(protected, (309, 270), (429, 379), 255, -1)
            elif idx == 602:
                cv2.fillPoly(protected, [np.array([(0,0),(22,17),(62,74),(113,148),(114,179),(83,189),(48,169),(0,182)])], 255)
            elif idx == 620:
                cv2.fillPoly(protected, [np.array([(0,0),(115,0),(155,48),(183,93),(180,117),(160,128),(115,113),(66,106),(24,73),(0,38)])], 255)
                cv2.line(protected, (83,44), (0,120), 255, 28)
            elif idx == 626:
                cv2.fillPoly(protected, [np.array([(78,0),(208,0),(241,48),(248,76),(230,104),(181,103),(131,92),(95,63),(75,27)])], 255)
                cv2.line(protected, (132,29), (0,176), 255, 28)
            alpha[protected > 0] = 0
            shifted = cv2.warpAffine(patch, np.float32([[1,0,dx],[0,1,dy]]), (430,380), borderMode=cv2.BORDER_REPLICATE)
            # Match exposure using a clear wall-only sample, avoiding drone
            # or cable pixels in the lower/right or upper/left crop edges.
            correction = np.median(local[120:230, 365:400].astype(float) - shifted[120:230, 365:400], axis=(0,1))
            corrected = np.clip(shifted + correction, 0, 255)
            updated = np.rint(local*(1-alpha[..., None]) + corrected*alpha[..., None]).astype(np.uint8)
            result = original.copy()
            result[BOX[1]:BOX[3], BOX[0]:BOX[2]] = updated
            mask = np.zeros(original.shape[:2], np.uint8)
            mask[BOX[1]:BOX[3], BOX[0]:BOX[2]] = np.ceil(alpha*255).astype(np.uint8)
            changed = np.any(result != original, axis=2)
            assert not np.any(changed & (mask == 0))
            assert np.array_equal(updated[protected > 0], local[protected > 0])
            output = Image.fromarray(result)
            output.save(OUT / f'frame_{idx:04d}_no_exit.png')
            Image.fromarray(mask).save(OUT / f'frame_{idx:04d}_exit_mask.png')
            results.append(output)
            checks.append(dict(frame=idx, shift=[dx,dy], changed_pixels=int(changed.sum()),
                               changed_pixels_outside_mask=0, protected_pixels_identical=True))
        strip(results, rows, 'm4_background_no_exit')
        (OUT / 'exit_removal_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
        print(json.dumps(checks, indent=2))

if __name__ == '__main__':
    main()
