"""Non-generative time composite from the photographic Figure 6 frames."""
from pathlib import Path
import json
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'output/paper/m4_video_sequence_reselected'
OUT = ROOT / 'output/paper/fig1_photographic'
IDS = (504, 546, 570, 620)
OUT.mkdir(parents=True, exist_ok=True)

def inspect():
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 24)
    for idx in IDS:
        im = Image.open(SRC / f'frame_{idx:04d}_no_exit.png').convert('RGB')
        draw = ImageDraw.Draw(im)
        for x in range(1100, 2901, 100):
            draw.line((x,0,x,2100), fill=(170,100,100), width=1)
            draw.text((x+2,80), str(x), font=font, fill='red')
        for y in range(100, 2101, 100):
            draw.line((1100,y,2900,y), fill=(170,100,100), width=1)
            draw.text((1100,y), str(y), font=font, fill='red')
        im.crop((1100,0,2900,2100)).resize((1080,1260)).save(OUT / f'grid_{idx}.png')

def build(response=False):
    destination = ROOT / 'output/paper/fig1_target_response' if response else OUT
    destination.mkdir(parents=True, exist_ok=True)
    # Same integer camera translations estimated from the stationary EXIT sign
    # in the Figure 6 cleanup workflow. No nonrigid warping of the apparatus.
    shifts = {504:(0,0), 546:(0,14), 570:(3,24), 602:(3,33), 620:(4,40)}
    paths = {
        546:[(2390,830),(2405,880),(2440,970),(2478,1050),(2510,1140),(2543,1240),(2578,1340),(2595,1430),(2605,1510),(2600,1610),(2588,1730)],
        570:[(2000,605),(2015,650),(2060,735),(2110,810),(2160,875),(2220,955),(2280,1020),(2345,1080),(2420,1138),(2500,1190),(2575,1230),(2600,1255)],
        602:[(1560,390),(1545,445),(1522,510),(1509,570),(1515,640),(1535,705),(1570,770),(1610,813),(1660,857),(1740,899),(1810,920),(1900,930),(1980,929),(2050,919)],
        620:[(1630,310),(1590,350),(1530,408),(1470,478),(1410,548),(1360,630),(1325,730),(1310,815),(1315,860),(1325,905),(1350,960),(1400,1008)]}
    drones = {
        546:[(2234,809),(2429,722),(2480,754),(2448,805),(2420,872),(2365,887),(2264,862)],
        570:[(1870,520),(2088,520),(2080,605),(2025,649),(1940,630),(1875,575)],
        602:[(1485,255),(1550,246),(1630,345),(1685,397),(1677,450),(1575,429),(1510,400),(1480,324)],
        620:[(1590,164),(1650,155),(1725,260),(1767,324),(1755,383),(1685,355),(1590,308),(1573,245)]}
    images = {i: np.array(Image.open(SRC / f'frame_{i:04d}_no_exit.png').convert('RGB')) for i in IDS}
    result = images[504].copy()
    checks = []
    for idx in IDS[1:]:
        mask = np.zeros(result.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.array(drones[idx], np.int32)], 255)
        cv2.polylines(mask, [np.array(paths[idx], np.int32)], False, 255, 40, cv2.LINE_AA)
        # Broad masks select existing photo pixels; paths are never drawn on
        # the final image. Core pixels stay opaque; feather only outside them.
        core = mask > 0
        distance = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
        alpha = np.clip(1-distance/12, 0, 1)
        dx, dy = shifts[idx]
        transform = np.float32([[1,0,-dx],[0,1,-dy]])
        shifted = cv2.warpAffine(images[idx], transform, (3840,2160), flags=cv2.INTER_NEAREST)
        alpha = cv2.warpAffine(alpha, transform, (3840,2160), flags=cv2.INTER_NEAREST)
        transformed_core = cv2.warpAffine(core.astype(np.uint8), transform, (3840,2160), flags=cv2.INTER_NEAREST) > 0
        before = result.copy()
        result = np.rint(result*(1-alpha[...,None]) + shifted*alpha[...,None]).astype(np.uint8)
        assert np.array_equal(result[transformed_core], shifted[transformed_core])
        assert np.array_equal(result[alpha == 0], before[alpha == 0])
        Image.fromarray(np.ceil(alpha*255).astype(np.uint8)).save(destination / f'mask_{idx}.png')
        checks.append(dict(frame=idx, translation_xy=[-dx,-dy], opaque_core_exactly_from_source=True))
    # Restore the initial hanging cable where later time layers cross it.
    # This is a source-pixel layer, not a newly drawn cable.
    initial_mask = np.zeros(result.shape[:2], np.uint8)
    cv2.polylines(initial_mask, [np.array([(2597,988),(2594,1140),(2591,1330),(2591,1525),(2590,1710),(2592,1915)])], False, 255, 22)
    result[initial_mask > 0] = images[504][initial_mask > 0]
    photo = Image.fromarray(result)
    # Render annotations separately, without redrawing any physical subject.
    draw = ImageDraw.Draw(photo)
    font = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 66)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 58)
    color = (44,78,98)
    for label, center in enumerate([(2790,953),(2505,765),(2118,527),(1805,196)], 1):
        x,y = center
        draw.ellipse((x-49,y-49,x+49,y+49), fill='white', outline=color, width=4)
        draw.text((x,y-2), str(label), fill=color, font=font, anchor='mm')
    # Real target insets, enlarged only; no generated hardware.
    if response:
        target_box = (1160,1010,1470,1230)
        response_font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 58)
        draw.rectangle((730,1310,1840,1875), fill='white')
        draw.line([(1000,1310),(1280,1178)], fill=color, width=4)
        draw.text((1270,1360), 'Target hit', font=small, fill=color, anchor='mm')
        for index, idx, x, label in [(4,620,740,'Before: 0.97 s'), (5,632,1310,'After: 1.07 s')]:
            original = Image.open(destination / f'frame_{idx:04d}_original.png').convert('RGB')
            inset = original.crop(target_box).resize((520,369), Image.Resampling.LANCZOS)
            photo.paste(inset, (x,1490))
            draw = ImageDraw.Draw(photo)
            draw.rectangle((x-3,1487,x+523,1862), outline=color, width=5)
            draw.rectangle((x-3,1400,x+523,1486), fill='white')
            draw.text((x+260,1443), label, font=response_font, fill=color, anchor='mm')
            draw.ellipse((x+12,1502,x+91,1581), fill='white', outline=color, width=3)
            draw.text((x+51,1540), str(index), fill=color, font=small, anchor='mm')
    else:
        target_box = (1190,985,1440,1195)
        inset = Image.fromarray(images[504]).crop(target_box).resize((500,420), Image.Resampling.LANCZOS)
        photo.paste(inset, (740,1455))
        draw = ImageDraw.Draw(photo)
        draw.rectangle((738,1453,1242,1877), outline=color, width=5)
        draw.text((990,1405), 'Target', font=small, fill=color, anchor='mm')
        draw.line([(1160,1453),(1275,1175)], fill=color, width=4)
    final = photo.crop((650,70,3090,2050))
    final.save(destination / 'fig1_photographic_composite.png', dpi=(600,600))
    final.thumbnail((1400,1400))
    final.save(destination / 'fig1_photographic_preview.png')
    metadata = dict(source_directory=str(SRC), frames=list(IDS),
                    times_s=[0,.35,.55,.9670833333],
                    processing='Non-generative masked photo compositing with integer camera translations, numbered labels, and an enlarged original target crop. Figure 6 background cleanup is inherited. No subject synthesis, sharpening, recoloring, or nonrigid warping.',
                    limitation='Illustrative multi-time montage, not a single exposure or quantitative trajectory. Later layers can occlude earlier layers at crossings.', checks=checks)
    if response:
        metadata['target_response_insets'] = dict(frames=[620,632], times_s=[.967083333333334,1.067083333333334], crop_xyxy=target_box,
                                                  source='Unretouched video frames; same crop and enlargement for both insets.',
                                                  observation='Target upright before cable passage; rotated nearly horizontal afterward. No exact contact-time estimate or quantitative displacement measurement is inferred.')
    (destination / 'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n', encoding='utf-8')
    print(destination / 'fig1_photographic_composite.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--response', action='store_true')
    args = parser.parse_args()
    inspect() if args.inspect else build(response=args.response)
