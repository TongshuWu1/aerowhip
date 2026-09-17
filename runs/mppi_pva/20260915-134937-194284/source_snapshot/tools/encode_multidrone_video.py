"""Encode a recorded Isaac frame sequence using the project's bundled FFmpeg."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import imageio_ffmpeg


def encode(folder):
    folder=Path(folder)
    metadata=json.loads((folder/'recording.json').read_text())
    destination=folder/'replay.mp4'
    if destination.exists():raise ValueError('A video already exists in this recording folder.')
    frames=[folder/f'frame_{i:05d}.png' for i in range(metadata['frame_count'])]
    if not frames or not all(p.is_file() for p in frames):raise ValueError('Recording is incomplete.')
    with Image.open(frames[0]) as first:size=first.size
    writer=imageio_ffmpeg.write_frames(str(destination),size,fps=metadata['fps'],codec='libx264',
        pix_fmt_in='rgb24',pix_fmt_out='yuv420p',quality=8,macro_block_size=2,
        output_params=['-metadata','title=Calibrated aerial cable model - deterministic checkpoint replay',
                       '-metadata','comment='+metadata['label']])
    writer.send(None)
    try:
        for frame in frames:
            with Image.open(frame) as im:
                if im.size!=size:raise ValueError('Frame resolution changed during recording.')
                writer.send(np.asarray(im.convert('RGB')).tobytes())
    finally:writer.close()
    print(str(destination.resolve()),flush=True)
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder');encode(p.parse_args().folder)
