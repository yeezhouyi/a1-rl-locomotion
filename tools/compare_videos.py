"""Compose synchronized evaluation clips, marking ended episodes explicitly."""
import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", action="append", required=True, metavar="LABEL=FILE")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    labels, readers = [], []
    try:
        for spec in args.clip:
            label, filename = spec.split("=", 1)
            labels.append(label)
            readers.append(iio.get_reader(filename))
        fps = [r.get_meta_data()["fps"] for r in readers]
        sizes = [r.get_meta_data()["size"] for r in readers]
        if len(set(fps)) != 1 or len(set(sizes)) != 1:
            raise ValueError("All clips must have the same FPS and size")
        width, height = sizes[0]
        ended = [False] * len(readers)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with iio.get_writer(args.out, fps=fps[0], macro_block_size=1) as writer:
            index = 0
            while True:
                frames = []
                for j, reader in enumerate(readers):
                    if not ended[j]:
                        try:
                            frame = reader.get_next_data()
                        except (IndexError, StopIteration):
                            ended[j] = True
                    if ended[j]:
                        frame = np.zeros((height, width, 3), dtype=np.uint8)
                    panel = Image.fromarray(frame)
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle((0, 0, width, 28), fill=(20, 20, 20))
                    draw.text((10, 8), f"{labels[j]}   t={index / fps[0]:.2f}s", fill="white")
                    if ended[j]:
                        draw.text((width // 3, height // 2), "EPISODE ENDED", fill="white")
                    frames.append(np.asarray(panel))
                if all(ended):
                    break
                writer.append_data(np.concatenate(frames, axis=1))
                index += 1
    finally:
        for reader in readers:
            reader.close()


if __name__ == "__main__":
    main()
