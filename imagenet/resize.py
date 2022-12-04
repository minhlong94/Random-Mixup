from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing

cpus = multiprocessing.cpu_count()
cpus = min(48, cpus)
import os
import argparse

szs = (160, 352)


def resize_img(p, im, fn, sz):
    w, h = im.size
    ratio = min(h / sz, w / sz)
    im = im.resize((int(w / ratio), int(h / ratio)), resample=Image.BICUBIC)
    new_fn = DEST / str(sz) / fn.relative_to(PATH)
    new_fn.parent.mkdir(exist_ok=True)
    im.convert("RGB").save(new_fn)


def resizes(p, fn):
    im = Image.open(fn)
    for sz in szs:
        resize_img(p, im, fn, sz)


def resize_imgs(p):
    files = p.glob("*/*.JPEG")
    with ProcessPoolExecutor(cpus) as e:
        e.map(partial(resizes, p), files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize ImageNet to 160 and 352")
    parser.add_argument(
        "--path", type=str, help="ImageNet path, which has train and val folders"
    )
    parser.add_argument("--dest", type=str, help="Destination for resized images.")
    args = parser.parse_args()
    PATH = Path(args.path)
    if not os.path.isdir(args.dest):
        os.mkdir(args.dest)
    DEST = Path(args.dest)

    for sz in szs:
        ssz = str(sz)
        (DEST / ssz).mkdir(exist_ok=True)
        for ds in ("val", "train"):
            (DEST / ssz / ds).mkdir(exist_ok=True)

    for ds in ("val", "train"):
        resize_imgs(PATH / ds)
