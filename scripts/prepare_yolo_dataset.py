#!/usr/bin/env python3
# 功能：整理 YOLO 训练数据集和标签文件。

"""Prepare a YOLO dataset from LabelImg output.

The original LabelImg labels are left untouched. This script can remap class
ids, drop images without labels, validate YOLO boxes, and split train/val.
"""

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_class_file(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_mapping(text):
    mapping = {}
    if not text:
        return mapping
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Bad mapping item: {item!r}. Expected old:new.")
        old, new = item.split(":", 1)
        mapping[int(old.strip())] = int(new.strip())
    return mapping


def validate_and_remap_label(src, dst, class_map):
    out_lines = []
    for lineno, raw in enumerate(src.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{src.name}:{lineno}: expected 5 columns, got {len(parts)}")
        old_cls = int(float(parts[0]))
        if old_cls not in class_map:
            raise ValueError(f"{src.name}:{lineno}: class id {old_cls} has no remap")
        vals = [float(v) for v in parts[1:]]
        if vals[2] <= 0.0 or vals[3] <= 0.0 or any(v < 0.0 or v > 1.0 for v in vals):
            raise ValueError(f"{src.name}:{lineno}: box values out of YOLO range")
        out_lines.append(
            "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(class_map[old_cls], *vals)
        )
    if not out_lines:
        raise ValueError(f"{src.name}: empty label file")
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return len(out_lines)


def write_data_yaml(path, dataset_root, class_names):
    names = ", ".join(f"'{name}'" for name in class_names)
    text = (
        f"path: {dataset_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: [{names}]\n"
    )
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-classes", required=True, type=Path)
    parser.add_argument("--target-classes", default="tiger,peacock,monkey,elephant,wolf")
    parser.add_argument("--map", dest="class_map", default="")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_classes = parse_class_file(args.source_classes)
    target_classes = [x.strip() for x in args.target_classes.split(",") if x.strip()]
    if args.class_map:
        class_map = parse_mapping(args.class_map)
    else:
        class_map = {}
        for old_id, name in enumerate(source_classes):
            if name not in target_classes:
                raise ValueError(f"Source class {name!r} not in target classes")
            class_map[old_id] = target_classes.index(name)

    images = {
        p.stem: p
        for p in args.images.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }
    labels = {
        p.stem: p
        for p in args.labels.glob("*.txt")
        if p.name not in {"classes.txt", "predefined_classes.txt"}
    }
    pairs = sorted((stem, images[stem], labels[stem]) for stem in images.keys() & labels.keys())
    missing_labels = sorted(images.keys() - labels.keys())
    extra_labels = sorted(labels.keys() - images.keys())
    if not pairs:
        raise RuntimeError("No image/label pairs found")

    if args.out.exists():
        if not args.force:
            raise RuntimeError(f"Output exists: {args.out}. Use --force to replace it.")
        shutil.rmtree(args.out)

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    val_count = max(1, round(len(pairs) * args.val_ratio))
    val_stems = {stem for stem, _, _ in pairs[:val_count]}

    counts = {"train": 0, "val": 0}
    object_count = 0
    for stem, image_path, label_path in pairs:
        split = "val" if stem in val_stems else "train"
        shutil.copy2(image_path, args.out / "images" / split / image_path.name)
        object_count += validate_and_remap_label(
            label_path,
            args.out / "labels" / split / f"{stem}.txt",
            class_map,
        )
        counts[split] += 1

    (args.out / "classes.txt").write_text("\n".join(target_classes) + "\n", encoding="utf-8")
    write_data_yaml(args.out / "data.yaml", args.out.resolve(), target_classes)

    print(f"source_classes={source_classes}")
    print(f"target_classes={target_classes}")
    print(f"class_map={class_map}")
    print(f"pairs={len(pairs)} train={counts['train']} val={counts['val']} objects={object_count}")
    print(f"missing_labels={len(missing_labels)}")
    for stem in missing_labels[:20]:
        print(f"  missing_label {stem}")
    print(f"extra_labels={len(extra_labels)}")
    for stem in extra_labels[:20]:
        print(f"  extra_label {stem}")
    print(f"out={args.out.resolve()}")


if __name__ == "__main__":
    main()
