#!/usr/bin/env python3

"""

Create EndoVis-18-VQLA train/val/test splits following the Surgery-R1 paper.

Assumptions:

- You have an annotations file like `qa_pairs.json` from the official

  EndoVis-18-VQLA (Surgical-VQLA) repo.

- Each QA entry has at least: "image_name" (or "img_name"/"image_path").

What this script does:

- Parses all QA pairs.

- Extracts sequence ID from image_name (expects things like "seq_10/xxx.png").

- Uses sequences 1, 5, 16 as TEST (as in Surgery-R1).

- Uses all other sequences to build TRAIN (1,560 frames) and VAL (447 frames),

  by sorting frames and taking them in order.

- Writes three JSON files: train.json, val.json, test.json.

Usage:

    python make_surgery_r1_split.py \

        --qa_json path/to/qa_pairs.json \

        --out_dir splits/

"""

import argparse

import json

import os

import re

from collections import defaultdict, Counter

from typing import Dict, List, Tuple

TARGET_TRAIN_FRAMES = 1560

TARGET_VAL_FRAMES = 447

TEST_SEQ_IDS = {1, 5, 16}

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument("--qa_json", type=str, required=True,

                    help="Path to qa_pairs.json (EndoVis-18-VQLA annotations).")

    ap.add_argument("--out_dir", type=str, required=True,

                    help="Directory to write train.json / val.json / test.json.")

    ap.add_argument("--image_key", type=str, default=None,

                    help="Key name for image path (default: auto-detect among "

                         "['image_name','img_name','image_path']).")

    return ap.parse_args()

def load_qa_pairs(path: str) -> List[dict]:

    with open(path, "r") as f:

        data = json.load(f)

    # Many repos store directly as a list, some as {"data": [...]}

    if isinstance(data, dict) and "data" in data:

        data = data["data"]

    if not isinstance(data, list):

        raise ValueError(f"Expected list (or dict['data']) in {path}, got {type(data)}")

    return data

def detect_image_key(sample: dict, explicit_key: str = None) -> str:

    if explicit_key is not None:

        if explicit_key not in sample:

            raise KeyError(f"Explicit image_key='{explicit_key}' not found in sample keys: {list(sample.keys())}")

        return explicit_key

    for k in ["image_name", "img_name", "image_path", "image"]:

        if k in sample:

            return k

    raise KeyError(f"Could not infer image key; sample keys: {list(sample.keys())}")

_seq_regex = re.compile(r"seq[_\-]?(\d+)", re.IGNORECASE)

def get_seq_id(image_name: str) -> int:

    """

    Extract numeric sequence id from image path.

    Works with paths like:

        'seq_1/frame000.png'

        'data/Seq-10/xxx.png'

    """

    m = _seq_regex.search(image_name)

    if not m:

        raise ValueError(f"Could not extract sequence id from image_name='{image_name}'")

    return int(m.group(1))

def split_by_frames(

    qa_pairs: List[dict],

    image_key: str

) -> Tuple[Dict[str, List[dict]], Dict[str, List[dict]], Dict[str, List[dict]]]:

    """

    Returns (train_frames, val_frames, test_frames) where each is:

        dict[image_name] -> list of QA dicts

    """

    # 1) Group QAs by image_name, and track seq_id for each image

    frame_to_qas: Dict[str, List[dict]] = defaultdict(list)

    frame_to_seq: Dict[str, int] = {}

    for qa in qa_pairs:

        img = qa[image_key]

        frame_to_qas[img].append(qa)

        if img not in frame_to_seq:

            seq_id = get_seq_id(img)

            frame_to_seq[img] = seq_id

    # 2) Partition frames into test vs other, based on seq_id

    test_frames = {}

    other_frames = {}  # candidates for train+val

    for img, qas in frame_to_qas.items():

        seq_id = frame_to_seq[img]

        if seq_id in TEST_SEQ_IDS:

            test_frames[img] = qas

        else:

            other_frames[img] = qas

    # 3) Sort "other" frames deterministically:

    #    first by seq_id, then by path (frame index inside sequence)

    def frame_sort_key(img: str):

        return (frame_to_seq[img], img)

    sorted_other = sorted(other_frames.keys(), key=frame_sort_key)

    # 4) Assign TRAIN and VAL by taking frames in order

    train_frames = {}

    val_frames = {}

    # Assign train frames

    for img in sorted_other:

        if len(train_frames) < TARGET_TRAIN_FRAMES:

            train_frames[img] = other_frames[img]

        else:

            break

    # Assign val frames (from remaining)

    for img in sorted_other[len(train_frames):]:

        if len(val_frames) < TARGET_VAL_FRAMES:

            val_frames[img] = other_frames[img]

        else:

            break

    # Note: any leftover frames in "other" are unused.

    return train_frames, val_frames, test_frames

def frames_to_flat_list(frame_dict: Dict[str, List[dict]]) -> List[dict]:

    # Just concatenate QA lists

    all_qas = []

    for qas in frame_dict.values():

        all_qas.extend(qas)

    return all_qas

def summarize_split(name: str, frame_dict: Dict[str, List[dict]]):

    num_frames = len(frame_dict)

    num_qas = sum(len(qas) for qas in frame_dict.values())

    if num_frames > 0:

        qa_per_img = num_qas / num_frames

    else:

        qa_per_img = 0.0

    print(f"{name} split:")

    print(f"  Frames: {num_frames}")

    print(f"  QA pairs: {num_qas}")

    print(f"  QA per image: {qa_per_img:.3f}")

    # Optional: frequency of sequences

    seq_counter = Counter(get_seq_id(img) for img in frame_dict.keys())

    print(f"  Sequences: {dict(sorted(seq_counter.items()))}")

    print()

def main():

    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    qa_pairs = load_qa_pairs(args.qa_json)

    if len(qa_pairs) == 0:

        raise ValueError("No QA pairs loaded; check qa_json path.")

    image_key = detect_image_key(qa_pairs[0], args.image_key)

    print(f"Using image key: {image_key}")

    train_frames, val_frames, test_frames = split_by_frames(qa_pairs, image_key)

    # Summaries (so you can compare with Surgery-R1 numbers)

    summarize_split("TRAIN", train_frames)

    summarize_split("VAL", val_frames)

    summarize_split("TEST", test_frames)

    # Write outputs

    train_list = frames_to_flat_list(train_frames)

    val_list = frames_to_flat_list(val_frames)

    test_list = frames_to_flat_list(test_frames)

    with open(os.path.join(args.out_dir, "train.json"), "w") as f:

        json.dump(train_list, f, indent=2)

    with open(os.path.join(args.out_dir, "val.json"), "w") as f:

        json.dump(val_list, f, indent=2)

    with open(os.path.join(args.out_dir, "test.json"), "w") as f:

        json.dump(test_list, f, indent=2)

    print("Saved splits to:", args.out_dir)

    print("NOTE: Frame counts should match 1560 / 447 and test=seq{1,5,16}.")

    print("      QA counts may differ slightly from the paper, depending on the "

          "exact annotation version you have.")

if __name__ == "__main__":

    main()












