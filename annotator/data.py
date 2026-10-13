"""Data side of the annotation tool.

Left ("human"): the dataset (train/ and test/). Instances come from each
experiment's COCO JSON -- every molten, droplet, arc, pool and spatter is its
own instance, even of the same class. Category names are spelled many ways
across folders ("weilding pool", " External arc", "splatter", ...) and are
mapped to the seven classes. Where the JSON has no annotations for a frame,
instances are the connected pieces of each class in the human RGB mask.

JSON instances overlap; on screen and in the saved RGB mask every pixel has
one owner, decided by class priority:
    molten_consumable = droplet = spatter  >  internal_arc  >  external_arc = weld_pool
Where two instances of the same priority overlap, it is a *conflict*: by
default the owner is the one whose class the human RGB mask shows there, and
the user can switch it in the app.

Right ("result"): the folder being annotated (e.g. final/spatter_refined),
matched by mask file name. Its instances are the connected pieces of each
class, each named after the human instance of the same class it overlaps
most. If the image was saved before, the saved edit is loaded instead.

Saving writes, under the output folder, <split>/<experiment>/<mask name>
(the RGB mask, same size) and <split>/<experiment>/instances_coco.json
(one non-overlapping instance per annotation, uncompressed RLE, with the
instance name).
"""

from __future__ import annotations

import base64
import json
import re
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.common import (CLASS_COLORS, class_binary, components, dilate,  # noqa: E402
                             discover_dataset, experiment_root, folder_colors, normalise_class,
                             read_mask, write_rgb)

CLASSES = ["background", "molten_consumable", "droplet", "external_arc", "internal_arc",
           "weld_pool", "spatter"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}
PRIORITY = {"molten_consumable": 3, "droplet": 3, "spatter": 3, "internal_arc": 2,
            "external_arc": 1, "weld_pool": 1}
COCO_CATEGORY_ID = {"molten_consumable": 1, "droplet": 2, "external_arc": 3, "internal_arc": 4,
                    "weld_pool": 5, "spatter": 6}
SHORT = {"molten_consumable": "molten", "droplet": "droplet", "external_arc": "ext arc",
         "internal_arc": "int arc", "weld_pool": "weld pool", "spatter": "spatter"}


# --------------------------------------------------------------------------
# RLE (COCO, uncompressed, column-major)
# --------------------------------------------------------------------------
def rle_decode(seg: dict) -> np.ndarray:
    h, w = seg["size"]
    flat = np.zeros(h * w, bool)
    pos, value = 0, False
    for n in seg["counts"]:
        if value:
            flat[pos:pos + n] = True
        pos += n
        value = not value
    return flat.reshape((w, h)).T


def rle_encode(mask: np.ndarray) -> dict:
    flat = mask.T.reshape(-1).astype(np.int8)
    change = np.flatnonzero(np.diff(np.concatenate([[0], flat, [0]])))
    runs = np.diff(np.concatenate([[0], change, [flat.size]]))
    counts = runs.tolist()
    if counts and counts[-1] == 0:
        counts = counts[:-1]
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": [int(c) for c in counts]}


def pack_labels(labels: np.ndarray) -> str:
    """uint16 label map -> base64(zlib(little-endian bytes)) for the browser."""
    return base64.b64encode(zlib.compress(labels.astype("<u2").tobytes(), 6)).decode("ascii")


def unpack_labels(text: str, h: int, w: int) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(text)), dtype="<u2").reshape(h, w)


# --------------------------------------------------------------------------
# catalogue
# --------------------------------------------------------------------------
@dataclass
class Entry:
    key: str
    split: str
    experiment: str
    mask_path: Path
    image_path: Path
    json_path: Path | None
    result_path: Path | None

    @property
    def frame(self) -> str:
        return self.image_path.name


class Catalog:
    """All images, plus where edits live.

    Every save goes to the work folder (<out>_in_progress). As soon as every
    image of an experiment folder has been reviewed (saved), that folder is
    moved to the output folder -- so the output folder only ever holds fully
    reviewed experiments. Later saves in a finished experiment go straight to
    the output folder.
    """

    def __init__(self, dataset_root: Path, result_root: Path | None, out_root: Path,
                 splits: list[str], work_root: Path | None = None):
        # (tidy() is called at the end, once the entries are known)
        self.dataset_root = dataset_root
        self.out_root = out_root
        self.work_root = work_root or out_root.parent / f"{out_root.name}_in_progress"
        results: dict[str, Path] = {}
        if result_root is not None and result_root.exists():
            for p in result_root.rglob("*.png"):
                results.setdefault(p.name, p)
        self.entries: dict[str, Entry] = {}
        json_cache: dict[Path, Path | None] = {}
        for job in discover_dataset(dataset_root, splits, None, None):
            exp_root = experiment_root(job.mask_dir, splits, job.mask_dir.parent)
            if exp_root not in json_cache:
                found = sorted(p for p in exp_root.rglob("*.json")
                               if "outputs" not in {s.lower() for s in p.parts})
                json_cache[exp_root] = found[0] if found else None
            key = f"{job.split}/{job.experiment}/{job.mask_path.name}"
            self.entries[key] = Entry(key, job.split, job.experiment, job.mask_path,
                                      job.image_path, json_cache[exp_root],
                                      results.get(job.mask_path.name))
        self.tidy()

    def tree(self) -> list[dict]:
        groups: dict[tuple[str, str], list] = {}
        for e in self.entries.values():
            groups.setdefault((e.split, e.experiment), []).append(e)
        out = []
        for (split, exp), items in sorted(groups.items()):
            images = [{"key": e.key, "frame": e.frame, "has_result": e.result_path is not None,
                       "saved": self.saved_path(e).exists()}
                      for e in sorted(items, key=lambda x: x.frame)]
            out.append({"split": split, "experiment": exp, "images": images,
                        "published": (self.out_root / split / exp).is_dir()
                        and all(im["saved"] for im in images)})
        return out

    def folder(self, e: Entry) -> Path:
        """Where this experiment's saves live: the output folder once it is
        complete (moved there), otherwise the work folder."""
        done = self.out_root / e.split / e.experiment
        return done if done.is_dir() else self.work_root / e.split / e.experiment

    def saved_path(self, e: Entry) -> Path:
        return self.folder(e) / e.mask_path.name

    def saved_json(self, e: Entry) -> Path:
        return self.folder(e) / "instances_coco.json"

    def progress(self, e: Entry) -> tuple[int, int]:
        """(images of e's experiment reviewed, images in it)."""
        same = [x for x in self.entries.values()
                if x.split == e.split and x.experiment == e.experiment]
        return sum(self.saved_path(x).exists() for x in same), len(same)

    def publish_if_complete(self, e: Entry) -> Path | None:
        """Move the experiment's work folder to the output folder once every
        image in it has been reviewed. Returns the output folder, or None."""
        done, total = self.progress(e)
        if done < total:
            return None
        src = self.work_root / e.split / e.experiment
        dst = self.out_root / e.split / e.experiment
        if src.is_dir():
            _move_folder(src, dst)
        return dst

    def tidy(self) -> None:
        """Folders completed with an older version were copied, not moved:
        fold any leftover work copy into the output folder."""
        if not self.work_root.is_dir():
            return
        for split_dir in [d for d in self.work_root.iterdir() if d.is_dir()]:
            for src in [d for d in split_dir.iterdir() if d.is_dir()]:
                dst = self.out_root / split_dir.name / src.name
                if dst.is_dir():
                    _move_folder(src, dst)


def _move_folder(src: Path, dst: Path) -> None:
    import shutil
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.move(str(f), str(dst / f.name))
    try:
        src.rmdir()
        if not any(src.parent.iterdir()):
            src.parent.rmdir()
    except OSError:
        pass


@lru_cache(maxsize=8)
def load_coco(path: Path) -> dict:
    """file name -> [(annotation id, class, segmentation)] in JSON order."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cats = {c["id"]: normalise_class(c["name"]) for c in data.get("categories", [])}
    images = {im["id"]: Path(im["file_name"]).name for im in data.get("images", [])}
    out: dict[str, list] = {}
    for a in data.get("annotations", []):
        cls = cats.get(a["category_id"])
        if cls is None or a["image_id"] not in images:
            continue
        out.setdefault(images[a["image_id"]], []).append((a["id"], cls, a["segmentation"]))
    return out


def _seg_mask(seg, h: int, w: int) -> np.ndarray:
    if isinstance(seg, dict):
        return rle_decode(seg)
    m = np.zeros((h, w), np.uint8)
    for poly in seg:
        pts = np.asarray(poly, np.float64).reshape(-1, 2).round().astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


# --------------------------------------------------------------------------
# building instances
# --------------------------------------------------------------------------
def _name_instances(classes: list[str]) -> list[str]:
    count: dict[str, int] = {}
    names = []
    for c in classes:
        count[c] = count.get(c, 0) + 1
        names.append(f"{SHORT[c]} {count[c]}")
    return names


def _rgb_class_map(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros(rgb.shape[:2], np.uint8)
    for cls in CLASSES[1:]:
        out[class_binary(rgb, CLASS_COLORS[cls])] = CLASS_INDEX[cls]
    return out


def build_human(e: Entry) -> dict:
    """Human side = exactly the human RGB mask.

    Every pixel keeps the class the RGB mask gives it. The COCO JSON is only
    used to tell objects of the same class apart (droplet 1 vs droplet 2):
    each RGB pixel of class c goes to the JSON instance of class c covering
    it. Pixels no JSON instance covers join the same-class instance they
    touch, or become their own piece. Same-class JSON instances that overlap
    are "conflicts" (the later one keeps the pixels by default; the user can
    switch). JSON instances with no pixels in the RGB mask are returned
    separately as `extra` (shown as "only in the JSON").
    """
    rgb = read_mask(e.mask_path)
    h, w = rgb.shape[:2]
    rgb_cls = _rgb_class_map(rgb)
    anns = load_coco(e.json_path).get(e.frame, []) if e.json_path else []
    json_masks = []
    for _, cls, seg in anns:
        m = _seg_mask(seg, h, w)
        if m.shape == (h, w) and m.any():
            json_masks.append((cls, m))

    labels = np.zeros((h, w), np.uint16)
    extra_labels = np.zeros((h, w), np.uint16)
    conflict_map = np.zeros((h, w), np.uint16)
    instances, extra, conflicts = [], [], []
    next_id = 1
    for cls in CLASSES[1:]:
        region = rgb_cls == CLASS_INDEX[cls]
        mine = [m for c, m in json_masks if c == cls]
        owner = np.zeros((h, w), np.int32)          # index into `pieces` + 1
        pieces = [("json", m) for m in mine]
        ties: dict[tuple[int, int], np.ndarray] = {}
        for j, (_, m) in enumerate(pieces, start=1):
            cover = m & region
            clash = cover & (owner > 0)
            for prev in np.unique(owner[clash]):
                pair = (int(prev), j)
                ties[pair] = ties.get(pair, np.zeros((h, w), bool)) | (clash & (owner == prev))
            owner[cover] = j
        for comp in components(region & (owner == 0)):
            ring = owner[dilate(comp, 1) & ~comp]
            ring = ring[ring > 0]
            if len(ring):
                owner[comp] = int(np.bincount(ring).argmax())
            else:
                pieces.append(("rgb", comp))
                owner[comp] = len(pieces)

        id_of = {}
        count = 0
        for j, (source, m) in enumerate(pieces, start=1):
            here = owner == j
            if here.any():
                count += 1
                id_of[j] = next_id
                labels[here] = next_id
                instances.append({"id": next_id, "name": f"{SHORT[cls]} {count}", "cls": cls,
                                  "source": source})
                next_id += 1
        for j, (source, m) in enumerate(pieces, start=1):
            if source == "json" and j not in id_of:
                count += 1
                extra_labels[m & (extra_labels == 0)] = next_id
                extra.append({"id": next_id, "name": f"{SHORT[cls]} {count} (only in JSON)",
                              "cls": cls, "pixels": int(m.sum())})
                next_id += 1
        for (a, b), region_ab in ties.items():
            if a not in id_of or b not in id_of:
                continue
            region_ab &= (owner == a) | (owner == b)
            if not region_ab.any():
                continue
            cid = len(conflicts) + 1
            conflict_map[region_ab] = cid
            conflicts.append({"id": cid, "a": id_of[a], "b": id_of[b], "pixels": int(region_ab.sum()),
                              "default": id_of[b]})
    return {"labels": labels, "instances": instances, "conflict_map": conflict_map,
            "conflicts": conflicts, "extra_labels": extra_labels, "extra": extra,
            "source": "coco" if json_masks else "rgb", "size": (h, w)}


def instances_from_rgb(rgb: np.ndarray, human_labels: np.ndarray, human_inst: list[dict]
                       ) -> tuple[np.ndarray, list[dict]]:
    """Connected pieces per class, named after the human instance they overlap most."""
    h, w = rgb.shape[:2]
    labels = np.zeros((h, w), np.uint16)
    human_by_id = {i["id"]: i for i in human_inst}
    out, used, new_count = [], {}, {}
    k = 0
    for cls in CLASSES[1:]:
        for comp in components(class_binary(rgb, CLASS_COLORS[cls])):
            k += 1
            labels[comp] = k
            ids, counts = np.unique(human_labels[comp], return_counts=True)
            match = None
            for hid, _ in sorted(zip(ids, counts), key=lambda t: -t[1]):
                if hid and human_by_id[int(hid)]["cls"] == cls:
                    match = int(hid)
                    break
            if match is not None:
                used[match] = used.get(match, 0) + 1
                name = human_by_id[match]["name"]
                if used[match] > 1:
                    name += f" (part {used[match]})"
            else:
                new_count[cls] = new_count.get(cls, 0) + 1
                name = f"{SHORT[cls]} new {new_count[cls]}"
            out.append({"id": k, "name": name, "cls": cls, "humanId": match, "samId": k})
    return labels, out


def build_image(cat: Catalog, key: str) -> dict:
    e = cat.entries[key]
    palette = folder_colors(e.mask_path.parent)   # this folder's own colour scheme
    human = build_human(e)
    h, w = human["size"]
    if e.result_path is not None:
        rgb = read_mask(e.result_path, e.mask_path.parent)
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_NEAREST)
        sam_labels, sam_inst = instances_from_rgb(rgb, human["labels"], human["instances"])
        sam_source = str(e.result_path)
    else:
        sam_labels = human["labels"].copy()
        sam_inst = [dict(i, humanId=i["id"], samId=i["id"]) for i in human["instances"]]
        sam_source = "human (no result file found)"

    result_labels, result_inst, result_source = sam_labels, sam_inst, "result folder"
    saved = load_saved(cat, e, h, w)
    if saved is not None:
        result_labels, result_inst = saved
        result_source = "saved edit"

    return {
        "key": key, "split": e.split, "experiment": e.experiment, "frame": e.frame,
        "width": w, "height": h,
        "classes": [{"name": c, "short": SHORT.get(c, c), "color": list(palette[c]),
                     "priority": PRIORITY.get(c, 0)} for c in CLASSES],
        "human": {"instances": human["instances"], "labels": pack_labels(human["labels"]),
                  "source": human["source"], "path": str(e.mask_path),
                  "extra": human["extra"], "extraLabels": pack_labels(human["extra_labels"])},
        "conflicts": human["conflicts"], "conflictMap": pack_labels(human["conflict_map"]),
        "sam": {"instances": sam_inst, "labels": pack_labels(sam_labels), "source": sam_source},
        "result": {"instances": result_inst, "labels": pack_labels(result_labels),
                   "source": result_source},
        "savedPath": str(cat.saved_path(e)),
    }


# --------------------------------------------------------------------------
# saving
# --------------------------------------------------------------------------
def load_saved(cat: Catalog, e: Entry, h: int, w: int):
    path = cat.saved_json(e)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    image = next((im for im in data.get("images", []) if im["file_name"] == e.frame), None)
    if image is None:
        return None
    labels = np.zeros((h, w), np.uint16)
    inst = []
    for k, a in enumerate((a for a in data["annotations"] if a["image_id"] == image["id"]), 1):
        cls = next(n for n, cid in COCO_CATEGORY_ID.items() if cid == a["category_id"])
        labels[rle_decode(a["segmentation"])] = k
        inst.append({"id": k, "name": a.get("instance_name", f"{SHORT[cls]} {k}"), "cls": cls,
                     "humanId": a.get("human_ref"), "samId": a.get("sam_ref")})
    return labels, inst


def save_folder(cat: Catalog, split: str, experiment: str) -> dict:
    """Mark every not-yet-reviewed image of one experiment folder as reviewed,
    saving each exactly as it currently is (the last save, else the result
    folder's mask)."""
    todo = [e for e in cat.entries.values()
            if e.split == split and e.experiment == experiment and not cat.saved_path(e).exists()]
    last = None
    for e in sorted(todo, key=lambda x: x.frame):
        d = build_image(cat, e.key)
        last = save_image(cat, {"key": e.key, "width": d["width"], "height": d["height"],
                                "labels": d["result"]["labels"], "instances": d["result"]["instances"]})
    any_entry = next(e for e in cat.entries.values() if e.split == split and e.experiment == experiment)
    done, total = cat.progress(any_entry)
    return {"saved": len(todo), "reviewed": done, "total": total,
            "published": last["published"] if last else None, "experiment": f"{split}/{experiment}"}


def save_image(cat: Catalog, payload: dict) -> dict:
    e = cat.entries[payload["key"]]
    palette = folder_colors(e.mask_path.parent)
    h, w = int(payload["height"]), int(payload["width"])
    labels = unpack_labels(payload["labels"], h, w)
    classes = {int(i["id"]): i for i in payload["instances"]}

    rgb = np.zeros((h, w, 3), np.uint8)
    annotations = []
    for iid in np.unique(labels):
        if iid == 0 or int(iid) not in classes:
            continue
        info = classes[int(iid)]
        m = labels == iid
        rgb[m] = palette[info["cls"]]
        ys, xs = np.nonzero(m)
        annotations.append({
            "category_id": COCO_CATEGORY_ID[info["cls"]], "iscrowd": 0,
            "area": int(m.sum()),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                     int(ys.max() - ys.min() + 1)],
            "segmentation": rle_encode(m), "instance_name": info.get("name", ""),
            "human_ref": info.get("humanId"), "sam_ref": info.get("samId")})
    mask_out = cat.saved_path(e)
    write_rgb(mask_out, rgb)
    # the folder's colour scheme travels with the saved masks
    mapping = mask_out.parent / "class_rgb_mapping.txt"
    names = {"background": "Background", "molten_consumable": "Molten Consumable",
             "droplet": "Droplet", "external_arc": "External arc", "internal_arc": "Internal arc",
             "weld_pool": "Weld pool", "spatter": "Spatter"}
    mapping.write_text("".join(f"{names[c]}: {', '.join(map(str, palette[c]))}\n" for c in CLASSES),
                       encoding="utf-8")

    json_out = cat.saved_json(e)
    data = (json.loads(json_out.read_text(encoding="utf-8")) if json_out.exists() else
            {"images": [], "categories": [{"id": cid, "name": n} for n, cid in
                                          COCO_CATEGORY_ID.items()], "annotations": []})
    image = next((im for im in data["images"] if im["file_name"] == e.frame), None)
    if image is None:
        image = {"id": max([im["id"] for im in data["images"]], default=0) + 1,
                 "file_name": e.frame, "height": h, "width": w}
        data["images"].append(image)
    data["annotations"] = [a for a in data["annotations"] if a["image_id"] != image["id"]]
    next_id = max([a["id"] for a in data["annotations"]], default=0) + 1
    for a in annotations:
        a.update(id=next_id, image_id=image["id"])
        next_id += 1
        data["annotations"].append(a)
    json_out.write_text(json.dumps(data), encoding="utf-8")
    done, total = cat.progress(e)
    published = cat.publish_if_complete(e)
    return {"mask": str(mask_out), "json": str(json_out), "instances": len(annotations),
            "reviewed": done, "total": total, "published": str(published) if published else None,
            "experiment": f"{e.split}/{e.experiment}"}
