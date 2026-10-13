"""Phân tích lỗi theo lớp (Câu 9). Chạy từ thư mục gốc của lab (có data/ và work/).

    python analysis_q9.py --pred work/e3.jsonl --out report/figures/q9_errors.png
In ra: kích thước box trung bình theo lớp (train), và vẽ ảnh test minh hoạ bỏ sót / dự đoán thừa.
"""
import argparse, json, random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as pa
from PIL import Image
from scripts.labcommon import read_jsonl

W, H = 640, 360


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def gt_boxes(e):  # YOLO chuẩn hoá -> pixel
    out = []
    for o in e["objects"]:
        xc, yc, w, h = o["yolo"]
        out.append((o["label"], [(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H]))
    return out


def pred_boxes(raw):  # 0-1000 -> pixel
    try:
        d = json.loads(raw)
        return [(o["label"], [o["bbox_2d"][0] * W / 1000, o["bbox_2d"][1] * H / 1000,
                              o["bbox_2d"][2] * W / 1000, o["bbox_2d"][3] * H / 1000]) for o in d]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="report/figures/q9_errors.png")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()

    # 1) Kích thước box trung bình theo lớp trên tập train
    tr = read_jsonl(f"{a.data}/splits/train.jsonl")
    stat = {}
    for e in tr:
        for o in e["objects"]:
            _, _, w, h = o["yolo"]
            stat.setdefault(o["label"], []).append(w * h)
    print("Diện tích box trung bình / diện tích ảnh (train):")
    for k, v in stat.items():
        v = sorted(v)
        print(f"  {k:8s} n={len(v):5d}  trung bình={sum(v) / len(v) * 100:.2f}%  trung vị={v[len(v) // 2] * 100:.2f}%")

    # 2) Ảnh minh hoạ: bỏ sót pothole (FN) và crack dự đoán thừa (FP)
    te = read_jsonl(f"{a.data}/splits/test.jsonl")
    preds = {r["image_id"]: pred_boxes(r["raw"]) for r in map(json.loads, open(a.pred, encoding="utf-8"))}
    fn_pothole, fp_crack = [], []
    for e in te:
        g, p = gt_boxes(e), preds.get(e["image_id"], [])
        if any(l == "pothole" and not any(pl == "pothole" and iou(b, pb) >= 0.5 for pl, pb in p) for l, b in g):
            fn_pothole.append(e)
        if any(pl == "crack" and not any(l == "crack" and iou(b, pb) >= 0.5 for l, b in g) for pl, pb in p):
            fp_crack.append(e)
    print(f"ảnh có pothole bị bỏ sót: {len(fn_pothole)} | ảnh có crack dự đoán thừa: {len(fp_crack)}")
    rnd = random.Random(0)
    picks = [("pothole bị bỏ sót", e) for e in rnd.sample(fn_pothole, min(3, len(fn_pothole)))] + \
            [("crack dự đoán thừa", e) for e in rnd.sample(fp_crack, min(3, len(fp_crack)))]
    fig, axs = plt.subplots(2, 3, figsize=(15, 6))
    for ax, (title, e) in zip(axs.ravel(), picks):
        ax.imshow(Image.open(Path(a.data) / e["image_path"]).convert("RGB"))
        for l, b in gt_boxes(e):
            ax.add_patch(pa.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color="lime", lw=2))
        for l, b in preds.get(e["image_id"], []):
            ax.add_patch(pa.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color="red", lw=1.5, ls="--"))
        ax.set_title(f"{title}  (xanh = nhãn thật, đỏ = dự đoán)", fontsize=9)
        ax.axis("off")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(a.out, dpi=120)
    print("đã lưu", a.out)


if __name__ == "__main__":
    main()
