# ===== CHẠY TOÀN BỘ LAB TRÊN KAGGLE, 2 GPU SONG SONG: E0, E1, E2, E3 → E4 (data_size) =====
# Dán vào một cell của notebook Kaggle (GPU T4 x2, Internet On). Mỗi job chạy ở một tiến trình riêng
# gắn cố định vào một GPU. Chạy lại sau khi bị ngắt: bước nào đã có kết quả sẽ được bỏ qua.
import os, sys, json, pathlib, shutil, subprocess, time, zipfile
from concurrent.futures import ThreadPoolExecutor

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
T0 = time.time()

LAB = pathlib.Path("/kaggle/working/lab")
if not (LAB / "lab_lib.py").exists():
    src = next(pathlib.Path("/kaggle/input").rglob("lab_lib.py")).parent
    shutil.copytree(src, LAB, dirs_exist_ok=True)
os.chdir(LAB)
print("thư mục làm việc:", os.getcwd())
assert '"model.visual"' in pathlib.Path("lab_lib.py").read_text(encoding="utf-8"), \
    "lab_lib.py chưa vá skip_modules (thiếu model.visual)"


def sh(cmd, tag="", env=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, **(env or {})})
    for line in p.stdout:
        print(f"{tag}{line}", end="", flush=True)
    return p.wait()


NG = max(1, len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()))
print(f"Số GPU thấy được: {NG}")

sh([sys.executable, "-m", "pip", "-q", "install", "-r", "requirements-colab.txt"])
if not pathlib.Path("data/splits/meta.json").exists():
    SRC = next(pathlib.Path("/kaggle/input").rglob("labels-YOLO")).parent
    assert sh([sys.executable, "scripts/prepare_data.py", "--source", str(SRC)]) == 0, "prepare_data lỗi"
sh([sys.executable, "-c",
    "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-4B-Instruct'); print('model đã tải xong')"])

pathlib.Path("worker.py").write_text('''
import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from lab_lib import *
job = json.loads(sys.argv[1]); kind = job.pop("kind")
if kind == "e0":
    m, p = load_model()
    run_inference(m, p, "test", "work/e0.jsonl")
elif kind == "train":
    train(**job)
elif kind == "infer":
    pm, p = load_trained(job["ckpt"])
    run_inference(pm, p, "test", job["out"], coord=job.get("coord", "norm1000"))
''', encoding="utf-8")


def job(gpu, **j):
    gpu = gpu % NG
    print(f"\n[GPU{gpu}] >>> {j}   (đã chạy {(time.time() - T0) / 60:.0f} phút)", flush=True)
    return sh([sys.executable, "-u", "worker.py", json.dumps(j)], f"[GPU{gpu}] ",
              {"CUDA_VISIBLE_DEVICES": str(gpu)}) == 0


def score(name, pred, out, ckpt=None):
    cmd = [sys.executable, "scripts/score.py", "--pred", pred, "--name", name, "--out", out]
    if ckpt:
        r = json.load(open(f"{ckpt}/run.json"))
        for k in ("trainable_params", "peak_vram_gb", "train_minutes"):
            cmd += ["--meta", f"{k}={r[k]}"]
    return sh(cmd, "[score] ") == 0


EVID = pathlib.Path("report/evidence")
EVID.mkdir(parents=True, exist_ok=True)
pathlib.Path("work").mkdir(exist_ok=True)
FAILED, NOTES = [], []


def experiment(gpu, name, setup, ckpt, evid, **kw):
    if pathlib.Path(evid).exists():
        print(f"[bỏ qua] {name}: đã có {evid}")
        return
    if not pathlib.Path(f"{ckpt}/run.json").exists():
        ok = job(gpu, kind="train", setup=setup, out_dir=ckpt, **kw)
        if not ok:
            print(f"!! {name}: train lỗi, thử lại với bs=1, accum=8")
            NOTES.append(f"{name}: phải dùng bs=1, accum=8 (ghi vào báo cáo)")
            ok = job(gpu, kind="train", setup=setup, out_dir=ckpt, **{**kw, "bs": 1, "accum": 8})
        if not ok:
            FAILED.append(name + " (train)")
            return
    pred = f"work/{name.lower()}.jsonl"
    if not job(gpu, kind="infer", ckpt=ckpt, out=pred):
        FAILED.append(name + " (suy luận)")
        return
    if not score(name, pred, str(evid), ckpt):
        FAILED.append(name + " (chấm điểm)")


def run_e0(gpu):
    if (EVID / "e0.json").exists():
        print("[bỏ qua] E0")
        return
    if job(gpu, kind="e0"):
        score("E0", "work/e0.jsonl", str(EVID / "e0.json"))
    else:
        FAILED.append("E0")


def exp_n(gpu, n):
    experiment(gpu, f"E{n}", f"E{n}", f"ckpt/e{n}", EVID / f"e{n}.json")


def lane0():
    exp_n(0, 3)
    run_e0(0)


def lane1():
    exp_n(1, 1)
    exp_n(1, 2)


with ThreadPoolExecutor(max_workers=min(2, NG)) as ex:
    for f in [ex.submit(lane0), ex.submit(lane1)]:
        f.result()

# E4: ablation data_size (10% vs 100%). 100% chính là setup tốt nhất của E1-E3.
f1 = {}
for n in ("e1", "e2", "e3"):
    p = EVID / f"{n}.json"
    if p.exists():
        f1[n] = json.load(open(p))["micro"]["f1"]
if f1:
    best = max(f1, key=f1.get)
    print(f"\nSetup tốt nhất theo F1: {best.upper()}  {f1}")
    if not (EVID / "e4_data_size_1.0.json").exists():
        score("E4_data_size_1.0", f"work/{best}.jsonl", str(EVID / "e4_data_size_1.0.json"), f"ckpt/{best}")
    experiment(0, "E4_data_size_0.1", best.upper(), "ckpt/e4_ds_0.1",
               EVID / "e4_data_size_0.1.json", fraction=0.1)
else:
    FAILED.append("E4 (không có kết quả E1-E3)")

print("\n" + "=" * 100)
for p in sorted(EVID.glob("*.json")):
    r = json.load(open(p))
    print(f"{r['name']:20s} F1={r['micro']['f1']:.3f} CI95={r['f1_ci95'][0]:.3f}-{r['f1_ci95'][1]:.3f} "
          f"FP/ảnh={r['fp_per_image']:.2f} valid={r['parse_valid_rate']:.2f} meta={r.get('meta', {})}")
if FAILED:
    print("\nBƯỚC LỖI:", FAILED)
for n in NOTES:
    print("GHI CHÚ:", n)
with zipfile.ZipFile("/kaggle/working/results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for pat in ("report/evidence/*.json", "work/*.jsonl", "ckpt/*/run.json"):
        for f in pathlib.Path(".").glob(pat):
            z.write(f)
print(f"\nXONG sau {(time.time() - T0) / 60:.0f} phút.")
