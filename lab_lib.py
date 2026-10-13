"""Hàm cho Lab Day 10 (GUIDE bước 2-7). Chạy từ thư mục gốc của lab (có scripts/, data/).

Dùng trong notebook:
    import os; os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # TRƯỚC khi import torch
    from lab_lib import *
"""
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image

from scripts.labcommon import read_jsonl

CLASSES = ("pothole", "crack", "manhole")
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DATA = Path("data")


# ───────────────────────── Bước 2: nhãn YOLO → JSON ─────────────────────────
def make_prompt(coord="norm1000"):
    scale = ("integers on a 0-1000 scale, relative to image width (x) and height (y)"
             if coord == "norm1000" else "integers in absolute pixels of the image")
    return (
        "Detect all road damage in this dashcam image. Classes: pothole, crack, manhole. "
        "Reply with ONLY a JSON list, no explanation and no markdown, in the form "
        '[{"label": "pothole", "bbox_2d": [x1, y1, x2, y2]}]. '
        f"Coordinates are {scale}. If there is no damage, reply [] ."
    )


def yolo_to_box(label, yolo, coord="norm1000", size=(640, 360)):
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else size
    hi_x, hi_y = sx, sy
    x1, x2 = (xc - w / 2) * sx, (xc + w / 2) * sx
    y1, y2 = (yc - h / 2) * sy, (yc + h / 2) * sy
    box = [min(hi_x, max(0, round(x1))), min(hi_y, max(0, round(y1))),
           min(hi_x, max(0, round(x2))), min(hi_y, max(0, round(y2)))]
    return {"label": label, "bbox_2d": box}


def objects_to_answer(objects, coord="norm1000", size=(640, 360)):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, size) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))  # trên→dưới, trái→phải
    return json.dumps(boxes, separators=(",", ":"))


def box_to_pixels(box, size=(640, 360)):
    """Quy ngược 0-1000 về pixel để vẽ lại kiểm tra."""
    x1, y1, x2, y2 = box
    return [x1 * size[0] / 1000, y1 * size[1] / 1000, x2 * size[0] / 1000, y2 * size[1] / 1000]


# ───────────────────────── Bước 3: nạp model + suy luận ─────────────────────────
def load_model():
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    q = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],  # ViT + projector giữ fp16
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=q, dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"  # bắt buộc cho generate theo batch
    return model, processor


@torch.no_grad()
def run_inference(model, processor, split, out_path, limit=None, batch_size=4,
                  coord="norm1000", max_new_tokens=256):
    entries = read_jsonl(DATA / "splits" / f"{split}.jsonl")
    if limit:
        entries = entries[:limit]
    prompt = make_prompt(coord)
    model.eval()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(0, len(entries), batch_size):
            chunk = entries[i:i + batch_size]
            msgs = [[{"role": "user", "content": [
                {"type": "image", "image": Image.open(DATA / e["image_path"]).convert("RGB")},
                {"type": "text", "text": prompt}]}] for e in chunk]
            inputs = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", padding=True).to("cuda")
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, use_cache=True)
            raws = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            for e, raw in zip(chunk, raws):
                f.write(json.dumps({"image_id": e["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            if (i // batch_size) % 10 == 0:
                print(f"{min(i + batch_size, len(entries))}/{len(entries)}")


# ───────────────────────── Bước 4: ví dụ huấn luyện + mask loss ─────────────────────────
def encode_example(processor, image, prompt, answer):
    user = {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}
    asst = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    p = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    f = processor.apply_chat_template([user, asst], add_generation_prompt=False, **kw)
    n = p["input_ids"].shape[1]
    assert torch.equal(f["input_ids"][0, :n], p["input_ids"][0]), "prompt không phải tiền tố"
    labels = f["input_ids"].clone()
    labels[:, :n] = -100                       # chỉ học phần trả lời
    if processor.tokenizer.decode(labels[0, -1:]) == "\n":
        labels[0, -1] = -100                   # bỏ "\n" sau <|im_end|>, giữ <|im_end|>
    out = {k: v for k, v in f.items()}
    out["labels"] = labels
    return out


def collate(batch, pad_id):
    pads = {"input_ids": pad_id, "attention_mask": 0, "labels": -100, "mm_token_type_ids": 0}
    L = max(b["input_ids"].shape[1] for b in batch)
    out = {}
    for k, v in pads.items():
        if k not in batch[0]:
            continue
        rows = []
        for b in batch:
            t = b[k][0]
            rows.append(torch.cat([t, t.new_full((L - t.shape[0],), v)]))
        out[k] = torch.stack(rows)
    for k in ("pixel_values", "image_grid_thw"):  # nối, KHÔNG pad
        out[k] = torch.cat([b[k] for b in batch], 0)
    return out


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, processor, split="train", fraction=1.0, coord="norm1000", seed=42):
        entries = read_jsonl(DATA / "splits" / f"{split}.jsonl")
        if fraction < 1.0:
            random.Random(seed).shuffle(entries)
            entries = entries[:max(1, int(len(entries) * fraction))]
        self.entries, self.processor, self.coord = entries, processor, coord
        self.prompt = make_prompt(coord)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        img = Image.open(DATA / e["image_path"]).convert("RGB")
        ans = objects_to_answer(e["objects"], self.coord, (e["width"], e["height"]))
        return encode_example(self.processor, img, self.prompt, ans)


# ───────────────────────── Bước 5: setup LoRA / đóng băng ─────────────────────────
LLM_RE = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_RE = r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"
PROJ_KEYS = ("visual.merger", "visual.deepstack_merger_list")


def apply_setup(model, setup, r=16, alpha=32):
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()
    regex = LLM_RE if setup in ("E1", "E2") else f"(?:{LLM_RE})|(?:{VIT_RE})"
    pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05,
                                          target_modules=regex))
    if setup in ("E2", "E3"):
        for n, p in pm.named_parameters():
            if any(k in n for k in PROJ_KEYS):
                p.data = p.data.float()        # fp32 để GradScaler unscale được
                p.requires_grad = True
    matched = sum(1 for _, m in pm.named_modules() if hasattr(m, "lora_A"))
    trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    print(f"{setup}: module LoRA={matched}, tham số huấn luyện={trainable:,}")
    return pm, matched, trainable


def extra_trainable(pm):
    return {n: p.detach().cpu() for n, p in pm.named_parameters()
            if p.requires_grad and any(k in n for k in PROJ_KEYS)}


# ───────────────────────── Bước 6: vòng huấn luyện ─────────────────────────
def train(setup, out_dir, max_steps=300, bs=2, accum=4, lr=2e-4, r=16, alpha=32,
          fraction=1.0, coord="norm1000", seed=42, save_every=100):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    model, processor = load_model()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm, matched, trainable = apply_setup(model, setup, r, alpha)
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0)
    warm = max(1, int(0.05 * max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, max_steps - warm))))
    scaler = torch.amp.GradScaler("cuda")
    pad_id = processor.tokenizer.pad_token_id
    dl = torch.utils.data.DataLoader(
        TrainSet(processor, "train", fraction, coord, seed), batch_size=bs, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id), num_workers=2, drop_last=True)

    def batches():
        while True:
            yield from dl
    it = batches()

    pm.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for step in range(1, max_steps + 1):
        tot = 0.0
        for _ in range(accum):
            batch = {k: v.to("cuda") for k, v in next(it).items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = pm(**batch).loss / accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step} (nguy cơ tràn fp16)")
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print(f"step {step}/{max_steps}  loss {tot:.4f}  {(time.time() - t0) / 60:.1f} phút")
        if step % save_every == 0 or step == max_steps:
            pm.save_pretrained(out_dir)                       # chỉ adapter LoRA
            extra = extra_trainable(pm)
            if extra:                                         # projector E2/E3 lưu riêng
                torch.save(extra, out_dir / "extra_trainable.pt")

    run = {"setup": setup, "trainable_params": trainable, "matched_modules": matched,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "train_minutes": round((time.time() - t0) / 60, 1), "max_steps": max_steps}
    (out_dir / "run.json").write_text(json.dumps(run, indent=2))
    print(run)
    del pm, model, opt, dl, it
    gc.collect()
    torch.cuda.empty_cache()
    return run


# ───────────────────────── Bước 7: nạp lại model đã train ─────────────────────────
def load_trained(out_dir):
    from peft import PeftModel
    model, processor = load_model()
    pm = PeftModel.from_pretrained(model, str(out_dir))
    extra_path = Path(out_dir) / "extra_trainable.pt"
    if extra_path.exists():
        res = pm.load_state_dict(torch.load(extra_path), strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
        print(f"đã nạp {len(torch.load(extra_path))} tensor projector")
    return pm, processor
