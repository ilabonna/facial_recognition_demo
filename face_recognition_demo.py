# face_recognition_demo.py
# Usage:
#   Train: python face_recognition_demo.py --mode train --dataset ./models/data --models_dir ./models
#   Recognize: python face_recognition_demo.py --mode recognize --models_dir ./models
#
# Dependencies:
#   pip install keras-facenet mtcnn opencv-python numpy scikit-learn tensorflow keras

import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Optional (used for multi-user SVM mode)
try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:
    SVC = None
    LabelEncoder = None

IMG_EXTS = (".jpg", ".jpeg", ".png")

# ---------- Utility Functions ----------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def l2_normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + eps)

def clip_box(x, y, w, h, W, H):
    x = max(0, x); y = max(0, y)
    w = max(1, w); h = max(1, h)
    if x + w > W: w = W - x
    if y + h > H: h = H - y
    return x, y, w, h

def read_image(p: Path) -> np.ndarray:
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"cv2.imread failed for {p}")
    return img

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_face(rgb: np.ndarray, size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)

# ---------- Face Pipeline ----------

class FacePipeline:
    def __init__(self, min_face_size: int = 40):
        self.detector = MTCNN()
        self.embedder = FaceNet()  # pretrained FaceNet model

    def detect(self, img_bgr: np.ndarray) -> List[Dict]:
        return self.detector.detect_faces(img_bgr)

    def crop_faces_rgb(self, img_bgr: np.ndarray, margin: float = 0.20) -> List[np.ndarray]:
        H, W = img_bgr.shape[:2]
        faces = self.detect(img_bgr)
        crops = []
        for f in faces:
            x, y, w, h = f.get("box", [0, 0, 0, 0])
            mx, my = int(w * margin), int(h * margin)
            x2, y2, w2, h2 = x - mx, y - my, w + 2 * mx, h + 2 * my
            x2, y2, w2, h2 = clip_box(x2, y2, w2, h2, W, H)
            crop_bgr = img_bgr[y2:y2 + h2, x2:x2 + w2]
            crop_rgb = bgr_to_rgb(crop_bgr)
            crops.append(preprocess_face(crop_rgb))
        return crops

    def embed_many(self, faces_rgb: List[np.ndarray]) -> np.ndarray:
        if not faces_rgb:
            return np.empty((0, 128), dtype=np.float32)
        emb = self.embedder.embeddings(faces_rgb)
        return emb.astype(np.float32)

# ---------- Dataset Handling ----------

def list_classes(root: Path) -> List[Path]:
    return sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name.lower())

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

def load_dataset(dataset_root: Path, pipeline: FacePipeline, max_per_class: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    if not dataset_root.exists():
        raise ValueError(f"Dataset not found: {dataset_root}")

    leaf_contains_images = any(f.is_file() and f.suffix.lower() in IMG_EXTS for f in dataset_root.iterdir())
    if leaf_contains_images:
        dataset_root = dataset_root.parent

    class_dirs = list_classes(dataset_root)
    if not class_dirs:
        raise ValueError(f"No class folders found under {dataset_root}")

    all_emb, all_lab = [], []

    for cdir in class_dirs:
        imgs = list_images(cdir)
        if not imgs:
            continue
        if max_per_class > 0:
            imgs = imgs[:max_per_class]

        for p in imgs:
            img_bgr = read_image(p)
            crops = pipeline.crop_faces_rgb(img_bgr)
            if not crops:
                continue
            areas = [c.shape[0] * c.shape[1] for c in crops]
            face = crops[int(np.argmax(areas))]
            emb = pipeline.embed_many([face])
            if emb.shape[0] == 0:
                continue
            all_emb.append(emb[0])
            all_lab.append(cdir.name)

    if not all_emb:
        raise ValueError(f"No embeddings extracted from dataset {dataset_root}")

    return np.vstack(all_emb), np.array(all_lab)

# ---------- Training ----------

def save_template(models_dir: Path, embeddings: np.ndarray, labels: np.ndarray, threshold_override: float = None):
    ensure_dir(models_dir)
    emb = l2_normalize(embeddings, axis=1)
    centroid = l2_normalize(emb.mean(axis=0, keepdims=True), axis=1)[0]

    dists = np.linalg.norm(emb - centroid, axis=1)
    mu, sigma = float(dists.mean()), float(dists.std() or 0.05)
    threshold = float(mu + 3.0 * sigma if threshold_override is None else threshold_override)

    with open(models_dir / "template.pkl", "wb") as f:
        pickle.dump({"centroid": centroid.astype(np.float32), "threshold": threshold}, f)
    print(f"[OK] Saved template: {models_dir/'template.pkl'} threshold={threshold:.4f}")

# ---------- Recognition Utilities ----------

def load_template(models_dir: Path):
    p = models_dir / "template.pkl"
    if not p.exists():
        return None, None
    with open(p, "rb") as f:
        obj = pickle.load(f)
    return obj["centroid"], float(obj["threshold"])

def predict_template(centroid: np.ndarray, threshold: float, emb: np.ndarray):
    emb = l2_normalize(emb[None, :], axis=1)[0]
    dist = float(np.linalg.norm(emb - centroid))
    label = "You" if dist <= threshold else "Unknown"
    score = 1.0 - min(dist / max(threshold, 1e-6), 1.0)
    return label, score, dist

# ---------- Drawing ----------

def draw_box(img: np.ndarray, box: Tuple[int, int, int, int], label: str, conf: float):
    x, y, w, h = box
    if "Unknown" in label or "NOT RECOGNIZED" in label:
        color = (0, 0, 255)  # Red for unknown
        display_label = "NOT RECOGNIZED"
    else:
        color = (0, 255, 0)  # Green for recognized
        display_label = label
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, display_label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

# ---------- Recognize ----------

def recognize(args):
    models_dir = Path(args.models_dir).expanduser().resolve()
    pipeline = FacePipeline(min_face_size=args.min_face_size)

    centroid, thr = load_template(models_dir)
    if centroid is None:
        raise RuntimeError("No model found. Train first.")

    print(f"[INFO] Using template verification. threshold={thr:.4f}")

    cap = cv2.VideoCapture(int(args.camera_index))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera_index}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            H, W = frame.shape[:2]
            faces = pipeline.detector.detect_faces(frame)
            for f in faces:
                x, y, w, h = f.get("box", [0, 0, 0, 0])
                x, y, w, h = clip_box(x, y, w, h, W, H)
                crop_bgr = frame[y:y + h, x:x + w]
                crop_rgb = bgr_to_rgb(crop_bgr)
                face = preprocess_face(crop_rgb)
                emb = pipeline.embed_many([face])[0]

                lab, score, dist = predict_template(centroid, thr, emb)
                if lab == "Unknown":
                    draw_box(frame, (x, y, w, h), "NOT RECOGNIZED", score)
                else:
                    draw_box(frame, (x, y, w, h), "You", score)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Facial recognition demo (template-based).")
    p.add_argument("--mode", choices=["train", "recognize"], required=True)
    p.add_argument("--dataset", type=str, help="Dataset root (subfolders per person). Required for train.")
    p.add_argument("--models_dir", type=str, default="./models")
    p.add_argument("--camera_index", type=int, default=0)
    p.add_argument("--min_face_size", type=int, default=40)
    p.add_argument("--max_per_class", type=int, default=0)
    p.add_argument("--threshold", type=float, default=None)
    return p.parse_args()

def train(args):
    dataset_root = Path(args.dataset).expanduser().resolve()
    models_dir = Path(args.models_dir).expanduser().resolve()
    ensure_dir(models_dir)
    pipeline = FacePipeline(min_face_size=args.min_face_size)
    embeddings, labels = load_dataset(dataset_root, pipeline, max_per_class=args.max_per_class)
    uniq = np.unique(labels)
    print(f"[INFO] Classes discovered: {list(uniq)} (N={len(labels)})")
    save_template(models_dir, embeddings, labels, threshold_override=args.threshold)

def main():
    args = parse_args()
    if args.mode == "train":
        if not args.dataset:
            raise SystemExit("Error: --dataset required for training")
        train(args)
    elif args.mode == "recognize":
        recognize(args)

if __name__ == "__main__":
    main()
