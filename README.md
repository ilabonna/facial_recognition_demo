# 🧠 Facial Recognition System (MTCNN + FaceNet + OpenCV)

This project implements a **real-time facial recognition system** using deep learning–based face embeddings (FaceNet) and the MTCNN detector.  
It allows you to:
- 📸 **Capture and store** face images from your webcam  
- 🧠 **Train** a model to recognize those faces  
- 🎥 **Recognize** faces in real time using your webcam  

---

## 🧩 Project Features
- ✅ Multi-step pipeline: Capture → Train → Recognize  
- 🧠 Deep learning embeddings via **keras-facenet (FaceNet)**  
- 👁️ Robust detection via **MTCNN**  
- 🔴 **Red box + “NOT RECOGNIZED”** for unknown faces  
- 🟢 **Green box + “You”** for recognized faces  
- 💾 Lightweight `.pkl` model storage  
- 🧱 Fully compatible with Windows + Python 3.10  

---

## ⚙️ Project Structure
```
facial_demo/
│
├── capture_faces.py              # Capture and save your facial images
├── face_recognition_demo.py      # Train + recognize pipeline
├── models/
│   ├── data/
│   │   └── me/                   # Folder containing your face images
│   └── template.pkl              # Auto-generated model after training
├── venv/                         # Virtual environment (excluded from Git)
└── README.md
```

---

## 🧰 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/facial_recognition_demo.git
cd facial_recognition_demo
```

---

### 2️⃣ Create a Virtual Environment
Create a clean Python environment to avoid dependency conflicts.

**Windows (PowerShell):**
```bash
python -m venv venv
```

Activate it:
```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\activate
```

You should see `(venv)` appear before your prompt.

To deactivate later:
```bash
deactivate
```

---

### 3️⃣ Install Required Libraries
Once your environment is active, install all dependencies:

```bash
pip install tensorflow keras keras-facenet mtcnn opencv-python numpy scikit-learn
```

Optional (recommended for IDEs):
```bash
pip install jupyter notebook
```

---

### 4️⃣ (Windows Only) Fix Missing DLLs
If you see this error:
```
ImportError: Could not find the DLL(s) 'msvcp140.dll or msvcp140_1.dll'
```
Install the **Microsoft Visual C++ Redistributable** from Microsoft:  
👉 [https://aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)

---

### 5️⃣ Folder Setup for Capturing Faces
Make sure your folder structure looks like this before capturing:
```
models/
└── data/
    └── me/
```

If `me/` doesn’t exist, create it manually.

---

## 🎥 1. Capture Your Face Images
Run this script to collect images through your webcam:

```bash
python capture_faces.py
```

**Controls:**
- Press **S** → Save the current frame  
- Press **Q** → Quit  

Images will be stored under:
```
models/data/me/
```

Aim for ~30–100 well-lit, front-facing samples.

---

## 🧠 2. Train the Model
Once you’ve captured images:
```bash
python face_recognition_demo.py --mode train --dataset ./models/data --models_dir ./models
```

This will:
- Extract face embeddings  
- Create a `template.pkl` file under `models/`  
- Display:
  ```
  [OK] Saved template: models/template.pkl threshold=0.85
  ```

---

## 👁️ 3. Run Real-Time Recognition
After training, start recognition:
```bash
python face_recognition_demo.py --mode recognize --models_dir ./models
```

**Controls:**
- Press **Q** to quit  
- Webcam window will show:
  - 🟢 Green box + “You” for recognized faces  
  - 🔴 Red box + “NOT RECOGNIZED” for unknown faces  

If your webcam doesn’t open, try a different index:
```bash
python face_recognition_demo.py --mode recognize --models_dir ./models --camera_index 1
```

---

## 🧾 Optional Settings
| Argument | Description | Default |
|-----------|--------------|----------|
| `--threshold` | Override recognition sensitivity | Auto-computed |
| `--prob_threshold` | SVM classification cutoff (for multi-user) | 0.60 |
| `--min_face_size` | MTCNN detection size | 40 |
| `--max_per_class` | Limit training images per person | 0 (no limit) |

---

## 🔒 .gitignore Setup
Your `.gitignore` excludes:
```
venv/
models/*.pkl
models/data/**/*.jpg
models/data/**/*.jpeg
models/data/**/*.png
```
✅ Keeps your dataset and model private  
✅ Only commits scripts, not images or large binaries  

---

## 🧩 Troubleshooting
| Issue | Cause | Solution |
|--------|--------|-----------|
| `No embeddings extracted` | MTCNN didn’t detect your face | Re-capture clearer images |
| `Could not open camera index 0` | Wrong webcam index | Try `--camera_index 1` |
| TensorFlow DLL error | Missing Microsoft VC++ redistributable | [Download fix](https://aka.ms/vs/17/release/vc_redist.x64.exe) |

---

## 💡 Future Enhancements
- Add **voice feedback** when recognized  
- Log recognition timestamps  
- Deploy with **Flask or Streamlit** for web-based recognition  
- Integrate hardware (e.g., Raspberry Pi door unlock demo)

---

## 🧑‍💻 Author
**Ipshita Labonna**  
Electrical & Computer Engineering, University of Michigan–Dearborn  
💼 [LinkedIn](https://www.linkedin.com/) (add your profile link here)

---

## 📜 License
This project is released under the [MIT License](https://opensource.org/licenses/MIT).

---

## ✅ TL;DR Quick Commands
```bash
# Create & activate venv
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install tensorflow keras keras-facenet mtcnn opencv-python numpy scikit-learn

# Capture images
python capture_faces.py

# Train model
python face_recognition_demo.py --mode train --dataset ./models/data --models_dir ./models

# Run recognition
python face_recognition_demo.py --mode recognize --models_dir ./models
```
