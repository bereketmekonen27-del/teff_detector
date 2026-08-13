# Teff Plant Disease Detector (`teff_detector`)

An importable, portable Edge AI computer vision package designed to classify Ethiopian Teff grass crop health conditions using transfer learning (ResNet-18). 

Developed for easy integration into mobile apps, web servers, or localized agricultural drone/satellite imagery analysis tools.

## 📊 Classifications (5 Core Conditions)
The model evaluates input images and categorizes Teff plant health into one of five distinct visual classes:
*   **`healthy`**: Clean, vibrant green Teff stalks and foliage.
*   **`teff_rust`**: Leaves showcasing localized, powdery orange or reddish-brown fungal pustules.
*   **`head_smudge`**: Fungal growth forming distinct dark brown or black sticky, soot-like mats directly on mature grain heads.
*   **`pest_damage`**: Physical leaf structural damage (chewed margins, holes, or withered centers) caused by local insects like the Teff Shoot Fly or Grasshoppers.
*   **`wilting_moisture_stress`**: Uniformly drooping, pale, or tightly curled leaves with straw-yellow tip burning due to parched soil or drought conditions.

---

## 🚀 Quick Setup Guide

### 1. Installation via GitHub
You can install this package globally directly from this repository by running:
```bash
pip install git+https://github.com
```

### 2. Local Editable Installation (For Developers)
If you are modifying the training script or updating weights, clone the directory, navigate inside, and run:
```bash
pip install -e .
```

---

## 💻 Quick-Start Code Template

Once installed, the AI engine can be imported and executed in any separate Python project or script using this minimal framework:

```python
from teff_detector import TeffClassifier
import os

# 1. Initialize the portable Teff AI brain (auto-loads internal weights)
print("🔄 Waking up Teff AI Engine...")
classifier = TeffClassifier()

# 2. Define the path to your field image
target_photo = "field_sample.jpg"

if os.path.exists(target_photo):
    print(f"📸 Scanning crop sample: {target_photo}...")
    
    # 3. Execute the prediction pipeline
    result = classifier.predict(target_photo)
    
    # 4. Display findings
    print("\n" + "="*40)
    print(f"🚨 DIAGNOSIS : {result['diagnosis'].upper().replace('_', ' ')}")
    print(f"📊 CONFIDENCE: {result['confidence']:.2f}%")
    print("="*40 + "\n")
else:
    print(f"❌ Error: Could not locate '{target_photo}'. Verify your file path.")
```

---

## 🛠️ Package Structure
```text
teff_detector_package/
├── setup.py                  # Installation metadata configuration
└── teff_detector/            # Source module
    ├── __init__.py           # Package exposure logic
    ├── detector.py           # Core PyTorch model execution logic
    └── teff_disease_model.pth # Fine-tuned neural network weights file
```

---

## ⚙️ Requirements
*   `torch`
*   `torchvision`
*   `Pillow`
