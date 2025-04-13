---
license: mit
language:
- en
library_name: transformers
tags:
- cv
- robotics
---
# 🌌 Celestial-Mini: Lightweight Object Detection Model (TF)

[![TensorFlow](https://img.shields.io/badge/framework-TensorFlow-orange)](https://www.tensorflow.org/)
[![Object Detection](https://img.shields.io/badge/task-Object%20Detection-blue)]()
[![Models](https://img.shields.io/badge/targets-80%20Objects-green)]()

**Celestial-Mini** is a compact, high-performance object detection model designed to recognize up to **80 distinct object classes**. Built with **TensorFlow**, it balances speed and accuracy for deployment in edge devices and real-time applications.

---

## 🚀 Key Features

- 🔍 Detects up to **80 different object categories**
- ⚡ Optimized for **real-time inference**
- 🧠 Built on a **lightweight backbone**
- 📦 TensorFlow SavedModel format for easy deployment
- 🧰 Compatible with TensorFlow Lite and TensorFlow.js

---

## 🧪 Intended Use

Celestial-Mini is designed for:

- Robotics and drones
- Smart home devices
- Augmented Reality (AR) systems
- Mobile applications
- Educational and prototyping environments

---

## 🏷 Object Classes

Includes detection support for the standard 80-class COCO-style object categories such as:

```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, ...
```

---

## 📦 How to Use

```python
import tensorflow as tf

# Load the model
model = tf.saved_model.load("path/to/celestial-mini")

# Run inference
detections = model(input_tensor)
```

---

## 📊 Performance

| Metric         | Value         |
|----------------|---------------|
| Classes        | 80            |
| Model Size     | ~15MB         |
| Inference Time | < 50ms/image  |
| Framework      | TensorFlow    |

> 📌 Performance may vary depending on hardware and TensorFlow backend optimizations.

---

## 🧠 Training & Dataset

Celestial-Mini was trained on a custom variant of the **COCO dataset**, emphasizing generalization and real-time inference. Model architecture includes quantization-friendly layers and depthwise separable convolutions.

---

## 📖 Citation

If you use **Celestial-Mini** in your work, please consider citing:

```
@misc{celestialmini2025,
  title={Celestial-Mini: A Lightweight Real-Time Object Detector},
  author={Lang, John},
  year={2025},
  howpublished={\url{https://huggingface.co/langutang/celestial-mini}}
}
```

---

## 📬 Contact & License

- 📫 For questions or collaboration, open an issue or contact the maintainer.
- ⚖️ License: MIT (see LICENSE file for details)

---

## 🌠 Hugging Face Model Hub

To load from Hugging Face:

```python
from transformers import AutoFeatureExtractor, TFModelForObjectDetection

model = TFModelForObjectDetection.from_pretrained("langutang/celestial-mini")
extractor = AutoFeatureExtractor.from_pretrained("langutang/celestial-mini")
```

---

Transform your edge AI projects with the power of **Celestial-Mini** 🌠
