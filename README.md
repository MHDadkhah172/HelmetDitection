# 🛡️ Safety Helmet Detection using YOLO & CRISP-DM

This project is a computer vision pipeline to detect safety helmets in construction sites and industrial working areas. The project follows the **CRISP-DM** framework step-by-step.

---

## 📌 Phase 1: Business Understanding

* **Problem:** In industrial and construction sites, workers not wearing safety helmets face high accident risks. Manually checking every worker is hard and inefficient.
* **Goal:** Build an automated system using YOLO object detection to identify who is wearing a helmet and who is not.
* **Target Metric:** Achieve a reliable **mAP@0.5 (>80%)** for helmet detection to use in real-time safety monitoring.

---

## 📌 Phase 2: Data Understanding

* **Source:** Hard Hat Workers dataset from Roboflow Universe (v11 - Augmented 3x).
* **Classes:** 3 target classes:
  1. `head` (Unprotected head)
  2. `helmet` (Safety helmet worn)
  3. `person` (Worker detected)
* **Exploration:** Checked dataset multi-class annotations ensuring balanced sample distribution across safety helmets and unprotected heads.

---

## 📌 Phase 3: Data Preparation

* **Preprocessing & Augmentation:** Images resized to 640x640 with 3x data augmentation for orientation, brightness, and positioning robustness.
* **Structure:** Train, Validation, and Test splits cleanly organized.
* **Configuration:** Finalized `data/data.yaml` with 3 target classes for YOLOv11 architecture.

---

## 📌 Phase 4: Modeling
* **Architecture:** YOLO11 Nano (`yolo11n.pt`) pre-trained backbone.
* **Pipeline:** Encapsulated training module inside `src/train.py`.
* **Hyperparameters:**
  * Image Resolution: $640 \times 640$
  * Batch Size: 16
  * Epochs: 15
* **Execution Strategy:** Scalable training pipeline utilizing Google Colab GPU (T4) infrastructure.