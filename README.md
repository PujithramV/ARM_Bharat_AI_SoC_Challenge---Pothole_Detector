# Real-Time Edge Pothole Detection 🚗🛣️
**ARM Bharat AI SoC Challenge - Final Submission**

This repository contains an edge-optimized, real-time object detection system designed to identify road anomalies (potholes) using a Raspberry Pi 4. The project strictly adheres to the challenge rubrics, balancing high accuracy with hardware-accelerated inference speeds on an Arm-based CPU.

## 📊 Performance Metrics
* **Hardware:** Raspberry Pi 4 (Model B)
* **Model Format:** TensorFlow Lite (INT8 Quantized)
* **Mean Average Precision (mAP):** `0.878`
* **Inference Speed:** `~5.6 FPS` (Exceeding the $\geq$5 FPS target requirement)

---

## 🧠 1. Model Training & Optimization Pipeline
To achieve the required balance of speed and accuracy on an edge device, the model pipeline was heavily optimized:
1. **Base Model:** A YOLOv5 object detection model was initially trained on a custom dataset of road potholes using PyTorch.
2. **Resolution Scaling:** The model inputs were scaled down to `320x320` to reduce computational overhead on the Pi's CPU.
3. **INT8 Quantization:** The massive PyTorch weights were converted and fully quantized to 8-bit integers (`best-int8.tflite`). This reduced the model size by over 70% and drastically accelerated memory access times without sacrificing significant accuracy.

---

## 💻 2. Hardware & Software Stack
* **OS:** Raspberry Pi OS (Debian Bookworm/Trixie)
* **Camera:** USB Web Camera (Processed via Video4Linux2 / V4L2)
* **Environment Manager:** [Pixi](https://pixi.sh/) (For highly reproducible, isolated package management)
* **Dependencies:** Python 3.10, `opencv-python`, `tflite-runtime`, `numpy < 2.0`

---

## ⚙️ 3. Under the Hood: Code Explanation
The core inference script (`main.py`) relies on highly optimized libraries to process video feeds in near-real-time. Here is how the engine works:

### Key Libraries Used:
* `cv2` **(OpenCV):** Handles all visual data. It captures the live webcam feed, resizes the frames to `320x320` for the AI, and draws the final bounding boxes and text overlays on the screen.
* `tflite_runtime.interpreter`: A lightweight alternative to the full TensorFlow library. It acts as the "brain," loading our `.tflite` model, allocating memory (`allocate_tensors`), and running the heavy math (`invoke`).
* `numpy`: Handles the complex array matrix transformations, ensuring the image data is in the exact `uint8` array format the INT8 model expects.

### Key Functions & Logic:
* **Multi-Threading:** The TFLite Interpreter is explicitly commanded to use `num_threads=4` to unlock all four physical cores of the Raspberry Pi's Arm CPU.
* **Buffer Management:** `cv2.CAP_PROP_BUFFERSIZE` is set to `1` to prevent the camera from caching old frames, completely eliminating input lag.
* **Non-Max Suppression (NMS):** We utilize `cv2.dnn.NMSBoxes()` with an IOU (Intersection Over Union) threshold of `0.45`. AI models often draw multiple overlapping boxes around a single pothole. NMS mathematically filters these out, keeping only the single box with the highest confidence score.

---

## 🚀 4. How to Run This Project
This project uses **Pixi** to ensure flawless cross-platform reproduction without dependency conflicts (specifically addressing the recent NumPy 2.0 architecture changes).

Initialize the Pixi environment: (Bash commands)

pixi init \n
pixi add "python=3.10" opencv "numpy<2" \n
pixi add --pypi tflite-runtime

running the script, use this command ---> pixi run python main.py
