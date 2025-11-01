
# 🧠 Brain Tumor Classification using Deep Learning (MRI Images)

This project focuses on automated *brain tumor detection and classification* using MRI images with state-of-the-art deep learning techniques. The model identifies *four categories* of brain scans:

* ✅ *Glioma*
* ✅ *Meningioma*
* ✅ *Pituitary tumor*
* ✅ *No tumor*

We evaluated three powerful transfer learning models — *ResNet50, **EfficientNetB3, and **DenseNet121* — and selected *DenseNet121* for deployment due to its superior accuracy and robustness.
The project also integrates *Grad-CAM heatmaps* to improve explainability and highlight tumor-affected regions.

A *Streamlit-based web app* is included for real-time MRI classification and visualization.

---

## 🚀 Project Highlights

| Feature        | Description                                         |
| -------------- | --------------------------------------------------- |
| Dataset        | Brain MRI Dataset (Kaggle)                          |
| Models Tested  | ResNet50, EfficientNetB3, DenseNet121, Baseline CNN |
| Final Model    | *DenseNet121*                                     |
| Explainability | *Grad-CAM heatmaps*                               |
| UI             | *Streamlit Web App*                               |
| Input Type     | Brain MRI (2D images)                               |
| Task           | Multi-class Image Classification                    |

---

## 📂 Project Structure


📦 Brain-Tumor-Classification
├─ 📁 dataset/                     # MRI images (train/test)
├─ 📁 notebooks/                  # Training & evaluation notebooks
├─ 📁 models/                     # Saved .h5 models
├─ 📁 plots/                      # Confusion matrices, reports
├─ app.py                         # Streamlit web app
├─ gradcam.py                     # GradCAM utility
├─ requirements.txt               # Dependencies
└─ README.md


---

## ⚙ Tech Stack

* *Python, **TensorFlow / Keras*
* *NumPy, Pandas, Matplotlib, OpenCV*
* *Streamlit* (web UI)
* *Grad-CAM for explainability*

---

## 📊 Model Performance

| Model                   | Accuracy     | Notes                            |
| ----------------------- | ------------ | -------------------------------- |
| Baseline CNN            | ~68%         | Limited feature learning         |
| ResNet50                | High         | Good performance, slower         |
| EfficientNetB3          | Very High    | Efficient + powerful             |
| *DenseNet121 (Final)* | ⭐ *Best* ⭐ | Deep feature reuse, top accuracy |

> DenseNet121 was selected for deployment based on metrics & generalization.

---

## 🔍 Grad-CAM Explainability

Grad-CAM visualizations provide heatmaps showing the tumor-focused areas in MRI images, enhancing model transparency and clinical trust.

---

## 🖥 Streamlit App

Launch the app:

bash
streamlit run app.py


Upload an MRI scan → model predicts tumor type + displays heatmap.

---

## 📥 Dataset Source

Dataset used: *Brain Tumor MRI Dataset from Kaggle*

> [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)

---

## 🛠 Installation

Clone repo:

bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification


Install dependencies:

bash
pip install -r requirements.txt


---

## ✅ Future Enhancements

* 📌 Add *3D MRI processing* (volumetric CNNs)
* 📌 Deploy as *cloud API / mobile app*
* 📌 Use *multi-center MRI datasets*
* 📌 Integrate *SHAP / LIME explainability*
* 📌 Lightweight model for *edge deployment*

---

## 👨‍⚕ Motivation

Early and accurate brain tumor detection saves lives.
This project demonstrates the potential of *AI-assisted radiology* in improving diagnostic efficiency and accuracy.

---

## 🤝 Contributing

Pull requests and enhancements are welcome!

---
