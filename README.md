#  Chest X-Ray Clustering (Unsupervised Learning)

This project implements an unsupervised machine learning pipeline to automatically group chest X-ray images into categories (e.g., "Normal" vs. "Pneumonia") without using any ground-truth labels during the training phase. It leverages Deep Learning for feature extraction and Classical ML for dimensionality reduction and clustering.

---

##  How It Works (The Pipeline)

The system processes images through a three-stage pipeline:

1.  **Feature Extraction (ResNet50):** Each image is passed through a ResNet50 model (pre-trained on ImageNet). The final classification layer is removed, and Global Average Pooling is applied to transform each image into a **2048-dimensional feature vector**.
2.  **Dimensionality Reduction (PCA):** To handle the "curse of dimensionality" and remove noise, Principal Component Analysis (PCA) compresses the 2048 features into the **50 most significant principal components**.
3.  **Clustering (K-Means):** The K-Means algorithm (k=2) analyzes the 50-D vectors to find two distinct clusters based purely on visual patterns and mathematical similarity.

The PCA and K-Means steps are encapsulated within a single `sklearn.pipeline.Pipeline` object for seamless inference.

---

##  Project Structure

```text
unsupervised_chest-xray/
├── data/raw/              # Input images (subfolders: NORMAL / PNEUMONIA)
├── saved_model/           # Artifacts generated after training
│   ├── pipeline.pkl       # CORE MODEL: Serialized PCA + KMeans pipeline
│   ├── clustering_plot.png # Visual validation of cluster separation
│   ├── original_labels.npy # Real labels used for post-train mapping
│   ├── features_2d.npy    # 2D projections for the plot
│   └── cluster_labels.npy # Cluster assignments for training data
├── train.py               # Main script for feature extraction and training
├── predict.py             # CLI tool for testing new images
├── app.py                 # Streamlit-based Web UI
└── requirements.txt       # Python dependencies
