# unsupervised_chest-xray

Unsupervised clustering of chest X-ray images.  
Pipeline: **ResNet50 (feature extractor) -> PCA -> KMeans**

## Dataset
[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
Classes: `NORMAL` / `PNEUMONIA`

## Project structure
```
unsupervised_chest-xray/
├── data/raw/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── saved_model/            # created after running train.py
│   ├── pipeline.pkl        # single model file: PCA + KMeans
│   ├── features_2d.npy     # 2-D projections for visualization
│   ├── cluster_labels.npy
│   ├── original_labels.npy
│   └── clustering_plot.png
├── train.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Train and save the model
python train.py

# 2. Predict from terminal (single file or folder)
python predict.py data/raw/NORMAL/IM-0001-0001.jpeg
python predict.py data/raw/PNEUMONIA/

# 3. Run web app
streamlit run app.py
```

## How it works

1. **ResNet50** extracts a 2048-dimensional feature vector from each image.  
   The classification head is removed; only convolutional features are used.
2. **PCA** compresses 2048 dimensions to 50, removing noise and speeding up clustering.
3. **KMeans** (k=2) groups images into two clusters without seeing any labels.

Steps 2 and 3 are wrapped in a single `sklearn.pipeline.Pipeline` object
and saved as `pipeline.pkl` with `joblib`.