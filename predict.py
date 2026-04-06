"""
predict.py -- тестирование сохранённого pipeline через терминал.

Использование:
    python predict.py path/to/image.jpg
    python predict.py path/to/folder/
"""

import sys
import os
import numpy as np
import joblib

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image

MODEL_DIR = 'saved_model'
IMG_SIZE  = (224, 224)


def load_pipeline():
    """Загружает обученный Pipeline (PCA + KMeans) из файла."""
    path = os.path.join(MODEL_DIR, 'pipeline.pkl')
    if not os.path.exists(path):
        print(f"Ошибка: файл {path} не найден. Сначала запустите train.py.")
        sys.exit(1)
    return joblib.load(path)


def build_cluster_map(pipeline):
    """
    Определяет соответствие номера кластера и имени класса.
    Загружает оригинальные метки, сохранённые во время обучения,
    и для каждого кластера находит доминирующий класс.
    """
    original_labels  = np.load(os.path.join(MODEL_DIR, 'original_labels.npy'), allow_pickle=True)
    cluster_labels   = np.load(os.path.join(MODEL_DIR, 'cluster_labels.npy'))

    cluster_map = {}
    for cid in np.unique(cluster_labels):
        mask    = cluster_labels == cid
        subset  = original_labels[mask]
        unique, counts = np.unique(subset, return_counts=True)
        cluster_map[int(cid)] = unique[np.argmax(counts)] 

    return cluster_map


def extract_features(img_path: str, extractor) -> np.ndarray:
    """Загружает изображение и возвращает вектор признаков 2048-D."""
    img  = image.load_img(img_path, target_size=IMG_SIZE)
    x    = image.img_to_array(img)
    x    = np.expand_dims(x, axis=0)
    x    = preprocess_input(x)
    return extractor.predict(x, verbose=0).flatten()


def predict_single(img_path: str, extractor, pipeline, cluster_map):
    """
    Предсказывает кластер для одного изображения.
    Возвращает (cluster_id, class_name, confidence_pct).

    Уверенность вычисляется как доля расстояния до своего центроида
    относительно суммы всех расстояний (чем меньше расстояние, тем выше).
    """
    feat      = extract_features(img_path, extractor)
    feat_2d   = feat.reshape(1, -1)                      

    cid       = int(pipeline.predict(feat_2d)[0])

    feat_pca  = pipeline['pca'].transform(feat_2d)      
    distances = pipeline['kmeans'].transform(feat_pca)[0] 
    confidence = (1 - distances[cid] / distances.sum()) * 100

    return cid, cluster_map.get(cid, f"Cluster {cid}"), confidence


def main():
    if len(sys.argv) < 2:
        print("Использование: python predict.py <путь_к_изображению_или_папке>")
        sys.exit(1)

    path = sys.argv[1]

    print("Загрузка ResNet50...")
    extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    print("Загрузка pipeline...")
    pipeline    = load_pipeline()
    cluster_map = build_cluster_map(pipeline)
    print(f"Маппинг кластеров: {cluster_map}\n")

    if os.path.isdir(path):
        files = [
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
    else:
        files = [path]

    if not files:
        print("Изображения не найдены.")
        sys.exit(1)

    print(f"{'Файл':<42} {'Кластер':>8} {'Класс':<14} {'Уверенность':>12}")
    print("-" * 78)

    for fpath in files:
        try:
            cid, cname, conf = predict_single(fpath, extractor, pipeline, cluster_map)
            fname = os.path.basename(fpath)
            print(f"{fname:<42} {cid:>8}   {cname:<14} {conf:>10.1f}%")
        except Exception as e:
            print(f"{os.path.basename(fpath):<42} [Ошибка: {e}]")


if __name__ == '__main__':
    main()