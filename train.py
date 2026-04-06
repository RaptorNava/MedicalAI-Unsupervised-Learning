import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------
DATA_PATH  = 'data/raw'   # путь к папкам NORMAL / PNEUMONIA
IMG_SIZE   = (224, 224)
N_SAMPLES  = 300          # количество изображений из каждой папки
N_CLUSTERS = 2
MODEL_DIR  = 'saved_model'

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Загрузка ResNet50 как экстрактора признаков
# include_top=False убирает финальный классификатор ImageNet (1000 классов),
# pooling='avg' применяет Global Average Pooling -> вектор 2048-D на выходе
# ---------------------------------------------------------------------------
print("Загрузка ResNet50...")
extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')


def extract_features(img_path: str) -> np.ndarray:
    """Загружает одно изображение и возвращает вектор признаков размером 2048."""
    img  = image.load_img(img_path, target_size=IMG_SIZE)
    x    = image.img_to_array(img)
    x    = np.expand_dims(x, axis=0)   # добавляем batch-измерение: (1, 224, 224, 3)
    x    = preprocess_input(x)          # нормализация под статистику ImageNet
    feat = extractor.predict(x, verbose=0)
    return feat.flatten()              # (1, 2048) -> (2048,)


def load_dataset(base_path: str):
    """
    Обходит подпапки base_path, извлекает признаки.
    Возвращает:
        features      -- np.ndarray формы (N, 2048)
        labels        -- list[str], реальные метки (только для оценки в конце)
    """
    features, labels = [], []

    categories = [
        f for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f))
    ]

    for category in categories:
        folder = os.path.join(base_path, category)
        files  = os.listdir(folder)[:N_SAMPLES]
        print(f"  {category}: {len(files)} файлов")

        for fname in files:
            fpath = os.path.join(folder, fname)
            try:
                features.append(extract_features(fpath))
                labels.append(category)
            except Exception as e:
                print(f"    Пропущен {fname}: {e}")

    return np.array(features), labels


# ---------------------------------------------------------------------------
# Шаг 1. Извлечение признаков из всего датасета
# ---------------------------------------------------------------------------
print("\nИзвлечение признаков из датасета...")
features, original_labels = load_dataset(DATA_PATH)
print(f"Готово. features.shape = {features.shape}")

# ---------------------------------------------------------------------------
# Шаг 2. Построение и обучение Pipeline
#
# Pipeline объединяет два шага в одну модель:
#   pca    -- снижает размерность с 2048 до 50 главных компонент,
#             убирает шум и ускоряет кластеризацию
#   kmeans -- делит пространство на N_CLUSTERS групп без использования меток
#
# Вся логика предобработки + кластеризации хранится в одном объекте,
# что упрощает сохранение и последующее использование.
# ---------------------------------------------------------------------------
pipeline = Pipeline([
    ('pca',    PCA(n_components=50, random_state=42)),
    ('kmeans', KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)),
])

print("\nОбучение Pipeline (PCA -> KMeans)...")
predicted_clusters = pipeline.fit_predict(features)

# ---------------------------------------------------------------------------
# Шаг 3. Сохранение Pipeline
# Единственный файл, необходимый для предсказания новых изображений
# ---------------------------------------------------------------------------
pipeline_path = os.path.join(MODEL_DIR, 'pipeline.pkl')
joblib.dump(pipeline, pipeline_path)
print(f"Pipeline сохранён: {pipeline_path}")

# ---------------------------------------------------------------------------
# Шаг 4. Сохранение вспомогательных данных
# PCA до 2-D нужен только для визуализации, не входит в pipeline
# ---------------------------------------------------------------------------
pca_2d      = PCA(n_components=2, random_state=42)
features_2d = pca_2d.fit_transform(features)

np.save(os.path.join(MODEL_DIR, 'features_2d.npy'),     features_2d)
np.save(os.path.join(MODEL_DIR, 'cluster_labels.npy'),  predicted_clusters)
np.save(os.path.join(MODEL_DIR, 'original_labels.npy'), np.array(original_labels))

# ---------------------------------------------------------------------------
# Шаг 5. Оценка качества (только для анализа, метки в обучении не участвовали)
# ARI близкий к 1 означает, что кластеры совпадают с реальными классами
# ---------------------------------------------------------------------------
ari = adjusted_rand_score(original_labels, predicted_clusters)
nmi = normalized_mutual_info_score(original_labels, predicted_clusters)
print(f"\nAdjusted Rand Index : {ari:.4f}  (1.0 = идеально)")
print(f"Normalized MI       : {nmi:.4f}  (1.0 = идеально)")

# ---------------------------------------------------------------------------
# Шаг 6. Визуализация результатов
# Левый  -- что алгоритм нашёл самостоятельно
# Правый -- реальные метки для сравнения (алгоритм их не видел)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Chest X-Ray Clustering (ResNet50 + PCA + KMeans)", fontsize=14)

axes[0].scatter(features_2d[:, 0], features_2d[:, 1],
                c=predicted_clusters, cmap='winter', alpha=0.55, s=12)
axes[0].set_title("KMeans (Unsupervised)")
axes[0].set_xlabel("PC 1")
axes[0].set_ylabel("PC 2")

true_colors = [0 if lbl == 'NORMAL' else 1 for lbl in original_labels]
sc = axes[1].scatter(features_2d[:, 0], features_2d[:, 1],
                     c=true_colors, cmap='coolwarm', alpha=0.55, s=12)
axes[1].set_title("Ground Truth (Normal vs Pneumonia)")
axes[1].set_xlabel("PC 1")
cbar = plt.colorbar(sc, ax=axes[1], ticks=[0, 1])
cbar.ax.set_yticklabels(['Normal', 'Pneumonia'])

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, 'clustering_plot.png')
plt.savefig(plot_path, dpi=150)
print(f"График сохранён: {plot_path}")
print("\nОбучение завершено.")