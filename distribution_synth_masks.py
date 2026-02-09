import os
import numpy as np
import imageio.v2 as imageio
from collections import Counter
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
import shutil
import tempfile
import random

# ============================================================
# CONFIGURATION
# ============================================================

# RGB -> class name (AJUSTA si cambiaste la paleta)
RGB_TO_LABEL = {
    (0,   0,   0):     "Background",
    (140, 128, 115):   "Soft tissue",
    (255, 255, 255):   "Bone",
    (64,  115, 140):   "CSF",
    (217, 217, 217):   "White matter",
    (153, 153, 153):   "Gray matter",
    (179, 38,  38):    "Ischemic infarct",
}

LABEL_ORDER = [
    "Background",
    "Soft tissue",
    "Bone",
    "CSF",
    "White matter",
    "Gray matter",
    "Ischemic infarct",
]

# Paths
REAL_MASKS_DIR = "path_to_real_masks"
SYN_MASKS_DIR  = "path_to_synthetic_masks"

real_imgs = [f for f in os.listdir(REAL_MASKS_DIR) if f.lower().endswith(".png")]
synthetic_imgs = [f for f in os.listdir(SYN_MASKS_DIR) if f.lower().endswith(".png")]

n_real = len(real_imgs)
n_syn = len(synthetic_imgs)

if n_real != n_syn:
    print(f"WARNING: Number of real masks ({n_real}) and synthetic masks ({n_syn}) differ.")
    
    n_common = min(n_real, n_syn)
    print(f"Using {n_common} masks from each dataset.")

    # Crear directorios temporales
    temp_real_dir = tempfile.mkdtemp(prefix="real_masks_")
    temp_syn_dir  = tempfile.mkdtemp(prefix="synthetic_masks_")

    # Muestreo aleatorio (sin reemplazo)
    real_selected = random.sample(real_imgs, n_common)
    syn_selected  = random.sample(synthetic_imgs, n_common)

    # Copiar archivos
    for fname in real_selected:
        shutil.copy(
            os.path.join(REAL_MASKS_DIR, fname),
            os.path.join(temp_real_dir, fname)
        )

    for fname in syn_selected:
        shutil.copy(
            os.path.join(SYN_MASKS_DIR, fname),
            os.path.join(temp_syn_dir, fname)
        )

    # Redefinir rutas para el resto del pipeline
    REAL_MASKS_DIR = temp_real_dir
    SYN_MASKS_DIR  = temp_syn_dir

else:
    print("Same number of masks. Using original directories.")


# Defineix RGB de la lesió
INFARCT_RGB = (179, 38, 38)
# ============================================================
# FUNCTIONS
# ============================================================

def compute_class_distribution_rgb(folder_path):
    pixel_counter = Counter()
    total_pixels = 0

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".png"):
            continue

        img = imageio.imread(os.path.join(folder_path, fname))

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"{fname} is not an RGB image")

        h, w, _ = img.shape
        total_pixels += h * w

        for rgb, label in RGB_TO_LABEL.items():
            mask = np.all(img == np.array(rgb, dtype=np.uint8), axis=-1)
            pixel_counter[label] += np.sum(mask)

    distribution = {
        label: 100.0 * pixel_counter[label] / total_pixels
        for label in LABEL_ORDER
    }

    return distribution


def print_distribution(title, dist):
    print(f"\n{title}")
    print("-" * len(title))
    for k, v in dist.items():
        print(f"{k:20s}: {v:6.2f} %")
        
def compute_infarct_areas(folder_path):
    areas = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".png"):
            continue
        img = imageio.imread(os.path.join(folder_path, fname))
        mask = np.all(img == np.array((179,38,38), dtype=np.uint8), axis=-1)
        areas.append(np.sum(mask))
    return np.array(areas)


def compute_infarct_complexity(folder_path, min_area=20):
    complexities = []

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".png"):
            continue

        img = imageio.imread(os.path.join(folder_path, fname))
        mask = np.all(img == np.array(INFARCT_RGB, dtype=np.uint8), axis=-1)

        labeled = label(mask)
        props = regionprops(labeled)

        for p in props:
            if p.area >= min_area and p.perimeter > 0:
                complexity = p.perimeter / np.sqrt(p.area)
                complexities.append(complexity)

    return np.array(complexities)


def compute_mask_class_features(folder_path):
    features = []

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".png"):
            continue

        img = imageio.imread(os.path.join(folder_path, fname))
        h, w, _ = img.shape
        total_pixels = h * w

        vec = []
        for rgb, label in RGB_TO_LABEL.items():
            mask = np.all(img == np.array(rgb, dtype=np.uint8), axis=-1)
            vec.append(np.sum(mask) / total_pixels)

        features.append(vec)

    return np.array(features)

def count_lesions(folder_path, min_area=10):
    """
    Cuenta el número de lesiones (objetos conectados) por imagen
    en la carpeta folder_path. Se descartan regiones menores a min_area.
    """
    lesion_counts = []

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".png"):
            continue

        img = imageio.imread(os.path.join(folder_path, fname))
        mask = np.all(img == np.array(INFARCT_RGB, dtype=np.uint8), axis=-1)

        labeled = label(mask)
        props = regionprops(labeled)

        # Solo contamos regiones mayores o iguales a min_area
        count = sum(1 for p in props if p.area >= min_area)
        lesion_counts.append(count)

    return np.array(lesion_counts)





# ============================================================
# COMPUTE DISTRIBUTIONS
# ============================================================

real_dist = compute_class_distribution_rgb(REAL_MASKS_DIR)
synthetic_dist = compute_class_distribution_rgb(SYN_MASKS_DIR)

print_distribution("REAL DATASET", real_dist)
print_distribution("SYNTHETIC DATASET", synthetic_dist)

# ============================================================
# PLOT COMPARISON
# ============================================================

labels = LABEL_ORDER
x = np.arange(len(labels))

real_vals = [real_dist[l] for l in labels]
syn_vals  = [synthetic_dist[l] for l in labels]

width = 0.35

plt.figure(figsize=(8, 4))
plt.bar(x - width/2, real_vals, width, label="Real")
plt.bar(x + width/2, syn_vals, width, label="Synthetic")

plt.xticks(x, labels, rotation=30, ha="right")
plt.ylabel("Pixel percentage (%)")
plt.title("Pixel-wise class distribution")
plt.legend()
plt.tight_layout()

plt.savefig("class_distribution_comparison.png", dpi=300)
plt.show()

# ============================================================
# COMPUTE INFARCT AREA
# ============================================================
real_areas = compute_infarct_areas(REAL_MASKS_DIR)
synthetic_areas = compute_infarct_areas(SYN_MASKS_DIR)

# Para lesion areas
min_area = min(np.min(real_areas), np.min(synthetic_areas))
max_area = max(np.max(real_areas), np.max(synthetic_areas))

n_bins = 30  # número de barras
bins = np.linspace(min_area, max_area, n_bins + 1)  # +1 porque linspace incluye los extremos

plt.figure(figsize=(8,4))
plt.hist(real_areas, bins=bins, alpha=0.6, label='Real')
plt.hist(synthetic_areas, bins=bins, alpha=0.6, label='Synthetic')
plt.xlabel('Infarct area (pixels)')
plt.ylabel('Number of masks')
plt.title('Distribution of lesion sizes')
plt.legend()
plt.savefig("infarct_area.png", dpi=300)

plt.show()

# ============================================================
# COMPUTE LESION COMPLEXITY
# ============================================================

real_complexity = compute_infarct_complexity(REAL_MASKS_DIR)
syn_complexity  = compute_infarct_complexity(SYN_MASKS_DIR)

min_complex = min(np.min(real_complexity), np.min(syn_complexity))
max_complex = max(np.max(real_complexity), np.max(syn_complexity))
n_bins = 30
bins_complex = np.linspace(min_complex, max_complex, n_bins + 1)


plt.figure(figsize=(8,4))
plt.hist(real_complexity, bins=bins_complex, alpha=0.6, label="Real")
plt.hist(syn_complexity, bins=bins_complex, alpha=0.6,label="Synthetic")
plt.xlabel("Shape complexity (perimeter / sqrt(area))")
plt.ylabel("Number of lesions")
plt.title("Infarct shape complexity distribution")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# COMPUTE VARIANCE PER FEATURE
# ============================================================
print("------VARIANCE PER FEATURE-----")
real_global = compute_mask_class_features(REAL_MASKS_DIR)
syn_global  = compute_mask_class_features(SYN_MASKS_DIR)

print("Global std per class (Real):", np.std(real_global, axis=0))
print("Global std per class (Synthetic):", np.std(syn_global, axis=0))

# ============================================================
# COMPUTE GLOBAL DIVERSITY
# ============================================================

print("-----GLOBAL DIVERSITY-----")
real_div = np.mean(pdist(real_global))
syn_div  = np.mean(pdist(syn_global))

print(f"Global diversity (Real): {real_div:.4f}")
print(f"Global diversity (Synthetic): {syn_div:.4f}")


# ============================================================
# PCA VISUALIZATION
# ============================================================
X = np.vstack([real_global, syn_global])
X = StandardScaler().fit_transform(X)

labels = ["Real"] * len(real_global) + ["Synthetic"] * len(syn_global)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(6,6))
plt.scatter(X_pca[:len(real_global),0], X_pca[:len(real_global),1],
            alpha=0.5, label="Real")
plt.scatter(X_pca[len(real_global):,0], X_pca[len(real_global):,1],
            alpha=0.5, label="Synthetic")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Global mask diversity (class composition)")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Wasserstein distance lesion area
# ============================================================
print("-----WASSERSTEIN DISTANCE LESTION AREA-----")
w_area = wasserstein_distance(real_areas, synthetic_areas)
print(f"Wasserstein distance (lesion area): {w_area:.2f} pixels")


# ============================================================
# Wasserstein distance lesion complexity
# ============================================================
print("-----WASSERSTEIN DISTANCE LESTION COMPLEXITY-----")

w_complexity = wasserstein_distance(real_complexity, syn_complexity)
print(f"Wasserstein distance (lesion complexity): {w_complexity:.4f}")

# ============================================================
# COUNT NUMBER OF LESIONS
# ============================================================

real_counts = count_lesions(REAL_MASKS_DIR)
synthetic_counts = count_lesions(SYN_MASKS_DIR)

print("/n ------NUMBER OF LESIONS PER IMAGE------")
print(f"Real dataset: mean={np.mean(real_counts):.2f}, total={np.sum(real_counts)}, max={np.max(real_counts)}, min={np.min(real_counts)}")
print(f"Synthetic dataset: mean={np.mean(synthetic_counts):.2f}, total={np.sum(synthetic_counts)}, max={np.max(synthetic_counts)}, min={np.min(synthetic_counts)}")

# Histograma de número de lesiones por máscara
plt.figure(figsize=(8,4))
bins_counts = np.arange(0, max(np.max(real_counts), np.max(synthetic_counts)) + 2) - 0.5
plt.hist(real_counts, bins=bins_counts, alpha=0.6, label="Real")
plt.hist(synthetic_counts, bins=bins_counts, alpha=0.6, label="Synthetic")
plt.xlabel("Number of lesions per mask")
plt.ylabel("Number of images")
plt.title("Lesion count per mask")
plt.legend()
plt.tight_layout()
plt.show()




