from pathlib import Path
from tempfile import gettempdir
import numpy as np
from skimage import io, color, transform, feature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from typing import Literal
from time import time

RESIZE_SHAPE = (224, 224)
PATH_BRAIN_TUMOR_1D = "./data/brain_tumor_1d.npy"
PATH_BRAIN_TUMOR_2D = "./data/brain_tumor_2d.npy"

# Class mapping for brain tumor types
CLASS_MAPPING = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}

TEMP_FOLDER = f"{gettempdir()}/brain-tumor-detection/"


def pre_import_hook():
    Path(TEMP_FOLDER).mkdir(exist_ok=True)

    if Path(PATH_BRAIN_TUMOR_1D).is_file() and Path(PATH_BRAIN_TUMOR_2D).is_file():
        return
    
    print("START PROCESSING...")

    image_pixels_1d = ([], [])  # (training, testing)
    image_pixels_2d = ({"image": [], "label": []}, {"image": [], "label": []})

    # Process Training data (is_test=0)
    for tumor_type, label in CLASS_MAPPING.items():
        print(tumor_type)
        train_folder = Path(f"./data/Training/{tumor_type}")
        if train_folder.exists():
            image_paths = list(train_folder.glob("*.jpg"))
            for image_path in tqdm(image_paths, desc=f"Training - {tumor_type}"):
                image_file = io.imread(str(image_path))
                if len(image_file.shape) == 3:
                    image_rgb = image_file[:, :, :3]
                    image_file = color.rgb2gray(image_rgb)
                image_pixel = transform.resize(image_file, RESIZE_SHAPE, preserve_range=True)
                
                image_pixels_1d[0].append(
                    np.concatenate([image_pixel.flatten(), [label]])
                )
                image_pixels_2d[0]["image"].append(np.stack((image_pixel,) * 3, axis=-1))
                image_pixels_2d[0]["label"].append(label)

    # Process Testing data (is_test=1)
    for tumor_type, label in CLASS_MAPPING.items():
        test_folder = Path(f"./data/Testing/{tumor_type}")
        if test_folder.exists():
            image_paths = list(test_folder.glob("*.jpg"))
            for image_path in tqdm(image_paths, desc=f"Testing - {tumor_type}"):
                image_file = io.imread(str(image_path))
                if len(image_file.shape) == 3:
                    image_file = color.rgb2gray(image_file)
                image_pixel = transform.resize(image_file, RESIZE_SHAPE, preserve_range=True)
                
                image_pixels_1d[1].append(
                    np.concatenate([image_pixel.flatten(), [label]])
                )
                image_pixels_2d[1]["image"].append(np.stack((image_pixel,) * 3, axis=-1))
                image_pixels_2d[1]["label"].append(label)

    # Convert to numpy arrays
    image_pixels_2d[0]["image"] = np.array(image_pixels_2d[0]["image"]).astype(np.uint8)
    image_pixels_2d[0]["label"] = np.array(image_pixels_2d[0]["label"])
    image_pixels_2d[1]["image"] = np.array(image_pixels_2d[1]["image"]).astype(np.uint8)
    image_pixels_2d[1]["label"] = np.array(image_pixels_2d[1]["label"])
    
    # Calculate actual train/test split sizes
    train_size = len(image_pixels_1d[0])
    test_size = len(image_pixels_1d[1])
    print(f"Training samples: {train_size}, Testing samples: {test_size}")
    
    np.save(PATH_BRAIN_TUMOR_1D, image_pixels_1d[0] + image_pixels_1d[1])
    np.save(PATH_BRAIN_TUMOR_2D, image_pixels_2d, allow_pickle=True)


def resolve_filename(
    include_raw: bool = True,
    include_hog: bool = False,
    include_lbp: bool = False,
    pca_mode: Literal["global", "local", "none"] = "none",
) -> str:
    tags = ["brain_tumor"]
    if include_raw:
        tags.append("raw")
    if include_hog:
        tags.append("hog")
    if include_lbp:
        tags.append("lbp")
    if pca_mode == "global":
        tags.append("pcag")
    elif pca_mode == "local":
        tags.append("pcal")
    filename = "_".join(tags) + ".npy"

    return filename


def load_brain_tumor_1d(
    include_raw: bool = True,
    include_hog: bool = False,
    include_lbp: bool = False,
    pca_mode: Literal["global", "local", "none"] = "none",
) -> tuple[np.ndarray, np.ndarray]:
    # Apply Argument Value Check
    valid_pca_modes = {"global", "local", "none"}
    if pca_mode not in valid_pca_modes:
        raise ValueError(
            f"{repr(pca_mode)} is not a valid PCA mode. Pick between {valid_pca_modes}."
        )
    if pca_mode == "global" and include_raw:
        raise ValueError("Cannot include_raw when pca_mode is set to `global`.")
    if pca_mode == "local" and not (include_raw + include_hog + include_lbp):
        raise ValueError("Include at least one of raw, hog, or lbp to apply local PCA.")
    if pca_mode == "none" and not (include_raw + include_hog + include_lbp):
        raise ValueError("No data is loaded.")

    temp_filename = resolve_filename(include_raw, include_hog, include_lbp, pca_mode)
    temp_filepath = f"{TEMP_FOLDER}{temp_filename}"

    IMAGE_PIXELS = RESIZE_SHAPE[0] * RESIZE_SHAPE[1]

    # Load the combined dataset to determine actual split
    src_file = np.load(PATH_BRAIN_TUMOR_1D)

    # Calculate train size dynamically from the saved data
    # We need to count how many samples were in training vs testing
    # This is done by loading the 2D file which has the split info
    data_2d = np.load(PATH_BRAIN_TUMOR_2D, allow_pickle=True)
    # Check if it's wrapped in an array
    if isinstance(data_2d, np.ndarray) and data_2d.shape == ():
        data_2d = data_2d.item()
    TRAIN_SIZE = len(data_2d[0]["label"])

    # Reuse Previously Computed Dataset (skip caching when sampling) // and sample_fraction == 1.0
    if Path(temp_filepath).is_file():
        return np.split(np.load(temp_filepath), [TRAIN_SIZE])

    src_features, src_labels = np.split(src_file, [IMAGE_PIXELS], axis=1)
    target_features = []

    for src_feature in tqdm(src_features, desc="Processing features"):
        target_feature = []

        if include_raw:
            target_feature.append(src_feature)

        if include_hog or include_lbp:
            image_2d = src_feature.reshape(RESIZE_SHAPE)
        if include_hog:
            target_feature.append(
                feature.hog(
                    image_2d,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys",
                    transform_sqrt=True,
                    feature_vector=True,
                    visualize=False,
                )
            )
        if include_lbp:
            image_lbp = feature.local_binary_pattern(
                image_2d.astype(np.uint8), 24, 3, "uniform"
            )
            image_hist = np.histogram(
                image_lbp.ravel(), bins=26, range=(0, 26), density=True
            )[0].astype("float")
            image_hist /= image_hist.sum() + 1e-6
            target_feature.append(image_hist)
        if len(target_feature) > 0:
            target_features.append(np.concatenate(target_feature))
        else:
            target_features.append([])

    sc = StandardScaler()
    if pca_mode == "global":
        pca = PCA(n_components=2)
        print("Calculating PCA...")
        target_dataset = np.concatenate(
            [
                sc.fit_transform(
                    np.concatenate([target_features, pca.fit_transform(src_features)], axis=1)
                ),
                src_labels,
            ],
            axis=1,
        )
    elif pca_mode == "local":
        pca = PCA(n_components=2)
        print("Calculating PCA...")
        target_dataset = np.concatenate(
            [sc.fit_transform(pca.fit_transform(target_features)), src_labels], axis=1
        )
    else:
        target_dataset = np.concatenate(
            [sc.fit_transform(target_features), src_labels], axis=1
        )
    np.save(temp_filepath, target_dataset)
    return np.split(target_dataset, [TRAIN_SIZE])

PARAMETER_PERMUTATION_1D = [
  [True, True, True, "none"],
  [False, True, True, "none"],
  [True, False, True, "none"],
  [True, True, False, "none"],
  [False, False, True, "none"],
  [False, True, False, "none"],
  [True, False, False, "none"],
  [True, True, True, "local"],
  [False, True, True, "local"],
  [True, False, True, "local"],
  [True, True, False, "local"],
  [False, False, True, "local"],
  [True, False, False, "local"],
  [False, True, False, "local"],
  [False, True, True, "global"],
  [False, False, True, "global"],
  [False, True, False, "global"],
  [False, False, False, "global"],
]

def load_brain_tumor_2d():
    train, test = np.load(PATH_BRAIN_TUMOR_2D, allow_pickle=True)
    return train, test

pre_import_hook()


# import os
# from tempfile import gettempdir

# TEMP_FOLDER = f"{gettempdir()}/brain-tumor-detection/"
# # Delete all cached files
# for f in Path(TEMP_FOLDER).glob("*.npy"):
#     os.remove(f)
#     print(f"Deleted: {f}")
