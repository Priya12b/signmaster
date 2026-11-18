import os
import io
import urllib.request
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

DATASET_URL = "https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03/resolve/main/data/train-00000-of-00001.parquet"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PARQUET_FILE = os.path.join(DATA_DIR, "asl_dataset.parquet")

LABEL_MAPPING = {i: chr(65 + i) for i in range(26)}  # 0->A, 1->B, ..., 25->Z

def download_file(url, target_path):
    print(f"Downloading dataset from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp, open(target_path, "wb") as f:
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                print(f"\rDownloaded {downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB", end="", flush=True)
    print("\nDownload complete!")

def prepare_asl_dataset(target_total=5000, random_state=42):
    """
    Downloads and prepares an ASL dataset of exactly `target_total` images across all 26 alphabet classes (A-Z).
    Splits into train (70%), val (15%), test (15%) splits avoiding data leakage.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")

    # If dataset already extracted and contains images, verify counts
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        total_existing = sum(len(files) for _, _, files in os.walk(DATA_DIR) if not _ .endswith(".parquet"))
        if total_existing >= target_total:
            print(f"Dataset already prepared with {total_existing} images in {DATA_DIR}.")
            return train_dir, val_dir, test_dir

    if not os.path.exists(PARQUET_FILE):
        download_file(DATASET_URL, PARQUET_FILE)

    print("Reading parquet file...")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Parquet loaded with {len(df)} total rows. Columns: {df.columns.tolist()}")

    # Ensure label values are 0..25
    df = df[df["label"].between(0, 25)].copy()

    # Calculate per-class quota to reach target_total
    num_classes = 26
    per_class_quota = target_total // num_classes
    remainder = target_total % num_classes

    sampled_dfs = []
    for label_id in range(num_classes):
        class_df = df[df["label"] == label_id]
        n_samples = per_class_quota + (1 if label_id < remainder else 0)
        n_samples = min(n_samples, len(class_df))
        sampled_dfs.append(class_df.sample(n=n_samples, random_state=random_state))

    dataset_df = pd.concat(sampled_dfs).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    print(f"Selected {len(dataset_df)} balanced samples across 26 classes.")

    # Split: 70% Train, 15% Val, 15% Test with stratification
    train_df, temp_df = train_test_split(
        dataset_df, test_size=0.30, stratify=dataset_df["label"], random_state=random_state
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=random_state
    )

    print(f"Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Create directory structure
    for split_dir, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_path = os.path.join(DATA_DIR, split_dir)
        for label_name in LABEL_MAPPING.values():
            os.makedirs(os.path.join(split_path, label_name), exist_ok=True)

        print(f"Saving {len(split_df)} images to {split_path}...")
        for idx, row in split_df.iterrows():
            label_char = LABEL_MAPPING[int(row["label"])]
            img_dict = row["image"]

            # Handle PIL image or raw bytes
            if isinstance(img_dict, dict) and "bytes" in img_dict and img_dict["bytes"] is not None:
                img = Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")
            elif isinstance(img_dict, Image.Image):
                img = img_dict.convert("RGB")
            else:
                continue

            save_name = f"{label_char}_{idx}.jpg"
            img.save(os.path.join(split_path, label_char, save_name), "JPEG", quality=95)

    print("Dataset preparation complete!")
    return train_dir, val_dir, test_dir

if __name__ == "__main__":
    prepare_asl_dataset()
