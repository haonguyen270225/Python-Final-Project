
from pathlib import Path
import random
import shutil

# =========================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================

# Thư mục gốc của dự án Python-Final-Project
BASE_DIR = Path(__file__).resolve().parent.parent

# Thư mục dữ liệu nguồn
SOURCE_DIR = BASE_DIR / "asset_demo_01"

# Thư mục dữ liệu đầu ra
OUTPUT_DIR = BASE_DIR / "yolo_currency_dataset"

CURRENCY_LABELS = {
    "AmericanDollar_Rename": "USD",
    "ChineseYuan_Rename": "CNY",
    "VietnameseDong_Rename": "VND",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42


def collect_images():
    """Thu thập ảnh và gán nhãn theo loại tiền."""
    samples = []

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {SOURCE_DIR}"
        )

    for folder_name, label in CURRENCY_LABELS.items():
        currency_dir = SOURCE_DIR / folder_name

        if not currency_dir.exists():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục: {currency_dir}"
            )

        for image_path in currency_dir.rglob("*"):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                samples.append((image_path, label))

    if not samples:
        raise RuntimeError("Không tìm thấy ảnh trong thư mục dữ liệu.")

    return samples


def main():
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
        raise ValueError("Tổng các tỷ lệ phải bằng 1.")

    random.seed(RANDOM_SEED)
    samples = collect_images()
    random.shuffle(samples)

    total = len(samples)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    # Tránh ghi đè dữ liệu cũ ngoài ý muốn.
    if OUTPUT_DIR.exists():
        confirm = input(
            f"{OUTPUT_DIR} đã tồn tại. Nhập DELETE để tạo lại: "
        )
        if confirm != "DELETE":
            print("Đã hủy. Không thay đổi dữ liệu.")
            return
        shutil.rmtree(OUTPUT_DIR)

    counts = {}

    for split_name, split_samples in splits.items():
        counts[split_name] = {}

        for index, (source_path, label) in enumerate(split_samples):
            target_dir = OUTPUT_DIR / split_name / label
            target_dir.mkdir(parents=True, exist_ok=True)

            target_name = (
                f"{label}_{index:06d}{source_path.suffix.lower()}"
            )
            target_path = target_dir / target_name

            shutil.copy2(source_path, target_path)

            counts[split_name][label] = (
                counts[split_name].get(label, 0) + 1
            )

    print("\nHoàn tất chuẩn bị dữ liệu.")
    print(f"Tổng số ảnh: {total}")
    print(f"Thư mục đầu ra: {OUTPUT_DIR}")

    for split_name, label_counts in counts.items():
        print(f"\n{split_name.upper()}:")
        for label in CURRENCY_LABELS.values():
            print(f"  {label}: {label_counts.get(label, 0)} ảnh")


if __name__ == "__main__":
    main()
