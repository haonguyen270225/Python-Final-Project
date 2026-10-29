
# from pathlib import Path
# import random
# import shutil

# # =========================
# # 1. CẤU HÌNH ĐƯỜNG DẪN
# # =========================

# # Thư mục gốc của dự án Python-Final-Project
# BASE_DIR = Path(__file__).resolve().parent.parent

# # Thư mục dữ liệu nguồn
# SOURCE_DIR = BASE_DIR / "asset_demo_01"

# # Thư mục dữ liệu đầu ra
# OUTPUT_DIR = BASE_DIR / "yolo_currency_dataset"

# CURRENCY_LABELS = {
#     "AmericanDollar_Rename": "USD",
#     "ChineseYuan_Rename": "CNY",
#     "VietnameseDong_Rename": "VND",
# }

# IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# TRAIN_RATIO = 0.70
# VAL_RATIO = 0.20
# TEST_RATIO = 0.10

# RANDOM_SEED = 42


# def collect_images():
#     """Thu thập ảnh và gán nhãn theo loại tiền."""
#     samples = []

#     if not SOURCE_DIR.exists():
#         raise FileNotFoundError(
#             f"Không tìm thấy thư mục dữ liệu: {SOURCE_DIR}"
#         )

#     for folder_name, label in CURRENCY_LABELS.items():
#         currency_dir = SOURCE_DIR / folder_name

#         if not currency_dir.exists():
#             raise FileNotFoundError(
#                 f"Không tìm thấy thư mục: {currency_dir}"
#             )

#         for image_path in currency_dir.rglob("*"):
#             if (
#                 image_path.is_file()
#                 and image_path.suffix.lower() in IMAGE_EXTENSIONS
#             ):
#                 samples.append((image_path, label))

#     if not samples:
#         raise RuntimeError("Không tìm thấy ảnh trong thư mục dữ liệu.")

#     return samples


# def main():
#     if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
#         raise ValueError("Tổng các tỷ lệ phải bằng 1.")

#     random.seed(RANDOM_SEED)
#     samples = collect_images()
#     random.shuffle(samples)

#     total = len(samples)
#     train_end = int(total * TRAIN_RATIO)
#     val_end = train_end + int(total * VAL_RATIO)

#     splits = {
#         "train": samples[:train_end],
#         "val": samples[train_end:val_end],
#         "test": samples[val_end:],
#     }

#     # Tránh ghi đè dữ liệu cũ ngoài ý muốn.
#     if OUTPUT_DIR.exists():
#         confirm = input(
#             f"{OUTPUT_DIR} đã tồn tại. Nhập DELETE để tạo lại: "
#         )
#         if confirm != "DELETE":
#             print("Đã hủy. Không thay đổi dữ liệu.")
#             return
#         shutil.rmtree(OUTPUT_DIR)

#     counts = {}

#     for split_name, split_samples in splits.items():
#         counts[split_name] = {}

#         for index, (source_path, label) in enumerate(split_samples):
#             target_dir = OUTPUT_DIR / split_name / label
#             target_dir.mkdir(parents=True, exist_ok=True)

#             target_name = (
#                 f"{label}_{index:06d}{source_path.suffix.lower()}"
#             )
#             target_path = target_dir / target_name

#             shutil.copy2(source_path, target_path)

#             counts[split_name][label] = (
#                 counts[split_name].get(label, 0) + 1
#             )

#     print("\nHoàn tất chuẩn bị dữ liệu.")
#     print(f"Tổng số ảnh: {total}")
#     print(f"Thư mục đầu ra: {OUTPUT_DIR}")

#     for split_name, label_counts in counts.items():
#         print(f"\n{split_name.upper()}:")
#         for label in CURRENCY_LABELS.values():
#             print(f"  {label}: {label_counts.get(label, 0)} ảnh")


# if __name__ == "__main__":
#     main()


from pathlib import Path
from collections import defaultdict
import random
import re
import shutil


# ==================================================
# 1. CẤU HÌNH
# ==================================================

# tao_dataset.py nằm trong thư mục model_YoLo.
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset ảnh gốc.
ASSET_DIR = BASE_DIR / "asset"

# Dataset đầu ra dành cho YOLO Classification.
OUTPUT_DIR = BASE_DIR / "yolo_currency_dataset"

# Tỷ lệ chia dữ liệu.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Cố định seed để có thể tái lập cách chia dữ liệu.
SEED = 42

# True: xóa dataset đầu ra cũ trước khi tạo lại.
# Chỉ bật khi OUTPUT_DIR đúng là thư mục dataset được tạo tự động.
RESET_OUTPUT = True

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

CURRENCY_CODES = {
    "AmericanDollar": "USD",
    "ChineseYuan": "CNY",
    "VietnameseDong": "VND",
}

SPLITS = {
    "train": TRAIN_RATIO,
    "val": VAL_RATIO,
    "test": TEST_RATIO,
}


# ==================================================
# 2. KIỂM TRA CẤU HÌNH
# ==================================================

if not ASSET_DIR.is_dir():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục ảnh nguồn: {ASSET_DIR}"
    )

if abs(sum(SPLITS.values()) - 1.0) > 1e-9:
    raise ValueError(
        "Tổng tỷ lệ train, val, test phải bằng 1."
    )

for currency_folder in CURRENCY_CODES:
    if not (ASSET_DIR / currency_folder).is_dir():
        raise FileNotFoundError(
            "Không tìm thấy thư mục tiền tệ: "
            f"{ASSET_DIR / currency_folder}"
        )

print("=" * 65)
print("TẠO DATASET YOLO - PHÂN LOẠI TIỀN VÀ MỆNH GIÁ")
print("=" * 65)

print(f"Thư mục nguồn: {ASSET_DIR}")
print(f"Thư mục đích : {OUTPUT_DIR}")


# ==================================================
# 3. ĐỌC CÁC THƯ MỤC MỆNH GIÁ
# ==================================================

# Mỗi phần tử có dạng:
# (tên lớp YOLO, đường dẫn thư mục ảnh)
class_sources = []

for currency_folder, currency_code in CURRENCY_CODES.items():

    currency_dir = ASSET_DIR / currency_folder

    for denomination_dir in sorted(currency_dir.iterdir()):

        if not denomination_dir.is_dir():
            continue

        # Bỏ qua thư mục không có mệnh giá,
        # ví dụ AmericanDollar/Test Set.
        match = re.search(
            r"\d+",
            denomination_dir.name
        )

        if not match:
            print(
                f"Bỏ qua thư mục không phải mệnh giá: "
                f"{denomination_dir.name}"
            )
            continue

        # Chuẩn hóa 000200 thành 200, 050000 thành 50000.
        denomination = int(match.group())

        # Ví dụ: USD_1, CNY_10, VND_50000.
        class_name = f"{currency_code}_{denomination}"

        class_sources.append(
            (class_name, denomination_dir)
        )

if not class_sources:
    raise ValueError(
        "Không tìm thấy thư mục mệnh giá hợp lệ."
    )

# Không cho phép hai thư mục nguồn tạo ra cùng một lớp.
class_names = [name for name, _ in class_sources]

if len(class_names) != len(set(class_names)):
    raise ValueError(
        "Có các thư mục nguồn trùng tên lớp sau khi chuẩn hóa. "
        "Hãy kiểm tra lại cấu trúc asset."
    )


# ==================================================
# 4. THU THẬP ẢNH
# ==================================================

rng = random.Random(SEED)

# Danh sách ảnh theo từng lớp.
images_by_class = {}

for class_name, source_dir in class_sources:

    images = sorted(
        file
        for file in source_dir.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(
            f"Cảnh báo: lớp {class_name} không có ảnh."
        )
        continue

    # Trộn ảnh có seed cố định.
    rng.shuffle(images)

    images_by_class[class_name] = images

    print(
        f"{class_name:<15} | "
        f"{len(images):>5} ảnh | "
        f"{source_dir.relative_to(ASSET_DIR)}"
    )

if not images_by_class:
    raise ValueError(
        "Không tìm thấy ảnh hợp lệ trong asset."
    )


# ==================================================
# 5. TẠO THƯ MỤC DATASET
# ==================================================

if OUTPUT_DIR.exists():

    if not RESET_OUTPUT:
        raise FileExistsError(
            f"Dataset đã tồn tại: {OUTPUT_DIR}\n"
            "Hãy đổi OUTPUT_DIR hoặc bật RESET_OUTPUT=True "
            "nếu muốn tạo lại."
        )

    # Chỉ xóa thư mục đầu ra, không xóa asset.
    shutil.rmtree(OUTPUT_DIR)

for split in SPLITS:
    for class_name in images_by_class:
        (OUTPUT_DIR / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )


# ==================================================
# 6. CHIA VÀ SAO CHÉP ẢNH
# ==================================================

statistics = defaultdict(int)
total_copied = 0

for class_name, images in images_by_class.items():

    image_count = len(images)

    # Tính số ảnh cho train và val.
    train_count = int(image_count * TRAIN_RATIO)
    val_count = int(image_count * VAL_RATIO)

    # Phần còn lại dành cho test.
    test_count = image_count - train_count - val_count

    # Không thể đảm bảo cả 3 tập đều có ảnh
    # nếu một lớp có ít hơn 3 ảnh.
    if image_count >= 3:
        train_count = max(1, train_count)
        val_count = max(1, val_count)
        test_count = max(1, test_count)

        # Điều chỉnh lại để tổng số ảnh không vượt quá số nguồn.
        while train_count + val_count + test_count > image_count:
            if train_count > val_count and train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                test_count -= 1

        while train_count + val_count + test_count < image_count:
            test_count += 1

    train_end = train_count
    val_end = train_count + val_count

    split_images = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split, split_images_list in split_images.items():

        destination_dir = OUTPUT_DIR / split / class_name

        for image_path in split_images_list:

            destination_path = destination_dir / image_path.name

            # Tránh ghi đè nếu tên ảnh bị trùng.
            counter = 1

            while destination_path.exists():
                destination_path = destination_dir / (
                    f"{image_path.stem}_{counter}"
                    f"{image_path.suffix}"
                )
                counter += 1

            shutil.copy2(
                image_path,
                destination_path
            )

            statistics[(split, class_name)] += 1
            total_copied += 1


# ==================================================
# 7. THỐNG KÊ KẾT QUẢ
# ==================================================

print("\n" + "=" * 65)
print("KẾT QUẢ TẠO DATASET")
print("=" * 65)

for split in SPLITS:

    print(f"\n[{split.upper()}]")

    split_total = 0

    for class_name in sorted(images_by_class):

        count = statistics[(split, class_name)]

        print(f"  {class_name:<15}: {count:>5} ảnh")
        split_total += count

    print(f"  Tổng: {split_total} ảnh")

print("\n" + "=" * 65)
print(f"Số lớp       : {len(images_by_class)}")
print(f"Tổng ảnh copy: {total_copied}")
print(f"Dataset      : {OUTPUT_DIR}")
print("=" * 65)

print("\nĐÃ TẠO DATASET THÀNH CÔNG!")
print("Bạn có thể chạy huan_luyen.py để bắt đầu huấn luyện.")
