
# from pathlib import Path

# from ultralytics import YOLO


# # ==================================================
# # 1. CẤU HÌNH ĐƯỜNG DẪN
# # ==================================================

# # huan_luyen.py nằm trong thư mục model_YoLo
# BASE_DIR = Path(__file__).resolve().parent.parent

# DATA_DIR = BASE_DIR / "yolo_currency_dataset"

# PROJECT_DIR = BASE_DIR / "runs" / "classify"
# MODEL_NAME = "currency_yolo11n_cls"


# # ==================================================
# # 2. KIỂM TRA DỮ LIỆU
# # ==================================================

# required_dirs = [
#     DATA_DIR / "train",
#     DATA_DIR / "val",
#     DATA_DIR / "test",
# ]

# for directory in required_dirs:
#     if not directory.is_dir():
#         raise FileNotFoundError(
#             f"Không tìm thấy thư mục dữ liệu: {directory}"
#         )

# print("=" * 60)
# print("THÔNG TIN HUẤN LUYỆN YOLO CLASSIFICATION")
# print("=" * 60)

# print(f"Thư mục dữ liệu: {DATA_DIR}")
# print(f"Thư mục lưu kết quả: {PROJECT_DIR}")

# for split in ["train", "val", "test"]:
#     print(f"\n{split.upper()}:")

#     for class_dir in sorted((DATA_DIR / split).iterdir()):
#         if class_dir.is_dir():
#             image_count = sum(
#                 1 for file in class_dir.iterdir()
#                 if file.is_file()
#                 and file.suffix.lower() in {
#                     ".jpg", ".jpeg", ".png",
#                     ".bmp", ".webp", ".tif", ".tiff"
#                 }
#             )

#             print(f"  {class_dir.name}: {image_count} ảnh")


# # ==================================================
# # 3. KHỞI TẠO MÔ HÌNH
# # ==================================================

# # Tự tải trọng số pretrained nếu chưa có trong máy.
# model = YOLO("yolo11n-cls.pt")


# # ==================================================
# # 4. HUẤN LUYỆN MÔ HÌNH
# # ==================================================

# results = model.train(
#     data=str(DATA_DIR),

#     epochs=50,
#     imgsz=224,
#     batch=16,

#     # PyTorch/Ultralytics tự chọn thiết bị phù hợp.
#     device="",

#     # Giảm vấn đề multiprocessing khi chạy trên Windows.
#     workers=0,

#     # Cố định seed để thuận tiện tái lập thử nghiệm.
#     seed=42,

#     project=str(PROJECT_DIR),
#     name=MODEL_NAME,

#     # Cho phép tiếp tục huấn luyện hoặc thử nghiệm sau này.
#     exist_ok=True,

#     # Hiển thị thông tin huấn luyện.
#     verbose=True,
# )


# # ==================================================
# # 5. THÔNG BÁO KẾT QUẢ
# # ==================================================

# save_dir = Path(results.save_dir)

# print("\n" + "=" * 60)
# print("HUẤN LUYỆN HOÀN TẤT")
# print("=" * 60)

# print(f"Thư mục kết quả: {save_dir}")
# print(f"Mô hình tốt nhất: {save_dir / 'weights' / 'best.pt'}")
# print(f"Mô hình cuối cùng: {save_dir / 'weights' / 'last.pt'}")


# Tùy chỉnh mã huấn luyện cả loại tiền và mệnh giá !

from pathlib import Path

from ultralytics import YOLO


# ==================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==================================================

# huan_luyen.py nằm trong thư mục model_YoLo
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "asset_of_yolo"

PROJECT_DIR = BASE_DIR / "ket_qua_yolo"
MODEL_NAME = "currency_yolo11n_cls"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


# ==================================================
# 2. KIỂM TRA DATASET
# ==================================================

required_dirs = [
    DATA_DIR / "train",
    DATA_DIR / "val",
    DATA_DIR / "test",
]

for directory in required_dirs:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {directory}"
        )

# Tên lớp được lấy từ tên thư mục con trong train.
class_names = sorted(
    directory.name
    for directory in (DATA_DIR / "train").iterdir()
    if directory.is_dir()
)

if not class_names:
    raise ValueError(
        "Không tìm thấy lớp dữ liệu trong thư mục train."
    )

# Kiểm tra train, val, test có cùng các lớp.
for split in ["train", "val", "test"]:
    split_dir = DATA_DIR / split

    split_classes = {
        directory.name
        for directory in split_dir.iterdir()
        if directory.is_dir()
    }

    missing_classes = set(class_names) - split_classes

    if missing_classes:
        raise ValueError(
            f"{split} thiếu các lớp: "
            f"{sorted(missing_classes)}"
        )


def count_images(directory):
    """Đếm ảnh trực tiếp trong thư mục của một lớp."""
    return sum(
        1
        for file in directory.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


print("=" * 65)
print("HUẤN LUYỆN YOLO11 - NHẬN DIỆN LOẠI TIỀN VÀ MỆNH GIÁ")
print("=" * 65)

print(f"Dataset: {DATA_DIR}")
print(f"Output:  {PROJECT_DIR}")
print(f"Số lớp:  {len(class_names)}")
print(f"Các lớp: {', '.join(class_names)}")

for split in ["train", "val", "test"]:
    print(f"\n[{split.upper()}]")

    total_images = 0

    for class_name in class_names:
        class_dir = DATA_DIR / split / class_name
        image_count = count_images(class_dir)

        total_images += image_count

        print(f"  {class_name:<20}: {image_count:>5} ảnh")

    print(f"  Tổng số ảnh: {total_images}")


# ==================================================
# 3. KHỞI TẠO MÔ HÌNH
# ==================================================

# YOLO11n Classification với trọng số pretrained.
model = YOLO("yolo11n-cls.pt")


# ==================================================
# 4. HUẤN LUYỆN
# ==================================================

results = model.train(
    data=str(DATA_DIR),

    epochs=50,
    imgsz=224,
    batch=16,

    # Để Ultralytics tự chọn thiết bị phù hợp.
    device="",

    # Phù hợp môi trường Windows.
    workers=0,

    # Tái lập thử nghiệm.
    seed=42,

    # Tự dừng nếu kết quả không cải thiện.
    patience=10,

    # Lưu biểu đồ và kết quả huấn luyện.
    plots=True,
    save=True,
    val=True,

    project=str(PROJECT_DIR),
    name=MODEL_NAME,
    exist_ok=True,

    verbose=True,
)


# ==================================================
# 5. ĐÁNH GIÁ TRÊN TẬP TEST
# ==================================================

save_dir = Path(results.save_dir)
best_model_path = save_dir / "weights" / "best.pt"
last_model_path = save_dir / "weights" / "last.pt"

if not best_model_path.is_file():
    raise FileNotFoundError(
        f"Không tìm thấy mô hình tốt nhất: {best_model_path}"
    )

print("\n" + "=" * 65)
print("ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST")
print("=" * 65)

best_model = YOLO(str(best_model_path))

test_results = best_model.val(
    data=str(DATA_DIR),
    split="test",
    imgsz=224,
    batch=16,
    workers=0,
    plots=True,
    verbose=True,
)


# ==================================================
# 6. THÔNG TIN KẾT QUẢ
# ==================================================

print("\n" + "=" * 65)
print("HOÀN TẤT HUẤN LUYỆN VÀ ĐÁNH GIÁ")
print("=" * 65)

print(f"Thư mục kết quả : {save_dir}")
print(f"Mô hình tốt nhất: {best_model_path}")
print(f"Mô hình cuối    : {last_model_path}")

print("\nCác lớp nhận diện:")
for index, name in enumerate(best_model.names.values()):
    print(f"  {index}: {name}")

print("\nCó thể dùng best.pt để dự đoán loại tiền và mệnh giá.")
