
from pathlib import Path

from ultralytics import YOLO


# ==================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==================================================

# huan_luyen.py nằm trong thư mục model_YoLo
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "yolo_currency_dataset"

PROJECT_DIR = BASE_DIR / "runs" / "classify"
MODEL_NAME = "currency_yolo11n_cls"


# ==================================================
# 2. KIỂM TRA DỮ LIỆU
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

print("=" * 60)
print("THÔNG TIN HUẤN LUYỆN YOLO CLASSIFICATION")
print("=" * 60)

print(f"Thư mục dữ liệu: {DATA_DIR}")
print(f"Thư mục lưu kết quả: {PROJECT_DIR}")

for split in ["train", "val", "test"]:
    print(f"\n{split.upper()}:")

    for class_dir in sorted((DATA_DIR / split).iterdir()):
        if class_dir.is_dir():
            image_count = sum(
                1 for file in class_dir.iterdir()
                if file.is_file()
                and file.suffix.lower() in {
                    ".jpg", ".jpeg", ".png",
                    ".bmp", ".webp", ".tif", ".tiff"
                }
            )

            print(f"  {class_dir.name}: {image_count} ảnh")


# ==================================================
# 3. KHỞI TẠO MÔ HÌNH
# ==================================================

# Tự tải trọng số pretrained nếu chưa có trong máy.
model = YOLO("yolo11n-cls.pt")


# ==================================================
# 4. HUẤN LUYỆN MÔ HÌNH
# ==================================================

results = model.train(
    data=str(DATA_DIR),

    epochs=50,
    imgsz=224,
    batch=16,

    # PyTorch/Ultralytics tự chọn thiết bị phù hợp.
    device="",

    # Giảm vấn đề multiprocessing khi chạy trên Windows.
    workers=0,

    # Cố định seed để thuận tiện tái lập thử nghiệm.
    seed=42,

    project=str(PROJECT_DIR),
    name=MODEL_NAME,

    # Cho phép tiếp tục huấn luyện hoặc thử nghiệm sau này.
    exist_ok=True,

    # Hiển thị thông tin huấn luyện.
    verbose=True,
)


# ==================================================
# 5. THÔNG BÁO KẾT QUẢ
# ==================================================

save_dir = Path(results.save_dir)

print("\n" + "=" * 60)
print("HUẤN LUYỆN HOÀN TẤT")
print("=" * 60)

print(f"Thư mục kết quả: {save_dir}")
print(f"Mô hình tốt nhất: {save_dir / 'weights' / 'best.pt'}")
print(f"Mô hình cuối cùng: {save_dir / 'weights' / 'last.pt'}")
