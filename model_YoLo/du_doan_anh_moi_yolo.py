# 1. Chạy trogn PowerShell : cd D:\Python-Final-Project
# 2 . Chạy trong PowerShell : .\.venv\Scripts\python.exe .\model_YoLo\du_doan_anh_moi_yolo.py

from pathlib import Path
import csv
from ultralytics import YOLO

# ==================================================
# 1. CẤU HÌNH
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Mô hình tốt nhất sau khi huấn luyện
MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "classify"
    / "currency_yolo11n_cls"
    / "weights"
    / "best.pt"
)

# Thư mục chứa ảnh cần dự đoán
IMAGE_DIR = BASE_DIR / "anh_kiem_tra"

# Thư mục lưu ảnh kết quả
OUTPUT_DIR = BASE_DIR / "ket_qua_YoLo"

# File CSV lưu kết quả dự đoán
CSV_PATH = Path(__file__).resolve().parent / "ket_qua_du_doan.csv"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}


# ==================================================
# 2. KIỂM TRA ĐƯỜNG DẪN
# ==================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Không tìm thấy mô hình: {MODEL_PATH}\n"
        "Hãy huấn luyện xong và kiểm tra lại đường dẫn best.pt."
    )

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục ảnh: {IMAGE_DIR}"
    )

image_paths = sorted(
    path for path in IMAGE_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_paths:
    raise ValueError(
        f"Không tìm thấy ảnh trực tiếp trong: {IMAGE_DIR}"
    )


# ==================================================
# 3. NẠP MÔ HÌNH
# ==================================================

model = YOLO(str(MODEL_PATH))

print("=" * 60)
print("DỰ ĐOÁN LOẠI TIỀN TỆ BẰNG YOLO")
print("=" * 60)

print(f"Mô hình: {MODEL_PATH}")
print(f"Thư mục ảnh: {IMAGE_DIR}")
print(f"Số ảnh cần dự đoán: {len(image_paths)}")


# ==================================================
# 4. DỰ ĐOÁN
# ==================================================

results = model.predict(
    source=[str(path) for path in image_paths],
    imgsz=224,
    batch=16,
    save=True,
    project=str(OUTPUT_DIR),
    name="predict",
    exist_ok=True,
    verbose=False
)


# ==================================================
# 5. HIỂN THỊ VÀ THU THẬP KẾT QUẢ
# ==================================================

csv_rows = []

print("\n" + "=" * 60)
print("KẾT QUẢ DỰ ĐOÁN")
print("=" * 60)

for image_path, result in zip(image_paths, results):
    probabilities = result.probs

    if probabilities is None:
        print(f"\nẢnh: {image_path.name}")
        print("  Không có kết quả phân loại")

        csv_rows.append({
            "ten_anh": image_path.name,
            "loai_tien_du_doan": "",
            "do_tin_cay_phan_tram": "",
            "duong_dan_anh_goc": str(image_path),
            "trang_thai": "Khong co ket qua phan loai"
        })
        continue

    class_id = int(probabilities.top1)
    class_name = result.names[class_id]
    confidence = float(probabilities.top1conf) * 100

    print(f"\nẢnh: {image_path.name}")
    print(f"  Loại tiền dự đoán: {class_name}")
    print(f"  Độ tin cậy: {confidence:.2f}%")

    csv_rows.append({
        "ten_anh": image_path.name,
        "loai_tien_du_doan": class_name,
        "do_tin_cay_phan_tram": round(confidence, 2),
        "duong_dan_anh_goc": str(image_path),
        "trang_thai": "Thanh cong"
    })


# ==================================================
# 6. XUẤT KẾT QUẢ RA FILE CSV
# ==================================================

CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "ten_anh",
    "loai_tien_du_doan",
    "do_tin_cay_phan_tram",
    "duong_dan_anh_goc",
    "trang_thai"
]

with CSV_PATH.open(
    mode="w",
    newline="",
    encoding="utf-8-sig"
) as csv_file:
    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(csv_rows)


# ==================================================
# 7. THÔNG BÁO HOÀN TẤT
# ==================================================

print("\n" + "=" * 60)
print("HOÀN TẤT DỰ ĐOÁN")
print("=" * 60)

print(f"Tổng số ảnh: {len(image_paths)}")
print(f"Số dòng kết quả CSV: {len(csv_rows)}")
print(f"File CSV: {CSV_PATH}")
print(f"Thư mục ảnh kết quả: {OUTPUT_DIR / 'predict'}")
