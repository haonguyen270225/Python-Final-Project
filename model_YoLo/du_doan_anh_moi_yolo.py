# # 1. Chạy trong PowerShell : cd D:\Python-Final-Project
# # 2 . Chạy trong PowerShell : .\.venv\Scripts\python.exe .\model_YoLo\du_doan_anh_moi_yolo.py

from pathlib import Path
import csv

from ultralytics import YOLO


# ==================================================
# 1. CẤU HÌNH
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Mô hình tốt nhất sau khi huấn luyện (Đã sửa bỏ thư mục "runs" dư thừa)
MODEL_PATH = (
    BASE_DIR
    / "ket_qua_yolo"
    / "currency_yolo11n_cls"
    / "weights"
    / "best.pt"
)

# Thư mục chứa ảnh cần dự đoán
IMAGE_DIR = BASE_DIR / "anh_kiem_tra/Bo_test_3"

# Thư mục lưu ảnh kết quả
OUTPUT_DIR = BASE_DIR / "model_YoLo"

# File CSV lưu kết quả dự đoán
CSV_PATH = (
    Path(__file__).resolve().parent
    / "ket_qua_du_doan_yolo_test_03.csv"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}


# ==================================================
# 2. HÀM TÁCH LOẠI TIỀN VÀ MỆNH GIÁ
# ==================================================

def parse_currency_label(class_name):
    """
    Ví dụ:
        CNY_100  -> CNY, 100
        USD_50   -> USD, 50
        VND_500000 -> VND, 500000
    """

    parts = str(class_name).strip().upper().split("_")

    if len(parts) != 2:
        return "", None

    currency = parts[0]
    denomination_text = parts[1]

    if currency not in {"VND", "USD", "CNY"}:
        return "", None

    if not denomination_text.isdigit():
        return "", None

    denomination = int(denomination_text)

    return currency, denomination


# ==================================================
# 3. KIỂM TRA ĐƯỜNG DẪN
# ==================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Không tìm thấy mô hình: {MODEL_PATH}\n"
        "Hãy kiểm tra lại đường dẫn best.pt."
    )

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục ảnh: {IMAGE_DIR}"
    )

image_paths = sorted(
    path
    for path in IMAGE_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_paths:
    raise ValueError(
        f"Không tìm thấy ảnh trực tiếp trong: {IMAGE_DIR}"
    )


# ==================================================
# 4. NẠP MÔ HÌNH
# ==================================================

model = YOLO(str(MODEL_PATH))

print("=" * 60)
print("DỰ ĐOÁN MỆNH GIÁ VÀ LOẠI TIỀN BẰNG YOLO")
print("=" * 60)

print(f"Mô hình: {MODEL_PATH}")
print(f"Thư mục ảnh: {IMAGE_DIR}")
print(f"Số ảnh cần dự đoán: {len(image_paths)}")

print("\nCác lớp nhận diện của mô hình:")

for class_id, class_name in model.names.items():
    currency, denomination = parse_currency_label(
        class_name
    )

    denomination_text = (
        f"{denomination:,}"
        if denomination is not None
        else "Chưa xác định"
    )

    print(
        f"  {class_id}: {class_name}"
        f" | Loại tiền: {currency or 'Chưa xác định'}"
        f" | Mệnh giá: {denomination_text}"
    )


# ==================================================
# 5. DỰ ĐOÁN ẢNH
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
# 6. HIỂN THỊ VÀ THU THẬP KẾT QUẢ
# ==================================================

csv_rows = []

print("\n" + "=" * 60)
print("KẾT QUẢ DỰ ĐOÁN")
print("=" * 60)

for image_path, result in zip(image_paths, results):

    probabilities = result.probs

    if probabilities is None:
        print(f"\nẢnh: {image_path.name}")
        print("  Không có kết quả phân loại.")

        csv_rows.append({
            "ten_anh": image_path.name,
            "loai_tien_du_doan": "",
            "menh_gia_du_doan": "",
            "do_tin_cay_phan_tram": "",
            "duong_dan_anh_goc": str(image_path),
            "trang_thai": "Khong co ket qua phan loai"
        })

        continue

    # Lấy lớp có xác suất cao nhất
    class_id = int(probabilities.top1)
    class_name = result.names[class_id]

    # Độ tin cậy (%)
    confidence = (
        float(probabilities.top1conf) * 100
    )

    # Tách nhãn thành loại tiền và mệnh giá
    currency, denomination = parse_currency_label(
        class_name
    )

    denomination_text = (
        f"{denomination:,}"
        if denomination is not None
        else "Chưa xác định"
    )

    if currency and denomination is not None:
        status = "Thanh cong"
    else:
        status = "Nhan lop khong hop le"

    print(f"\nẢnh: {image_path.name}")
    print(f"  Nhãn lớp: {class_name}")
    print(f"  Loại tiền dự đoán: {currency or 'Chưa xác định'}")
    print(f"  Mệnh giá dự đoán: {denomination_text}")
    print(f"  Độ tin cậy: {confidence:.2f}%")

    csv_rows.append({
        "ten_anh": image_path.name,
        "loai_tien_du_doan": currency,
        "menh_gia_du_doan": (
            denomination
            if denomination is not None
            else ""
        ),
        "do_tin_cay_phan_tram": round(
            confidence, 2
        ),
        "duong_dan_anh_goc": str(image_path),
        "trang_thai": status
    })


# ==================================================
# 7. XUẤT KẾT QUẢ RA FILE CSV
# ==================================================

CSV_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

fieldnames = [
    "ten_anh",
    "loai_tien_du_doan",
    "menh_gia_du_doan",
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
# 8. THÔNG BÁO HOÀN TẤT
# ==================================================

print("\n" + "=" * 60)
print("HOÀN TẤT DỰ ĐOÁN")
print("=" * 60)

print(f"Tổng số ảnh: {len(image_paths)}")
print(f"Số dòng kết quả CSV: {len(csv_rows)}")
print(f"File CSV: {CSV_PATH}")
print(f"Thư mục ảnh kết quả: {OUTPUT_DIR / 'predict'}")
