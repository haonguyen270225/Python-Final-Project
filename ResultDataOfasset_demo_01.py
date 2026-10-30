

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Thư mục gốc chứa ảnh tiền
DATASET_DIR = Path("asset_finally")

# File CSV đầu ra
OUTPUT_CSV = "cvsDataOfAssetFinally.csv"


def extract_features(image_path):
    """Đọc ảnh và trích xuất đặc trưng màu sắc."""

    # Đọc ảnh màu 
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Không đọc được ảnh: {image_path}")
        return None

    # Chuẩn hóa kích thước để thống nhất xử lý
    image = cv2.resize(image, (256, 128))

    # OpenCV đọc ảnh theo thứ tự BGR
    features = {}

    # Đặc trưng màu sắc trung bình và độ lệch chuẩn
    for i, channel in enumerate(["B", "G", "R"]):
        features[f"mean_{channel}"] = float(
            np.mean(image[:, :, i])
        )
        features[f"std_{channel}"] = float(
            np.std(image[:, :, i])
        )

    # Chuyển sang HSV để mô tả màu sắc
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Histogram màu: 8 bins cho mỗi kênh H, S, V
    for i, channel in enumerate(["H", "S", "V"]):
        hist = cv2.calcHist(
            [hsv], [i], None, [8],
            [0, 180] if channel == "H" else [0, 256]
        )

        # Chuẩn hóa histogram
        hist = cv2.normalize(hist, hist).flatten()

        for j, value in enumerate(hist):
            features[f"hist_{channel}_{j}"] = float(value)

    return features


all_rows = []
supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Cấu trúc thư mục: currency / denomination / image
for currency_dir in DATASET_DIR.iterdir():
    if not currency_dir.is_dir():
        continue

    currency = currency_dir.name.upper()

    for denomination_dir in currency_dir.iterdir():
        if not denomination_dir.is_dir():
            continue

        denomination = denomination_dir.name

        for image_path in denomination_dir.rglob("*"):
            if image_path.suffix.lower() not in supported_extensions:
                continue

            features = extract_features(image_path)

            if features is None:
                continue

            # Thêm nhãn và tên ảnh
            features["image_name"] = image_path.name
            features["currency"] = currency
            features["denomination"] = denomination

            all_rows.append(features)

# Xuất CSV
if all_rows:
    df = pd.DataFrame(all_rows)

    # Đưa thông tin nhận dạng lên đầu bảng
    label_columns = [
        "image_name", "currency", "denomination"
    ]
    other_columns = [
        col for col in df.columns
        if col not in label_columns
    ]
    df = df[label_columns + other_columns]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("Đã chuyển đổi thành công!")
    print("Số ảnh xử lý:", len(df))
    print("Kích thước CSV:", df.shape)
    print(df.head())
else:
    print("Không tìm thấy ảnh hợp lệ. Hãy kiểm tra thư mục dữ liệu.")
