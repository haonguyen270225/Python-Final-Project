from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf



# =========================================================
# 1. CAU HINH DUONG DAN
# =========================================================

# Thư mục chứa file Python hiện tại: model_custom_cnn
SCRIPT_DIR = Path(__file__).resolve().parent

# Thư mục gốc của dự án: Python-Final-Project
BASE_DIR = SCRIPT_DIR.parent

# Thư mục chứa ảnh cần dự đoán
IMAGE_DIR = BASE_DIR / "anh_kiem_tra/Bo_test_3"

# Thư mục chứa mô hình Custom CNN đã huấn luyện
OUTPUT_DIR = BASE_DIR / "ket_qua_custom_cnn"

# Đường dẫn mô hình tốt nhất
MODEL_PATH = OUTPUT_DIR / "best_custom_cnn.keras"

# Đường dẫn danh sách nhãn
CLASS_NAMES_PATH = OUTPUT_DIR / "class_names.json"

# Đường dẫn lưu bảng kết quả dự đoán
CSV_PATH = BASE_DIR / "model_custom_cnn/ket_qua_du_doan_custom_cnn_test_03.csv"

# Các định dạng ảnh được hỗ trợ
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# =========================================================
# 2. KIEM TRA CAC DUONG DAN
# =========================================================

for path in [MODEL_PATH, CLASS_NAMES_PATH]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy tệp: {path}"
        )

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục ảnh: {IMAGE_DIR}"
    )

# Lấy tất cả ảnh nằm trực tiếp trong thư mục anh_kiem_tra
image_paths = sorted(
    path
    for path in IMAGE_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_paths:
    raise FileNotFoundError(
        f"Không tìm thấy ảnh hợp lệ trong: {IMAGE_DIR}"
    )

print("=" * 70)
print("THU NGHIEM MO HINH CUSTOM CNN")
print("=" * 70)

print("Thư mục dự án:", BASE_DIR)
print("Thư mục ảnh:", IMAGE_DIR)
print("Mô hình:", MODEL_PATH)
print("Danh sách lớp:", CLASS_NAMES_PATH)
print("Số ảnh cần dự đoán:", len(image_paths))


# =========================================================
# 3. TAI MO HINH VA DANH SACH LOP
# =========================================================

print("\nĐang tải mô hình Custom CNN...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

# Hỗ trợ class_names lưu dưới dạng list hoặc dictionary
if isinstance(class_names, dict):
    class_names = [
        name
        for name, index in sorted(
            class_names.items(),
            key=lambda item: int(item[1])
        )
    ]

print("Đã tải mô hình thành công!")
print("Số lớp:", len(class_names))


# =========================================================
# 4. KIEM TRA KICH THUOC DAU VAO
# =========================================================

input_shape = model.input_shape

if isinstance(input_shape, list):
    input_shape = input_shape[0]

image_height = input_shape[1] or 224
image_width = input_shape[2] or 224

print(
    f"Kích thước ảnh đầu vào: "
    f"{image_width} x {image_height}"
)

# Kiểm tra số lớp đầu ra và danh sách nhãn
output_shape = model.output_shape

if isinstance(output_shape, list):
    output_shape = output_shape[0]

if output_shape[-1] != len(class_names):
    raise ValueError(
        f"Mô hình có {output_shape[-1]} lớp đầu ra, "
        f"nhưng class_names.json có {len(class_names)} nhãn."
    )


# =========================================================
# 5. DU DOAN TUNG ANH
# =========================================================

results = []

for image_path in image_paths:
    try:
        # Đọc ảnh RGB và resize theo mô hình
        image = tf.keras.utils.load_img(
            image_path,
            target_size=(image_height, image_width),
            color_mode="rgb"
        )

        # Chuyển ảnh thành mảng NumPy
        image_array = tf.keras.utils.img_to_array(image)

        # Thêm chiều batch
        image_array = np.expand_dims(image_array, axis=0)

        # TIỀN XỬ LÝ:
        # Phải giống với cách tiền xử lý trong lúc huấn luyện.
        # Đoạn này giả định mô hình được huấn luyện với ảnh / 255.0.
        image_array = image_array / 255.0

        # Dự đoán
        probabilities = model.predict(
            image_array,
            verbose=0
        )[0]

        # Lấy lớp có điểm dự đoán cao nhất
        predicted_index = int(np.argmax(probabilities))
        predicted_label = class_names[predicted_index]

        confidence = float(
            probabilities[predicted_index]
        ) * 100

        # Lấy Top 3 lớp dự đoán
        top_indices = np.argsort(probabilities)[::-1][
            :min(3, len(probabilities))
        ]

        top_3 = " | ".join(
            f"{class_names[int(index)]}: "
            f"{probabilities[index] * 100:.2f}%"
            for index in top_indices
        )

        results.append({
            "Tên ảnh": image_path.name,
            "Nhãn dự đoán": predicted_label,
            "Độ tin cậy (%)": round(confidence, 2),
            "Top 3 dự đoán": top_3,
            "Trạng thái": "Thành công"
        })

        print(
            f"Đã dự đoán: {image_path.name} "
            f"-> {predicted_label} ({confidence:.2f}%)"
        )

    except Exception as error:
        results.append({
            "Tên ảnh": image_path.name,
            "Nhãn dự đoán": "Không dự đoán được",
            "Độ tin cậy (%)": None,
            "Top 3 dự đoán": "",
            "Trạng thái": f"Lỗi: {error}"
        })

        print(f"Lỗi ảnh {image_path.name}: {error}")


# =========================================================
# 6. HIEN THI BANG KET QUA
# =========================================================

df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("BANG KET QUA DU DOAN TIEN TE BANG CUSTOM CNN")
print("=" * 100)

with pd.option_context(
    "display.max_rows", None,
    "display.max_columns", None,
    "display.width", 240,
    "display.max_colwidth", 80
):
    print(df.to_string(index=False))


# =========================================================
# 7. THONG KE KET QUA
# =========================================================

total = len(results)

success = sum(
    row["Trạng thái"] == "Thành công"
    for row in results
)

failed = total - success

print("\n========== THONG KE ==========")
print("Tổng số ảnh:", total)
print("Dự đoán thành công:", success)
print("Ảnh xảy ra lỗi:", failed)


# =========================================================
# 8. LUU BANG KET QUA RA CSV
# =========================================================

df.to_csv(
    CSV_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nĐã lưu bảng kết quả tại:")
print(CSV_PATH)

print("\n========== HOAN TAT ==========")
