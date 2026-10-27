
# from pathlib import Path
# import json
# import numpy as np
# import tensorflow as tf

# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# # ==================================================
# # 1. KHAI BÁO ĐƯỜNG DẪN
# # ==================================================

# BASE_DIR = Path(__file__).resolve().parent

# MODEL_PATH = (
#     BASE_DIR
#     / "ket_qua_mobilenetv2"
#     / "mobilenetv2_currency_denomination.keras"
# )

# CLASS_NAMES_PATH = (
#     BASE_DIR
#     / "ket_qua_mobilenetv2"
#     / "class_names.json"
# )

# # Thay đường dẫn này bằng ảnh bạn muốn kiểm tra
# IMAGE_PATH = (
#     BASE_DIR
#     / "anh_kiem_tra"
#     / "10_usd_544.jpg"
# )


# # ==================================================
# # 2. KIỂM TRA CÁC TỆP CẦN THIẾT
# # ==================================================

# for path in [MODEL_PATH, CLASS_NAMES_PATH, IMAGE_PATH]:
#     if not path.is_file():
#         raise FileNotFoundError(
#             f"Không tìm thấy tệp: {path}"
#         )


# # ==================================================
# # 3. TẢI MÔ HÌNH VÀ DANH SÁCH NHÃN
# # ==================================================

# print("Đang tải mô hình MobileNetV2...")

# model = tf.keras.models.load_model(
#     MODEL_PATH,
#     compile=False
# )

# with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
#     class_names = json.load(file)

# print("Đã tải mô hình thành công!")
# print("Số lớp:", len(class_names))


# # ==================================================
# # 4. ĐỌC VÀ TIỀN XỬ LÝ ẢNH
# # ==================================================

# image = tf.keras.utils.load_img(
#     IMAGE_PATH,
#     target_size=(224, 224),
#     color_mode="rgb"
# )

# image_array = tf.keras.utils.img_to_array(image)

# # Thêm chiều batch: (224, 224, 3) -> (1, 224, 224, 3)
# image_array = np.expand_dims(image_array, axis=0)

# # Tiền xử lý giống lúc huấn luyện MobileNetV2
# image_array = preprocess_input(image_array)


# # ==================================================
# # 5. DỰ ĐOÁN
# # ==================================================

# print("\nĐang dự đoán ảnh...")

# predictions = model.predict(image_array, verbose=0)[0]

# predicted_index = int(np.argmax(predictions))
# predicted_label = class_names[predicted_index]
# confidence = float(predictions[predicted_index]) * 100


# # ==================================================
# # 6. HIỂN THỊ KẾT QUẢ
# # ==================================================

# print("\n========== KẾT QUẢ DỰ ĐOÁN ==========")
# print("Ảnh kiểm tra:", IMAGE_PATH.name)
# print("Nhãn dự đoán:", predicted_label)
# print(f"Độ tin cậy của mô hình: {confidence:.2f}%")

# print("\nTop 3 lớp dự đoán:")

# top_indices = np.argsort(predictions)[::-1][:3]

# for rank, index in enumerate(top_indices, start=1):
#     print(
#         f"{rank}. {class_names[int(index)]}: "
#         f"{predictions[index] * 100:.2f}%"
#     )

# print("\nĐã hoàn tất dự đoán!")



from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==================================================
# 1. KHAI BÁO ĐƯỜNG DẪN
# ==================================================

# Thư mục chứa file Python hiện tại:
# Python-Final-Project/model_mobilenetv2
BASE_DIR = Path(__file__).resolve().parent

# Thư mục gốc của dự án:
# Python-Final-Project
PROJECT_DIR = BASE_DIR.parent

# Thư mục chứa mô hình đã huấn luyện
MODEL_DIR = PROJECT_DIR / "ket_qua_mobilenetv2"

MODEL_PATH = (
    MODEL_DIR / "mobilenetv2_currency_denomination.keras"
)

CLASS_NAMES_PATH = (
    MODEL_DIR / "class_names.json"
)

# Thư mục chứa ảnh cần kiểm tra
IMAGE_DIR = PROJECT_DIR / "anh_kiem_tra"

# File lưu bảng kết quả dự đoán
CSV_PATH = BASE_DIR / "ket_qua_du_doan.csv"

# Các định dạng ảnh được hỗ trợ
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# ==================================================
# 2. KIỂM TRA CÁC TỆP VÀ THƯ MỤC
# ==================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Không tìm thấy mô hình: {MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.is_file():
    raise FileNotFoundError(
        f"Không tìm thấy danh sách nhãn: {CLASS_NAMES_PATH}"
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
    print(f"Không tìm thấy ảnh trong thư mục: {IMAGE_DIR}")
    raise SystemExit


# ==================================================
# 3. TẢI MÔ HÌNH VÀ DANH SÁCH NHÃN
# ==================================================

print("Đang tải mô hình MobileNetV2...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

print("Đã tải mô hình thành công!")
print("Số lớp:", len(class_names))
print("Số ảnh cần dự đoán:", len(image_paths))


# ==================================================
# 4. DỰ ĐOÁN TỪNG ẢNH
# ==================================================

results = []

for image_path in image_paths:
    try:
        # Đọc ảnh và thay đổi kích thước về 224 x 224
        image = tf.keras.utils.load_img(
            image_path,
            target_size=(224, 224),
            color_mode="rgb"
        )

        # Chuyển ảnh thành mảng NumPy
        image_array = tf.keras.utils.img_to_array(image)

        # Thêm chiều batch
        image_array = np.expand_dims(image_array, axis=0)

        # Tiền xử lý giống khi huấn luyện MobileNetV2
        image_array = preprocess_input(image_array)

        # Dự đoán
        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        # Lấy lớp có điểm dự đoán cao nhất
        predicted_index = int(np.argmax(predictions))
        predicted_label = class_names[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        # Lấy Top 3 lớp dự đoán
        top_indices = np.argsort(predictions)[::-1][:3]

        top_3 = " | ".join(
            f"{class_names[int(index)]}: "
            f"{predictions[index] * 100:.2f}%"
            for index in top_indices
        )

        results.append({
            "Tên ảnh": image_path.name,
            "Nhãn dự đoán": predicted_label,
            "Độ tin cậy (%)": round(confidence, 2),
            "Top 3 dự đoán": top_3,
            "Trạng thái": "Thành công"
        })

        print(f"Đã dự đoán: {image_path.name}")

    except Exception as error:
        # Một ảnh lỗi không làm dừng toàn bộ chương trình
        results.append({
            "Tên ảnh": image_path.name,
            "Nhãn dự đoán": "Không dự đoán được",
            "Độ tin cậy (%)": None,
            "Top 3 dự đoán": "",
            "Trạng thái": f"Lỗi: {error}"
        })

        print(f"Lỗi ảnh {image_path.name}: {error}")


# ==================================================
# 5. HIỂN THỊ KẾT QUẢ DƯỚI DẠNG TABLE
# ==================================================

df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("BẢNG KẾT QUẢ DỰ ĐOÁN TIỀN TỆ BẰNG MOBILENETV2")
print("=" * 100)

# Không giới hạn độ rộng cột khi hiển thị trong Terminal
with pd.option_context(
    "display.max_rows", None,
    "display.max_columns", None,
    "display.width", 250,
    "display.max_colwidth", 100
):
    print(df.to_string(index=False))


# ==================================================
# 6. THỐNG KÊ VÀ LƯU KẾT QUẢ
# ==================================================

total_images = len(results)
successful = sum(
    row["Trạng thái"] == "Thành công"
    for row in results
)
failed = total_images - successful

print("\n========== THỐNG KÊ ==========")
print("Tổng số ảnh:", total_images)
print("Dự đoán thành công:", successful)
print("Ảnh xảy ra lỗi:", failed)

# Xuất bảng ra CSV, hỗ trợ mở bằng Excel
df.to_csv(
    CSV_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nĐã lưu bảng kết quả tại:")
print(CSV_PATH)
