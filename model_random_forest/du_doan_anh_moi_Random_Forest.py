# # import cv2
# # import joblib

# # # 1. Tải mô hình đã huấn luyện lên
# # model_path = "rf_cny_classifier.pkl"
# # rf_model = joblib.load(model_path)

# # # 2. Đường dẫn tới bức ảnh mới bạn muốn nhận diện (Thay bằng ảnh của bạn)
# # image_to_predict = "duong_dan_anh_moi_cua_ban.jpg" 

# # try:
# #     # 3. Tiền xử lý ảnh mới đúng chuẩn với lúc huấn luyện (Grayscale -> Resize 64x64 -> Flatten)
# #     img = cv2.imread(image_to_predict, cv2.IMREAD_GRAYSCALE)
# #     if img is not None:
# #         img_resized = cv2.resize(img, (64, 64))
# #         img_flatten = img_resized.flatten()
        
# #         # Random Forest yêu cầu đầu vào dạng ma trận (n_samples, n_features)
# #         input_data = [img_flatten] 
        
# #         # 4. Dự đoán loại tiền và mệnh giá
# #         prediction = rf_model.predict(input_data)
        
# #         # Xuất kết quả
# #         print(f"\n Kết quả nhận diện: Đây là tờ tiền mệnh giá [{prediction[0]}]")
# #     else:
# #         print("Không thể đọc được file ảnh. Vui lòng kiểm tra lại đường dẫn.")
        
# # except Exception as e:
# #     print(f"Có lỗi xảy ra: {e}")


# import cv2
# import joblib
# from pathlib import Path

# # ============================================================
# # 1. CẤU HÌNH
# # ============================================================

# # Đường dẫn mô hình Random Forest đã huấn luyện
# MODEL_PATH = "rf_cny_classifier.pkl"

# # Thư mục chứa ảnh cần kiểm tra
# IMAGE_FOLDER = "anh_kiem_tra"

# # Kích thước ảnh đầu vào
# IMAGE_SIZE = (64, 64)

# # Các định dạng ảnh được hỗ trợ
# IMAGE_EXTENSIONS = {
#     ".jpg",
#     ".jpeg",
#     ".png",
#     ".bmp",
#     ".webp"
# }


# # ============================================================
# # 2. TẢI MÔ HÌNH
# # ============================================================

# try:
#     rf_model = joblib.load(MODEL_PATH)
#     print("Đã tải mô hình Random Forest thành công!")

# except Exception as e:
#     print(f"Lỗi khi tải mô hình: {e}")
#     exit()


# # ============================================================
# # 3. LẤY DANH SÁCH ẢNH
# # ============================================================

# image_dir = Path(IMAGE_FOLDER)

# if not image_dir.exists():
#     print(f"Không tìm thấy thư mục: {IMAGE_FOLDER}")
#     exit()

# image_files = [
#     file for file in image_dir.iterdir()
#     if file.is_file()
#     and file.suffix.lower() in IMAGE_EXTENSIONS
# ]

# if not image_files:
#     print("Không tìm thấy ảnh trong thư mục kiểm tra!")
#     exit()


# # ============================================================
# # 4. DỰ ĐOÁN TOÀN BỘ ẢNH
# # ============================================================

# print("\n" + "=" * 65)
# print("DỰ ĐOÁN MỆNH GIÁ TIỀN BẰNG RANDOM FOREST")
# print("=" * 65)

# success_count = 0
# error_count = 0

# for index, image_path in enumerate(image_files, start=1):

#     try:
#         # Đọc ảnh dưới dạng Grayscale
#         img = cv2.imread(
#             str(image_path),
#             cv2.IMREAD_GRAYSCALE
#         )

#         if img is None:
#             print(
#                 f"[{index}/{len(image_files)}] "
#                 f"{image_path.name}: Không thể đọc ảnh!"
#             )
#             error_count += 1
#             continue

#         # Resize ảnh về 64x64
#         img_resized = cv2.resize(
#             img,
#             IMAGE_SIZE
#         )

#         # Flatten thành vector 1 chiều
#         img_flatten = img_resized.flatten()

#         # Chuyển thành ma trận (1, 4096)
#         input_data = [img_flatten]

#         # Dự đoán mệnh giá
#         prediction = rf_model.predict(input_data)

#         predicted_label = prediction[0]

#         # Hiển thị kết quả
#         print(
#             f"[{index}/{len(image_files)}] "
#             f"{image_path.name:<30} "
#             f"=> Mệnh giá: {predicted_label}"
#         )

#         success_count += 1

#     except Exception as e:
#         print(
#             f"[{index}/{len(image_files)}] "
#             f"{image_path.name}: Lỗi - {e}"
#         )
#         error_count += 1


# # ============================================================
# # 5. THỐNG KÊ KẾT QUẢ
# # ============================================================

# print("\n" + "=" * 65)
# print("THỐNG KÊ KẾT QUẢ DỰ ĐOÁN")
# print("=" * 65)

# print(f"Tổng số ảnh:          {len(image_files)}")
# print(f"Dự đoán thành công:   {success_count}")
# print(f"Dự đoán thất bại:     {error_count}")

# print("=" * 65)
# print("HOÀN TẤT DỰ ĐOÁN!")
# print("=" * 65)


"""
DỰ ĐOÁN LOẠI TIỀN TỆ VÀ MỆNH GIÁ BẰNG RANDOM FOREST

Dữ liệu đầu vào:
    D:\\Python-Final-Project\\anh_kiem_tra\\

Mô hình:
    D:\\Python-Final-Project\\ket_qua_random_forset\\
        rf_currency_classifier.pkl

Kết quả:
    D:\\Python-Final-Project\\model_random_forest\\
        ket_qua_du_doan_Random_Forest.csv

Quy trình xử lý:
    1. Đọc ảnh Grayscale.
    2. Resize ảnh về 64x64.
    3. Flatten thành vector 4096 đặc trưng.
    4. Dự đoán nhãn tiền tệ và mệnh giá.
    5. Tính độ tin cậy bằng predict_proba().
    6. Xuất kết quả thành CSV.
"""

from pathlib import Path
import csv
import sys

import cv2
import joblib
import numpy as np


# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Thư mục chứa ảnh cần dự đoán
IMAGE_DIR = BASE_DIR / "anh_kiem_tra"

# Thư mục chứa mô hình đã huấn luyện
RESULT_DIR = BASE_DIR / "ket_qua_random_forset"

# Thư mục chứa file Python hiện tại
MODEL_DIR = Path(__file__).resolve().parent

# Đường dẫn mô hình Random Forest
MODEL_PATH = RESULT_DIR / "rf_currency_classifier.pkl"

# Đường dẫn file CSV kết quả dự đoán
CSV_PATH = MODEL_DIR / "ket_qua_du_doan_Random_Forest.csv"


# =========================================================
# 2. CẤU HÌNH XỬ LÝ ẢNH
# =========================================================

IMAGE_SIZE = (64, 64)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# =========================================================
# 3. TIỀN XỬ LÝ ẢNH
# =========================================================

def preprocess_image(image_path: Path):
    """
    Đọc ảnh Grayscale, resize 64x64 và flatten.

    Kết quả:
        Vector có 4096 đặc trưng.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        return None

    resized_image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    flattened_image = resized_image.flatten()

    return flattened_image


# =========================================================
# 4. TẢI MÔ HÌNH RANDOM FOREST
# =========================================================

def load_model():

    print("\nĐANG TẢI MÔ HÌNH RANDOM FOREST")
    print("=" * 65)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mô hình: {MODEL_PATH}\n"
            "Hãy chạy huan_luyen_Random_Forest.py trước."
        )

    model = joblib.load(MODEL_PATH)

    print("Đã tải mô hình thành công!")
    print(f"Đường dẫn: {MODEL_PATH}")

    print("\nCác lớp mô hình có thể nhận diện:")

    for label in model.classes_:
        print(f"- {label}")

    return model


# =========================================================
# 5. PHÂN TÁCH NHÃN TIỀN TỆ VÀ MỆNH GIÁ
# =========================================================

def parse_label(label: str):
    """
    Phân tách nhãn kết hợp.

    Ví dụ:
        USD_1     -> USD, 1
        CNY_10    -> CNY, 10
        VND_50000 -> VND, 50000
    """

    parts = str(label).split("_", 1)

    if len(parts) != 2:
        return str(label), ""

    currency = parts[0]
    denomination = parts[1]

    return currency, denomination


# =========================================================
# 6. DỰ ĐOÁN MỘT ẢNH
# =========================================================

def predict_image(model, image_path: Path):

    features = preprocess_image(image_path)

    if features is None:
        raise ValueError("Không thể đọc ảnh.")

    # Random Forest nhận đầu vào dạng (n_samples, n_features)
    input_data = np.asarray(
        [features],
        dtype=np.uint8
    )

    # Dự đoán nhãn
    prediction = model.predict(input_data)

    predicted_label = str(prediction[0])

    # Lấy xác suất dự đoán của từng lớp
    probabilities = model.predict_proba(input_data)[0]

    # Xác suất của lớp được dự đoán
    confidence = float(np.max(probabilities))

    # Tách loại tiền và mệnh giá
    currency, denomination = parse_label(
        predicted_label
    )

    return {
        "predicted_label": predicted_label,
        "currency": currency,
        "denomination": denomination,
        "confidence": confidence,
        "confidence_percentage": confidence * 100,
    }


# =========================================================
# 7. LƯU KẾT QUẢ CSV
# =========================================================

def save_results_csv(results):

    fieldnames = [
        "image_name",
        "image_path",
        "currency",
        "denomination",
        "predicted_label",
        "confidence",
        "confidence_percentage",
        "status",
        "error",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    print(f"\nĐã xuất kết quả CSV: {CSV_PATH}")


# =========================================================
# 8. DỰ ĐOÁN TOÀN BỘ ẢNH
# =========================================================

def predict_all_images(model):

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh: {IMAGE_DIR}"
        )

    image_files = sorted(
        path
        for path in IMAGE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        raise RuntimeError(
            f"Không tìm thấy ảnh trong thư mục: {IMAGE_DIR}"
        )

    print("\nĐANG DỰ ĐOÁN ẢNH")
    print("=" * 75)

    results = []

    success_count = 0
    error_count = 0

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        try:
            prediction = predict_image(
                model,
                image_path
            )

            result = {
                "image_name": image_path.name,
                "image_path": str(
                    image_path.relative_to(BASE_DIR)
                ),
                "currency": prediction["currency"],
                "denomination": prediction["denomination"],
                "predicted_label": prediction["predicted_label"],
                "confidence": f"{prediction['confidence']:.6f}",
                "confidence_percentage": (
                    f"{prediction['confidence_percentage']:.2f}%"
                ),
                "status": "Success",
                "error": "",
            }

            results.append(result)

            print(
                f"[{index}/{len(image_files)}] "
                f"{image_path.name:<30} "
                f"=> {prediction['predicted_label']:<15} "
                f"| Độ tin cậy: "
                f"{prediction['confidence_percentage']:.2f}%"
            )

            success_count += 1

        except Exception as error:

            results.append({
                "image_name": image_path.name,
                "image_path": str(
                    image_path.relative_to(BASE_DIR)
                ),
                "currency": "",
                "denomination": "",
                "predicted_label": "",
                "confidence": "",
                "confidence_percentage": "",
                "status": "Failed",
                "error": str(error),
            })

            print(
                f"[{index}/{len(image_files)}] "
                f"{image_path.name}: Lỗi - {error}"
            )

            error_count += 1

    return results, len(image_files), success_count, error_count


# =========================================================
# 9. THỐNG KÊ KẾT QUẢ DỰ ĐOÁN
# =========================================================

def print_summary(
    total_count,
    success_count,
    error_count
):

    print("\n" + "=" * 65)
    print("THỐNG KÊ KẾT QUẢ DỰ ĐOÁN")
    print("=" * 65)

    print(f"Tổng số ảnh          : {total_count}")
    print(f"Dự đoán thành công   : {success_count}")
    print(f"Dự đoán thất bại     : {error_count}")

    if total_count > 0:
        success_rate = (
            success_count / total_count
        ) * 100

        print(
            f"Tỷ lệ xử lý thành công: {success_rate:.2f}%"
        )

    print("=" * 65)


# =========================================================
# 10. CHƯƠNG TRÌNH CHÍNH
# =========================================================

def main():

    print("=" * 75)
    print("DỰ ĐOÁN LOẠI TIỀN TỆ VÀ MỆNH GIÁ")
    print("SỬ DỤNG MÔ HÌNH RANDOM FOREST")
    print("=" * 75)

    # Tạo thư mục kết quả nếu chưa tồn tại
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Bước 1: Tải mô hình
    rf_model = load_model()

    # Bước 2: Dự đoán toàn bộ ảnh
    results, total_count, success_count, error_count = (
        predict_all_images(rf_model)
    )

    # Bước 3: Xuất CSV
    save_results_csv(results)

    # Bước 4: Thống kê
    print_summary(
        total_count,
        success_count,
        error_count
    )

    print("\nHOÀN TẤT DỰ ĐOÁN RANDOM FOREST!")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print(
            f"\nLỖI: {error}",
            file=sys.stderr
        )

        raise
