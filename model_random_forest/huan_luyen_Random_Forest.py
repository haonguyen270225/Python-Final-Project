# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# import joblib

# # 1. Định nghĩa cấu hình
# DATASET_DIR = "asset_Random_Forest"  # Tên thư mục gốc của bạn
# IMG_SIZE = (64, 64)                  # Kích thước ảnh chuẩn hóa (giảm dung lượng bộ nhớ)

# X = []
# y = []

# print(" bắt đầu đọc dữ liệu từ các thư mục...")

# # 2. Đọc ảnh và gán nhãn tự động dựa trên tên thư mục
# # Danh sách thư mục sẽ là: ['CNY_1', 'CNY_10', 'CNY_100', 'CNY_20']
# categories = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]

# for category in categories:
#     category_path = os.path.join(DATASET_DIR, category)
#     print(f"Đang xử lý thư mục: {category}")
    
#     for img_name in os.listdir(category_path):
#         # Bỏ qua file ẩn hoặc file không phải ảnh nếu có
#         if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
#             continue
            
#         img_path = os.path.join(category_path, img_name)
        
#         try:
#             # Đọc ảnh dưới dạng ảnh xám (Grayscale) để giảm kích thước đặc trưng
#             # Nếu muốn dùng ảnh màu, hãy bỏ cv2.IMREAD_GRAYSCALE và đổi IMG_SIZE thành 64*64*3
#             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
#             if img is not None:
#                 # Thay đổi kích thước ảnh
#                 img_resized = cv2.resize(img, IMG_SIZE)
#                 # Làm phẳng ảnh từ 2D (64x64) thành 1D (4096 đặc trưng)
#                 img_flatten = img_resized.flatten()
                
#                 X.append(img_flatten)
#                 y.append(category)
#         except Exception as e:
#             print(f"Lỗi khi đọc file {img_name}: {e}")

# # Chuyển đổi sang mảng Numpy
# X = np.array(X)
# y = np.array(y)

# print(f"\nTổng số mẫu ảnh đọc được: {len(X)}")

# # 3. Chia tập dữ liệu thành Train và Test (80% Train, 20% Test)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # 4. Khởi tạo và huấn luyện mô hình Random Forest
# print("\n Đang huấn luyện mô hình Random Forest...")
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# rf_model.fit(X_train, y_train)
# print(" Huấn luyện thành công!")

# # 5. Đánh giá mô hình trên tập Test
# y_pred = rf_model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"\n Độ chính xác (Accuracy): {accuracy * 100:.2f}%")
# print("\n Báo cáo chi tiết hiệu suất phân loại:")
# print(classification_report(y_test, y_pred))

# # 6. Lưu mô hình đã huấn luyện thành file để tái sử dụng sau này
# model_filename = "rf_cny_classifier.pkl"
# joblib.dump(rf_model, model_filename)
# print(f"\n Đã lưu mô hình thành công vào file: {model_filename}")


"""
HUẤN LUYỆN MÔ HÌNH RANDOM FOREST PHÂN LOẠI TIỀN TỆ

Dữ liệu đầu vào:
    D:\\Python-Final-Project\\asset_Random_Forest\\
        USD_1\\
        USD_2\\
        CNY_10\\
        CNY_20\\
        VND_200\\
        VND_50000\\

Dữ liệu đầu ra:
    model_random_forest\\ket_qua_random_forset\\
        rf_currency_classifier.pkl
        classification_report.csv
        confusion_matrix.csv
        ket_qua_danh_gia.csv
        training_summary.txt

Mô hình phân loại theo nhãn kết hợp:
    USD_1   -> USD, mệnh giá 1
    CNY_10  -> CNY, mệnh giá 10
    VND_200 -> VND, mệnh giá 200
"""

from pathlib import Path
from datetime import datetime
import csv
import sys

import cv2
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================

# Thư mục gốc dự án:
# D:\Python-Final-Project
BASE_DIR = Path(__file__).resolve().parent.parent

# Thư mục dữ liệu huấn luyện
DATASET_DIR = BASE_DIR / "asset_Random_Forest"

# Thư mục lưu kết quả (nằm ngoài model_random_forest)
RESULT_DIR = BASE_DIR / "ket_qua_random_forset"

# Đường dẫn file mô hình
MODEL_PATH = RESULT_DIR / "rf_currency_classifier.pkl"

# Đường dẫn báo cáo phân loại
REPORT_PATH = RESULT_DIR / "classification_report.csv"

# Đường dẫn Confusion Matrix
CONFUSION_MATRIX_PATH = RESULT_DIR / "confusion_matrix.csv"

# Đường dẫn chỉ số đánh giá
METRICS_PATH = RESULT_DIR / "ket_qua_danh_gia.csv"

# Đường dẫn báo cáo tổng hợp
SUMMARY_PATH = RESULT_DIR / "training_summary.txt"


# =========================================================
# 2. CẤU HÌNH HUẤN LUYỆN
# =========================================================

IMG_SIZE = (64, 64)

TEST_SIZE = 0.2

RANDOM_STATE = 42

N_ESTIMATORS = 100

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
# 3. HÀM XỬ LÝ ẢNH
# =========================================================

def preprocess_image(image_path: Path):
    """
    Đọc ảnh dưới dạng Grayscale,
    resize về 64x64 và chuyển thành vector 1 chiều.

    64 x 64 = 4096 đặc trưng.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        return None

    resized_image = cv2.resize(
        image,
        IMG_SIZE
    )

    flattened_image = resized_image.flatten()

    return flattened_image


# =========================================================
# 4. ĐỌC DỮ LIỆU VÀ GÁN NHÃN
# =========================================================

def load_dataset():
    """
    Đọc toàn bộ ảnh trong asset_Random_Forest.

    Tên thư mục được sử dụng làm nhãn:
        USD_1
        CNY_10
        VND_200
    """

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {DATASET_DIR}"
        )

    X = []
    y = []

    class_counts = {}

    skipped_count = 0

    categories = sorted(
        folder.name
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    if not categories:
        raise RuntimeError(
            "Không tìm thấy thư mục lớp dữ liệu nào."
        )

    print("\nĐANG ĐỌC DỮ LIỆU HUẤN LUYỆN")
    print("=" * 65)

    for category in categories:

        category_path = DATASET_DIR / category

        print(f"\nĐang xử lý lớp: {category}")

        image_paths = sorted(
            path
            for path in category_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        valid_count = 0

        for image_path in image_paths:

            try:
                image_features = preprocess_image(image_path)

                if image_features is None:
                    skipped_count += 1
                    print(f"  Không đọc được ảnh: {image_path.name}")
                    continue

                X.append(image_features)
                y.append(category)

                valid_count += 1

            except Exception as error:
                skipped_count += 1
                print(
                    f"  Lỗi xử lý {image_path.name}: {error}"
                )

        class_counts[category] = valid_count

        print(f"  Số ảnh hợp lệ: {valid_count}")

    if not X:
        raise RuntimeError(
            "Không đọc được ảnh nào từ dataset."
        )

    X = np.asarray(X, dtype=np.uint8)

    y = np.asarray(y)

    print("\n" + "=" * 65)
    print("THỐNG KÊ DỮ LIỆU")
    print("=" * 65)

    print(f"Tổng số ảnh hợp lệ : {len(X)}")
    print(f"Tổng số lớp        : {len(class_counts)}")
    print(f"Số ảnh bị bỏ qua   : {skipped_count}")
    print(f"Số đặc trưng/ảnh   : {X.shape[1]}")

    for category, count in class_counts.items():
        print(f"{category:<15}: {count:>6} ảnh")

    return X, y, class_counts, skipped_count


# =========================================================
# 5. KIỂM TRA DỮ LIỆU TRƯỚC KHI CHIA TẬP
# =========================================================

def validate_dataset(y, class_counts):

    if len(class_counts) < 2:
        raise ValueError(
            "Cần ít nhất 2 lớp dữ liệu để huấn luyện "
            "mô hình phân loại."
        )

    insufficient_classes = {
        label: count
        for label, count in class_counts.items()
        if count < 2
    }

    if insufficient_classes:
        details = "\n".join(
            f"  {label}: {count} ảnh"
            for label, count in insufficient_classes.items()
        )

        raise ValueError(
            "Một số lớp có ít hơn 2 ảnh nên không thể "
            "chia tập Train/Test có stratify:\n"
            f"{details}\n"
            "Hãy bổ sung ảnh cho các lớp này."
        )

    number_of_classes = len(class_counts)

    number_of_test_samples = int(
        len(y) * TEST_SIZE
    )

    number_of_train_samples = (
        len(y) - number_of_test_samples
    )

    if (
        number_of_test_samples < number_of_classes
        or number_of_train_samples < number_of_classes
    ):
        raise ValueError(
            "Số lượng ảnh không đủ để chia tập Train/Test "
            "với stratify.\n"
            "Hãy bổ sung dữ liệu hoặc điều chỉnh TEST_SIZE."
        )


# =========================================================
# 6. CHIA DỮ LIỆU TRAIN VÀ TEST
# =========================================================

def split_dataset(X, y):

    print("\nĐANG CHIA DỮ LIỆU TRAIN VÀ TEST")
    print("=" * 65)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Số ảnh Train: {len(X_train)}")
    print(f"Số ảnh Test : {len(X_test)}")

    print(
        f"Tỷ lệ Train: {(1 - TEST_SIZE) * 100:.0f}%"
    )

    print(
        f"Tỷ lệ Test : {TEST_SIZE * 100:.0f}%"
    )

    return X_train, X_test, y_train, y_test


# =========================================================
# 7. HUẤN LUYỆN RANDOM FOREST
# =========================================================

def train_model(X_train, y_train):

    print("\nĐANG HUẤN LUYỆN RANDOM FOREST")
    print("=" * 65)

    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    rf_model.fit(
        X_train,
        y_train
    )

    print("Huấn luyện mô hình thành công!")

    return rf_model


# =========================================================
# 8. ĐÁNH GIÁ MÔ HÌNH
# =========================================================

def evaluate_model(model, X_test, y_test):

    print("\nĐANG ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 65)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision_macro = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision_weighted = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    metrics = {
        "Accuracy": accuracy,
        "Precision_Macro": precision_macro,
        "Recall_Macro": recall_macro,
        "F1_Macro": f1_macro,
        "Precision_Weighted": precision_weighted,
        "Recall_Weighted": recall_weighted,
        "F1_Weighted": f1_weighted,
    }

    print(f"Accuracy          : {accuracy * 100:.2f}%")
    print(f"Precision (Macro) : {precision_macro * 100:.2f}%")
    print(f"Recall (Macro)    : {recall_macro * 100:.2f}%")
    print(f"F1-score (Macro)  : {f1_macro * 100:.2f}%")

    print("\nBÁO CÁO PHÂN LOẠI")
    print("-" * 65)

    report = classification_report(
        y_test,
        y_pred,
        labels=model.classes_,
        output_dict=True,
        zero_division=0
    )

    print(
        classification_report(
            y_test,
            y_pred,
            labels=model.classes_,
            zero_division=0
        )
    )

    return y_pred, metrics, report


# =========================================================
# 9. LƯU BÁO CÁO PHÂN LOẠI
# =========================================================

def save_classification_report(report):

    with REPORT_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "class",
            "precision",
            "recall",
            "f1_score",
            "support"
        ])

        for label, values in report.items():

            if not isinstance(values, dict):
                continue

            writer.writerow([
                label,
                values.get("precision", 0),
                values.get("recall", 0),
                values.get("f1-score", 0),
                values.get("support", 0)
            ])

    print(f"Đã lưu báo cáo: {REPORT_PATH}")


# =========================================================
# 10. LƯU CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(y_test, y_pred, classes):

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=classes
    )

    with CONFUSION_MATRIX_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Actual / Predicted",
            *classes
        ])

        for label, row in zip(classes, matrix):

            writer.writerow([
                label,
                *row.tolist()
            ])

    print(f"Đã lưu Confusion Matrix: {CONFUSION_MATRIX_PATH}")


# =========================================================
# 11. LƯU CÁC CHỈ SỐ ĐÁNH GIÁ
# =========================================================

def save_metrics(metrics):

    with METRICS_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "metric",
            "score",
            "percentage"
        ])

        for metric_name, value in metrics.items():

            writer.writerow([
                metric_name,
                f"{value:.6f}",
                f"{value * 100:.2f}%"
            ])

    print(f"Đã lưu chỉ số đánh giá: {METRICS_PATH}")


# =========================================================
# 12. LƯU MÔ HÌNH
# =========================================================

def save_model(model):

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(f"\nĐã lưu mô hình: {MODEL_PATH}")


# =========================================================
# 13. LƯU THÔNG TIN QUÁ TRÌNH HUẤN LUYỆN
# =========================================================

def save_training_summary(
    X,
    X_train,
    X_test,
    class_counts,
    skipped_count,
    metrics
):

    lines = [
        "=" * 65,
        "BÁO CÁO HUẤN LUYỆN RANDOM FOREST",
        "=" * 65,
        "",
        f"Thời gian huấn luyện: {datetime.now():%d/%m/%Y %H:%M:%S}",
        f"Thư mục dữ liệu: {DATASET_DIR}",
        f"Thư mục kết quả: {RESULT_DIR}",
        "",
        "CẤU HÌNH:",
        f"- Kích thước ảnh: {IMG_SIZE}",
        f"- Số đặc trưng mỗi ảnh: {X.shape[1]}",
        f"- Số cây quyết định: {N_ESTIMATORS}",
        f"- Random State: {RANDOM_STATE}",
        f"- Tỷ lệ Train: {(1 - TEST_SIZE) * 100:.0f}%",
        f"- Tỷ lệ Test: {TEST_SIZE * 100:.0f}%",
        "",
        "THỐNG KÊ DỮ LIỆU:",
        f"- Tổng ảnh hợp lệ: {len(X)}",
        f"- Số ảnh Train: {len(X_train)}",
        f"- Số ảnh Test: {len(X_test)}",
        f"- Số ảnh bị bỏ qua: {skipped_count}",
        f"- Số lớp: {len(class_counts)}",
        "",
        "SỐ LƯỢNG ẢNH THEO LỚP:"
    ]

    for label, count in sorted(class_counts.items()):
        lines.append(f"- {label}: {count} ảnh")

    lines.extend([
        "",
        "KẾT QUẢ ĐÁNH GIÁ:"
    ])

    for metric_name, value in metrics.items():
        lines.append(
            f"- {metric_name}: {value * 100:.2f}%"
        )

    lines.extend([
        "",
        f"Mô hình: {MODEL_PATH.name}",
        "=" * 65
    ])

    SUMMARY_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print(f"Đã lưu báo cáo tổng hợp: {SUMMARY_PATH}")


# =========================================================
# 14. CHƯƠNG TRÌNH CHÍNH
# =========================================================

def main():

    print("=" * 65)
    print("HUẤN LUYỆN MÔ HÌNH RANDOM FOREST")
    print("PHÂN LOẠI TIỀN TỆ USD - CNY - VND")
    print("=" * 65)

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Bước 1: Đọc dữ liệu
    X, y, class_counts, skipped_count = load_dataset()

    # Bước 2: Kiểm tra dữ liệu
    validate_dataset(
        y,
        class_counts
    )

    # Bước 3: Chia Train/Test
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    # Bước 4: Huấn luyện mô hình
    rf_model = train_model(
        X_train,
        y_train
    )

    # Bước 5: Đánh giá mô hình
    y_pred, metrics, report = evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    # Bước 6: Lưu báo cáo
    save_classification_report(report)

    # Bước 7: Lưu Confusion Matrix
    save_confusion_matrix(
        y_test,
        y_pred,
        rf_model.classes_
    )

    # Bước 8: Lưu các chỉ số đánh giá
    save_metrics(metrics)

    # Bước 9: Lưu mô hình
    save_model(rf_model)

    # Bước 10: Lưu báo cáo tổng hợp
    save_training_summary(
        X,
        X_train,
        X_test,
        class_counts,
        skipped_count,
        metrics
    )

    print("\n" + "=" * 65)
    print("HOÀN TẤT HUẤN LUYỆN RANDOM FOREST")
    print("=" * 65)

    print(f"Accuracy: {metrics['Accuracy'] * 100:.2f}%")

    print(f"\nThư mục lưu kết quả:\n{RESULT_DIR}")

    print("\nCác file đã tạo:")
    for path in sorted(RESULT_DIR.iterdir()):
        if path.is_file():
            print(f"- {path.name}")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print(
            f"\nLỖI: {error}",
            file=sys.stderr
        )

        raise
