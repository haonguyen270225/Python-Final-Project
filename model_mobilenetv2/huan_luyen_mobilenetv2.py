"""
HUẤN LUYỆN MOBILENETV2 PHÂN LOẠI LOẠI TIỀN VÀ MỆNH GIÁ

Cấu trúc dữ liệu đầu vào:
asset_demo_02_anh_cai_thien/
    VND/
        10000/
            anh1.jpg
        20000/
            anh2.jpg
    USD/
        1/
            anh3.jpg
    CNY/
        5/
            anh4.jpg

Cài thư viện:
    pip install tensorflow scikit-learn matplotlib numpy

Chạy:
    python huan_luyen_mobilenetv2.py
"""

from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ============================================================
# 1. CẤU HÌNH
# ============================================================

# DATA_DIR = Path("asset_demo_02_anh_cai_thien")
# OUTPUT_DIR = Path("ket_qua_mobilenetv2")
from pathlib import Path

# Thư mục chứa file huan_luyen_mobilenetv2.py
BASE_DIR = Path(__file__).resolve().parent

# Thư mục dữ liệu nằm bên ngoài model_mobilenetv2
DATA_DIR = BASE_DIR.parent / "asset_finally"

# Thư mục lưu kết quả nằm bên ngoài model_mobilenetv2
OUTPUT_DIR = BASE_DIR.parent / "ket_qua_mobilenetv2"

# Tạo thư mục đầu ra nếu chưa tồn tại
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Kiểm tra tồn tại thư mục dữ liệu
print("Thư mục dữ liệu:", DATA_DIR.resolve())
print("Thư mục đầu ra:", OUTPUT_DIR.resolve())

print("Thư mục dữ liệu tồn tại:", DATA_DIR.exists())


OUTPUT_DIR = OUTPUT_DIR.resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS_HEAD = 10
EPOCHS_FINE_TUNE = 5

# Tỷ lệ chia dữ liệu: 70% train, 15% validation, 15% test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

np.random.seed(SEED)
tf.random.set_seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. QUÉT ẢNH VÀ TẠO NHÃN TỪ CẤU TRÚC THƯ MỤC
# ============================================================

def scan_images(data_dir: Path):
    """
    Đọc cấu trúc loại tiền/mệnh giá và tạo nhãn kết hợp:
    VND/10000 -> VND_10000
    USD/1     -> USD_1
    CNY/5     -> CNY_5
    """
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {data_dir.resolve()}\n"
            "Hãy đặt file Python cạnh thư mục ảnh hoặc sửa DATA_DIR."
        )

    image_paths = []
    class_names = []

    # Lấy các thư mục loại tiền (VND, USD, CNY)
    currency_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    for currency_dir in currency_dirs:
        # Các thư mục mệnh giá bên trong loại tiền
        denomination_dirs = sorted(p for p in currency_dir.iterdir() if p.is_dir())

        for denomination_dir in denomination_dirs:
            class_name = f"{currency_dir.name}_{denomination_dir.name}"

            files = sorted(
                p for p in denomination_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )

            if not files:
                print(f"[CẢNH BÁO] Không có ảnh trong: {denomination_dir}")

            for file_path in files:
                image_paths.append(str(file_path))
                class_names.append(class_name)

    if not image_paths:
        raise ValueError(
            f"Không tìm thấy ảnh trong {data_dir.resolve()}. "
            "Kiểm tra lại cấu trúc thư mục và phần mở rộng ảnh."
        )

    return np.array(image_paths), np.array(class_names)


image_paths, labels_text = scan_images(DATA_DIR)

class_names = sorted(set(labels_text.tolist()))
class_to_id = {name: index for index, name in enumerate(class_names)}
labels = np.array([class_to_id[label] for label in labels_text], dtype=np.int32)

print("\n========== THỐNG KÊ DỮ LIỆU ==========")
print(f"Tổng số ảnh: {len(image_paths)}")
print(f"Số lớp kết hợp loại tiền/mệnh giá: {len(class_names)}")
for class_name in class_names:
    print(f"  {class_name}: {int(np.sum(labels_text == class_name))} ảnh")
print("======================================\n")

if len(class_names) < 2:
    raise ValueError("Cần ít nhất 2 lớp để huấn luyện mô hình phân loại.")

# Cần tối thiểu vài ảnh mỗi lớp để chia train/validation/test có ý nghĩa.
class_counts = {name: int(np.sum(labels_text == name)) for name in class_names}
rare_classes = {name: count for name, count in class_counts.items() if count < 7}
if rare_classes:
    raise ValueError(
        "Một số lớp có ít hơn 7 ảnh nên khó chia train/validation/test "
        f"một cách đáng tin cậy: {rare_classes}. Hãy bổ sung ảnh cho các lớp này."
    )


# ============================================================
# 3. CHIA TRAIN / VALIDATION / TEST CÓ PHÂN TẦNG
# ============================================================

# Chia 70% train và 30% phần còn lại
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=(1.0 - TRAIN_RATIO),
    random_state=SEED,
    stratify=labels,
)

# Chia phần còn lại thành validation và test, mỗi phần 15% tổng dữ liệu
relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=relative_test_size,
    random_state=SEED,
    stratify=temp_labels,
)

print("Số ảnh train      :", len(train_paths))
print("Số ảnh validation :", len(val_paths))
print("Số ảnh test       :", len(test_paths))


# ============================================================
# 4. ĐỌC ẢNH, RESIZE VÀ CHUẨN HÓA CHO MOBILENETV2
# ============================================================

def load_and_preprocess_image(path, label):
    # Đọc ảnh RGB; decode_image hỗ trợ JPG/PNG và một số định dạng phổ biến.
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])

    # Đưa ảnh về 224 x 224
    image = tf.image.resize(image, IMAGE_SIZE)

    # preprocess_input chuyển pixel về khoảng xấp xỉ [-1, 1].
    # Không cần biết trước ảnh lưu ở pixel 0-255 hay ảnh được xử lý ở bước nào:
    # khi đọc JPG/PNG thông thường, pixel sẽ được giải mã về thang 0-255.
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, label


AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(paths, labels_array, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels_array))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset


train_ds = make_dataset(train_paths, train_labels, training=True)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)


# ============================================================
# 5. TẠO MÔ HÌNH MOBILENETV2 (TRANSFER LEARNING)
# ============================================================

# Tải trọng số ImageNet; lần chạy đầu cần kết nối Internet.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)

# Giai đoạn đầu giữ nguyên mạng nền, chỉ học lớp phân loại mới
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# 6. HUẤN LUYỆN LỚP PHÂN LOẠI
# ============================================================

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(OUTPUT_DIR / "mobilenetv2_best.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
]

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks,
)


# ============================================================
# 7. FINE-TUNING NHẸ CÁC LỚP CUỐI (TÙY CHỌN)
# ============================================================

# Mở khóa một phần các lớp cuối để thích nghi với ảnh tiền giấy.
# BatchNormalization vẫn giữ ở chế độ không huấn luyện để ổn định.
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=callbacks,
)


# ============================================================
# 8. ĐÁNH GIÁ TRÊN TẬP TEST CHƯA DÙNG ĐỂ HUẤN LUYỆN
# ============================================================

# Nạp mô hình tốt nhất được lưu theo val_accuracy
best_model_path = OUTPUT_DIR / "mobilenetv2_best.keras"
if best_model_path.exists():
    model = tf.keras.models.load_model(best_model_path)

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"\nTest loss     : {test_loss:.4f}")
print(f"Test accuracy : {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

probabilities = model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)

print("\n========== CLASSIFICATION REPORT ==========")
print(
    classification_report(
        test_labels,
        predicted_labels,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    )
)

cm = confusion_matrix(
    test_labels,
    predicted_labels,
    labels=np.arange(len(class_names))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.7), max(6, len(class_names) * 0.6)))
display.plot(ax=ax, xticks_rotation=90, values_format="d", colorbar=False)
plt.title("Ma trận nhầm lẫn - MobileNetV2")
plt.tight_layout()

confusion_path = OUTPUT_DIR / "confusion_matrix.png"

fig.savefig(
    str(confusion_path),
    dpi=200,
    format="png",
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Đã lưu ma trận nhầm lẫn tại:", confusion_path)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.show()


# ============================================================
# 9. LƯU MÔ HÌNH VÀ DANH SÁCH LỚP
# ============================================================

final_model_path = OUTPUT_DIR / "mobilenetv2_currency_denomination.keras"
model.save(final_model_path)

with open(OUTPUT_DIR / "class_names.json", "w", encoding="utf-8") as file:
    json.dump(class_names, file, ensure_ascii=False, indent=2)

print("\nĐã lưu mô hình:", final_model_path.resolve())
print("Đã lưu danh sách lớp:", (OUTPUT_DIR / "class_names.json").resolve())
print("Đã lưu ma trận nhầm lẫn:", (OUTPUT_DIR / "confusion_matrix.png").resolve())


# ============================================================
# 10. HÀM DỰ ĐOÁN ẢNH MỚI
# ============================================================

def du_doan_anh(image_path, model_path=final_model_path):
    """
    Dự đoán loại tiền và mệnh giá của một ảnh mới.
    Ví dụ kết quả lớp: VND_10000 -> tiền VND, mệnh giá 10000.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    loaded_model = tf.keras.models.load_model(model_path)

    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)

    probs = loaded_model.predict(image, verbose=0)[0]
    best_idx = int(np.argmax(probs))
    class_name = class_names[best_idx]
    confidence = float(probs[best_idx])

    currency, denomination = class_name.split("_", 1)

    print("\n========== KẾT QUẢ DỰ ĐOÁN ==========")
    print("Ảnh              :", image_path.name)
    print("Loại tiền        :", currency)
    print("Mệnh giá         :", denomination)
    print(f"Độ tin cậy mô hình: {confidence * 100:.2f}%")
    print("=====================================")

    return {
        "currency": currency,
        "denomination": denomination,
        "confidence": confidence,
        "class_name": class_name,
    }


# Để thử dự đoán ảnh mới, bỏ dấu # và sửa đường dẫn:
# du_doan_anh("C:/Users/tranh/OneDrive/Desktop/Python-Final-Project/asset_demo_02_anh_cai_thien/ChineseYuan_Rename/50_cny.jpg")
