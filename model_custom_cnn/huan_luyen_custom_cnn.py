# import json
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# from tensorflow.keras import layers, models
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # =========================================================
# # 1. CAU HINH
# # =========================================================

# DATASET_DIR = Path(r"D:\CurrencyDataset")  # SUA THANH DUONG DAN THUC TE

# IMAGE_SIZE = (244, 244)
# BATCH_SIZE = 8
# EPOCHS = 30
# RANDOM_STATE = 42

# OUTPUT_DIR = Path("outputs")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# tf.keras.utils.set_random_seed(RANDOM_STATE)

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# =========================================================
# 1. CAU HINH
# =========================================================

# Thư mục chứa file huan_luyen_custom_cnn.py
SCRIPT_DIR = Path(__file__).resolve().parent

# Thư mục gốc của dự án Python-Final-Project
BASE_DIR = SCRIPT_DIR.parent

# Thư mục dữ liệu dùng chung với MobileNetV2
DATASET_DIR = BASE_DIR / "asset_demo_02_anh_cai_thien"

# Kích thước ảnh đầu vào
IMAGE_SIZE = (224, 224)

BATCH_SIZE = 8
EPOCHS = 30
RANDOM_STATE = 42

# Lưu kết quả riêng cho Custom CNN
OUTPUT_DIR = BASE_DIR / "ket_qua_custom_cnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cố định seed để tăng khả năng tái lập kết quả
tf.keras.utils.set_random_seed(RANDOM_STATE)

# Kiểm tra đường dẫn dữ liệu
if not DATASET_DIR.is_dir():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục dữ liệu: {DATASET_DIR}"
    )

print("Thư mục code:", SCRIPT_DIR)
print("Thư mục dự án:", BASE_DIR)
print("Thư mục dữ liệu:", DATASET_DIR)
print("Thư mục lưu kết quả:", OUTPUT_DIR)


# =========================================================
# 2. DOC DUONG DAN ANH VA NHAN
# Cau truc: CurrencyDataset / Currency / Denomination / Image
# =========================================================

valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_paths = []
labels = []

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Khong tim thay thu muc dataset: {DATASET_DIR}"
    )

for image_path in DATASET_DIR.rglob("*"):
    if image_path.is_file() and image_path.suffix.lower() in valid_extensions:
        # Ten thu muc chua anh la nhan menh gia
        label = image_path.parent.name.lower()

        image_paths.append(str(image_path))
        labels.append(label)

if not image_paths:
    raise ValueError("Khong tim thay anh trong dataset!")

df = pd.DataFrame({
    "image_path": image_paths,
    "label": labels
})

print("\n========== THONG TIN DATASET ==========")
print("Tong so anh:", len(df))
print("So lop:", df["label"].nunique())
print("\nSo luong anh moi lop:")
print(df["label"].value_counts().sort_index())

# Ma hoa nhan thanh so nguyen
class_names = sorted(df["label"].unique().tolist())
class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}

df["label_id"] = df["label"].map(class_to_index)

# Luu danh sach lop de dung khi du doan sau nay
with open(OUTPUT_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=4)

# =========================================================
# 3. CHIA TRAIN / VALIDATION / TEST = 70 / 15 / 15
# =========================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["label_id"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["label_id"]
)

print("\n========== CHIA DU LIEU ==========")
print(f"Train      : {len(train_df)} anh")
print(f"Validation : {len(val_df)} anh")
print(f"Test       : {len(test_df)} anh")

# =========================================================
# 4. TAO PIPELINE DOC ANH
# =========================================================

def load_image(image_path, label):
    image_bytes = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    # Chuan hoa pixel tu [0, 255] ve [0, 1]
    image = image / 255.0

    return image, label


def create_dataset(dataframe, training=False):
    paths = dataframe["image_path"].to_numpy()
    targets = dataframe["label_id"].to_numpy(dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, targets))

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(dataframe), 1000),
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=2
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset


train_ds = create_dataset(train_df, training=True)
val_ds = create_dataset(val_df)
test_ds = create_dataset(test_df)

# =========================================================
# 5. XAY DUNG CUSTOM CNN
# =========================================================

NUM_CLASSES = len(class_names)

model = models.Sequential([
    layers.Input(shape=(244, 244, 3)),

    # Khoi CNN 1
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=2),

    # Khoi CNN 2
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=2),

    # Khoi CNN 3
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=2),

    # Khoi CNN 4
    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=2),

    # Phan phan loai
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n========== KIEN TRUC MO HINH ==========")
model.summary()

# =========================================================
# 6. CALLBACKS
# =========================================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        filepath=str(OUTPUT_DIR / "best_custom_cnn.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# =========================================================
# 7. HUAN LUYEN
# =========================================================

print("\n========== BAT DAU HUAN LUYEN ==========")

history = model.fit(
    train_ds,
    validation_data=val_ds,epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Luu mo hinh voi trong so tot nhat
model.save(OUTPUT_DIR / "custom_cnn_final.keras")

# =========================================================
# 8. DANH GIA TREN TAP TEST
# =========================================================

print("\n========== DANH GIA TEST ==========")

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4%}")

# Lay nhan that va du doan
y_true = np.concatenate([
    labels_batch.numpy()
    for _, labels_batch in test_ds
])

y_prob = model.predict(test_ds, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    zero_division=0,
    digits=4
))

# Luu ket qua phan loai
report = classification_report(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    zero_division=0,
    output_dict=True
)

pd.DataFrame(report).transpose().to_csv(
    OUTPUT_DIR / "classification_report.csv",
    encoding="utf-8-sig"
)

# =========================================================
# 9. VE CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

fig_size = max(10, min(20, NUM_CLASSES * 0.7))

fig, ax = plt.subplots(figsize=(fig_size, fig_size))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=False
)

plt.title("Confusion Matrix - Custom CNN")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

# =========================================================
# 10. VE DO THI HUAN LUYEN
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "training_history.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()

print("\n========== HOAN TAT ==========")
print("Mo hinh: outputs/best_custom_cnn.keras")
print("Mo hinh cuoi: outputs/custom_cnn_final.keras")
print("Danh sach lop: outputs/class_names.json")
print("Bao cao: outputs/classification_report.csv")
print("Confusion matrix: outputs/confusion_matrix.png")
print("Bieu do huan luyen: outputs/training_history.png")