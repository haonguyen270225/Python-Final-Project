
import json
import random
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Thư mục dữ liệu ảnh
DATA_DIR = PROJECT_DIR / "asset_finally"

# Thư mục chứa class_mapping.json
MAPPING_DIR = PROJECT_DIR / "asset_restnet"
MAPPING_PATH = MAPPING_DIR / "class_mapping.json"

# Thư mục lưu kết quả huấn luyện
OUTPUT_DIR = PROJECT_DIR / "ket_qua_resnet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CẤU HÌNH HUẤN LUYỆN
# ============================================================

IMG_SIZE = (224, 224)

BATCH_SIZE = 16
EPOCHS = 30

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

NUM_WORKERS = 0
USE_PRETRAINED = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

CURRENCY_MAPPING = {
    "americandollar": "USD",
    "chineseyuan": "CNY",
    "vietnamesedong": "VND"
}

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


# ============================================================
# 3. KHỞI TẠO RANDOM SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Tăng tính ổn định kết quả giữa các lần chạy
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 4. CHUẨN HÓA TÊN THƯ MỤC
# ============================================================

def normalize_currency(currency_name):

    key = currency_name.strip().casefold()

    return CURRENCY_MAPPING.get(key)


def normalize_denomination(denomination_name):

    import re

    name = denomination_name.strip()

    match = re.fullmatch(
        r"(\d+)\s*(?:dollar|doolar)?",
        name,
        flags=re.IGNORECASE
    )

    if not match:
        return None

    return str(int(match.group(1)))


# ============================================================
# 5. ĐỌC CLASS MAPPING
# ============================================================

def load_class_mapping():

    if not MAPPING_PATH.exists():

        raise FileNotFoundError(
            f"Không tìm thấy file mapping: {MAPPING_PATH}\n"
            "Hãy chạy cau_hinh_du_lieu_resnet.py trước."
        )

    with open(
        MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        mapping = json.load(file)

    currency_classes = mapping["currency_classes"]

    denomination_classes = mapping["denomination_classes"]

    denomination_display = mapping.get(
        "denomination_display",
        {}
    )

    if not currency_classes:
        raise ValueError("Danh sách currency_classes đang rỗng.")

    if not denomination_classes:
        raise ValueError("Danh sách denomination_classes đang rỗng.")

    currency_to_idx = {
        name: idx
        for idx, name in enumerate(currency_classes)
    }

    denomination_to_idx = {
        name: idx
        for idx, name in enumerate(denomination_classes)
    }

    print("\nĐã đọc class_mapping.json")

    print("Các lớp tiền tệ:", currency_classes)

    print("Số lớp mệnh giá:", len(denomination_classes))

    return (
        currency_classes,
        denomination_classes,
        denomination_display,
        currency_to_idx,
        denomination_to_idx
    )


# ============================================================
# 6. QUÉT DỮ LIỆU ẢNH
# ============================================================

def scan_dataset(
    currency_to_idx,
    denomination_to_idx
):

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {DATA_DIR}"
        )

    samples = []

    skipped_count = 0

    print("\nĐang quét dữ liệu:", DATA_DIR)

    for currency_dir in sorted(DATA_DIR.iterdir()):

        if not currency_dir.is_dir():
            continue

        currency_code = normalize_currency(
            currency_dir.name
        )

        if currency_code is None:

            print(
                f"[BỎ QUA] Loại tiền không hỗ trợ: "
                f"{currency_dir.name}"
            )

            continue

        if currency_code not in currency_to_idx:

            print(
                f"[BỎ QUA] Loại tiền không có trong mapping: "
                f"{currency_code}"
            )

            continue

        print(f"\n[{currency_code}] {currency_dir.name}")

        for denomination_dir in sorted(currency_dir.iterdir()):

            if not denomination_dir.is_dir():
                continue

            # Bỏ qua Test Set
            if denomination_dir.name.strip().casefold() == "test set":

                print("  [BỎ QUA] Test Set")

                continue

            denomination_value = normalize_denomination(
                denomination_dir.name
            )

            if denomination_value is None:

                print(
                    f"  [BỎ QUA] Thư mục không hợp lệ: "
                    f"{denomination_dir.name}"
                )

                continue

            # Tạo nhãn giống file cấu hình dữ liệu
            denomination_label = (
                f"{currency_code}_{denomination_value}"
            )

            # Kiểm tra nhãn có trong mapping không
            if denomination_label not in denomination_to_idx:

                print(
                    f"  [BỎ QUA] Nhãn không có trong mapping: "
                    f"{denomination_label}"
                )

                continue

            image_count = 0

            # Quét cả thư mục con
            for image_path in sorted(
                denomination_dir.rglob("*")
            ):

                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                samples.append({

                    "path": image_path,

                    "currency": currency_to_idx[currency_code],

                    "denomination": denomination_to_idx[
                        denomination_label
                    ]

                })

                image_count += 1

            print(
                f"  {denomination_dir.name:15} "
                f"-> {denomination_label:12} "
                f"| {image_count} ảnh"
            )

    if len(samples) == 0:

        raise RuntimeError(
            "Không tìm thấy ảnh hợp lệ.\n"
            "Hãy kiểm tra asset_finally và class_mapping.json."
        )

    return samples


# ============================================================
# 7. CHIA TRAIN / VALIDATION
# ============================================================

def stratified_split(samples):

    groups = defaultdict(list)

    # Gom ảnh theo từng lớp mệnh giá
    for index, sample in enumerate(samples):

        key = (
            sample["currency"],
            sample["denomination"]
        )

        groups[key].append(index)

    rng = random.Random(RANDOM_SEED)

    train_indices = []
    val_indices = []

    for indices in groups.values():

        rng.shuffle(indices)

        # Lớp chỉ có 1 ảnh thì giữ trong Train
        if len(indices) < 2:

            train_indices.extend(indices)

            continue

        train_count = int(
            len(indices) * TRAIN_RATIO
        )

        train_count = max(
            1,
            min(train_count, len(indices) - 1)
        )

        train_indices.extend(
            indices[:train_count]
        )

        val_indices.extend(
            indices[train_count:]
        )

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


# ============================================================
# 8. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(IMG_SIZE),

    transforms.RandomRotation(
        degrees=7
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        NORMALIZE_MEAN,
        NORMALIZE_STD
    )

])


val_transform = transforms.Compose([

    transforms.Resize(IMG_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        NORMALIZE_MEAN,
        NORMALIZE_STD
    )

])


# ============================================================
# 9. CUSTOM DATASET
# ============================================================

class MoneyDataset(Dataset):

    def __init__(
        self,
        samples,
        transform=None
    ):

        self.samples = samples
        self.transform = transform

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image_path = sample["path"]

        try:

            with Image.open(image_path) as image:

                image = image.convert("RGB")

                if self.transform:

                    image = self.transform(image)

        except (OSError, UnidentifiedImageError) as error:

            raise RuntimeError(
                f"Không thể đọc ảnh: {image_path}"
            ) from error

        currency_label = torch.tensor(
            sample["currency"],
            dtype=torch.long
        )

        denomination_label = torch.tensor(
            sample["denomination"],
            dtype=torch.long
        )

        return (
            image,
            currency_label,
            denomination_label
        )


# ============================================================
# 10. XÂY DỰNG MÔ HÌNH RESNET18
# ============================================================

class ResNetMultiTask(nn.Module):

    def __init__(
        self,
        num_currencies,
        num_denominations,
        pretrained=True
    ):

        super().__init__()

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = models.resnet18(
            weights=weights
        )

        input_features = self.backbone.fc.in_features

        # Thay lớp phân loại mặc định
        self.backbone.fc = nn.Identity()

        # Đầu ra 1: loại tiền
        self.currency_head = nn.Linear(
            input_features,
            num_currencies
        )

        # Đầu ra 2: mệnh giá
        self.denomination_head = nn.Linear(
            input_features,
            num_denominations
        )

    def forward(self, x):

        features = self.backbone(x)

        currency_output = self.currency_head(
            features
        )

        denomination_output = self.denomination_head(
            features
        )

        return (
            currency_output,
            denomination_output
        )


# ============================================================
# 11. HÀM HUẤN LUYỆN MỘT EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0

    correct_currency = 0
    correct_denomination = 0

    total = 0

    for images, currency_labels, denomination_labels in loader:

        images = images.to(DEVICE)

        currency_labels = currency_labels.to(DEVICE)

        denomination_labels = denomination_labels.to(DEVICE)

        optimizer.zero_grad()

        currency_output, denomination_output = model(images)

        currency_loss = criterion(
            currency_output,
            currency_labels
        )

        denomination_loss = criterion(
            denomination_output,
            denomination_labels
        )

        # Tổng loss của hai nhiệm vụ
        loss = currency_loss + denomination_loss

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size

        currency_predictions = currency_output.argmax(dim=1)

        denomination_predictions = denomination_output.argmax(dim=1)

        correct_currency += (
            currency_predictions == currency_labels
        ).sum().item()

        correct_denomination += (
            denomination_predictions == denomination_labels
        ).sum().item()

        total += batch_size

    return {

        "loss": total_loss / total,

        "currency_accuracy": correct_currency / total,

        "denomination_accuracy": correct_denomination / total

    }


# ============================================================
# 12. ĐÁNH GIÁ VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    correct_currency = 0
    correct_denomination = 0

    total = 0

    with torch.no_grad():

        for images, currency_labels, denomination_labels in loader:

            images = images.to(DEVICE)

            currency_labels = currency_labels.to(DEVICE)

            denomination_labels = denomination_labels.to(DEVICE)

            currency_output, denomination_output = model(images)

            currency_loss = criterion(
                currency_output,
                currency_labels
            )

            denomination_loss = criterion(
                denomination_output,
                denomination_labels
            )

            loss = currency_loss + denomination_loss

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size

            currency_predictions = currency_output.argmax(dim=1)

            denomination_predictions = denomination_output.argmax(dim=1)

            correct_currency += (
                currency_predictions == currency_labels
            ).sum().item()

            correct_denomination += (
                denomination_predictions == denomination_labels
            ).sum().item()

            total += batch_size

    if total == 0:

        raise RuntimeError(
            "Validation không có dữ liệu để đánh giá."
        )

    return {

        "loss": total_loss / total,

        "currency_accuracy": correct_currency / total,

        "denomination_accuracy": correct_denomination / total

    }


# ============================================================
# 13. LƯU CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    currency_classes,
    denomination_classes,
    filename
):

    checkpoint = {

        "epoch": epoch,

        "model_name": "ResNet18",

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "metrics": metrics,

        "currency_classes": currency_classes,

        "denomination_classes": denomination_classes,

        "img_size": IMG_SIZE

    }

    torch.save(
        checkpoint,
        OUTPUT_DIR / filename
    )


# ============================================================
# 14. THỐNG KÊ DỮ LIỆU
# ============================================================

def print_dataset_statistics(
    samples,
    currency_classes,
    denomination_classes
):

    statistics = defaultdict(int)

    for sample in samples:

        key = (
            sample["currency"],
            sample["denomination"]
        )

        statistics[key] += 1

    print("\nTHỐNG KÊ DỮ LIỆU")

    print("-" * 65)

    for (currency_idx, denomination_idx), count in sorted(
        statistics.items()
    ):

        currency_name = currency_classes[currency_idx]

        denomination_name = denomination_classes[denomination_idx]

        print(
            f"{currency_name:8} | "
            f"{denomination_name:15} | "
            f"{count:5} ảnh"
        )

    print("-" * 65)


# ============================================================
# 15. LƯU LỊCH SỬ HUẤN LUYỆN
# ============================================================

def save_training_history(history):

    if not history:
        return

    history_path = OUTPUT_DIR / "training_history.csv"

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys()
        )

        writer.writeheader()

        writer.writerows(history)

    print(
        f"\nĐã lưu lịch sử huấn luyện: {history_path}"
    )


# ============================================================
# 16. CHƯƠNG TRÌNH HUẤN LUYỆN CHÍNH
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    print("=" * 70)
    print("HUẤN LUYỆN MÔ HÌNH RESNET18")
    print("=" * 70)

    print("Thiết bị:", DEVICE)

    print("Thư mục dự án:", PROJECT_DIR)

    print("Thư mục dữ liệu:", DATA_DIR)

    print("File mapping:", MAPPING_PATH)

    print("Thư mục kết quả:", OUTPUT_DIR)

    # --------------------------------------------------------
    # BƯỚC 1: ĐỌC MAPPING
    # --------------------------------------------------------

    print("\n[1] Đang đọc class_mapping.json...")

    (
        currency_classes,
        denomination_classes,
        denomination_display,
        currency_to_idx,
        denomination_to_idx
    ) = load_class_mapping()

    num_currencies = len(currency_classes)

    num_denominations = len(denomination_classes)

    print("\nSố lớp tiền tệ:", num_currencies)

    print("Số lớp mệnh giá:", num_denominations)

    # --------------------------------------------------------
    # BƯỚC 2: QUÉT DỮ LIỆU
    # --------------------------------------------------------

    print("\n[2] Đang quét dữ liệu ảnh...")

    samples = scan_dataset(
        currency_to_idx,
        denomination_to_idx
    )

    print("\nTổng số ảnh tìm thấy:", len(samples))

    # --------------------------------------------------------
    # BƯỚC 3: THỐNG KÊ DỮ LIỆU
    # --------------------------------------------------------

    print("\n[3] Thống kê số lượng ảnh...")

    print_dataset_statistics(
        samples,
        currency_classes,
        denomination_classes
    )

    # --------------------------------------------------------
    # BƯỚC 4: CHIA TRAIN / VALIDATION
    # --------------------------------------------------------

    print("\n[4] Đang chia Train / Validation...")

    train_indices, val_indices = stratified_split(
        samples
    )

    if len(train_indices) == 0:

        raise RuntimeError(
            "Không có dữ liệu Train."
        )

    if len(val_indices) == 0:

        raise RuntimeError(
            "Không có dữ liệu Validation. "
            "Hãy kiểm tra số ảnh trong từng lớp."
        )

    train_samples = [
        samples[index]
        for index in train_indices
    ]

    val_samples = [
        samples[index]
        for index in val_indices
    ]

    print("Tổng số ảnh:", len(samples))

    print("Số ảnh Train:", len(train_samples))

    print("Số ảnh Validation:", len(val_samples))

    # --------------------------------------------------------
    # BƯỚC 5: KHỞI TẠO DATASET
    # --------------------------------------------------------

    print("\n[5] Đang khởi tạo Dataset...")

    train_dataset = MoneyDataset(
        train_samples,
        transform=train_transform
    )

    val_dataset = MoneyDataset(
        val_samples,
        transform=val_transform
    )

    print("Train Dataset:", len(train_dataset))

    print("Validation Dataset:", len(val_dataset))

    # --------------------------------------------------------
    # BƯỚC 6: KHỞI TẠO DATALOADER
    # --------------------------------------------------------

    print("\n[6] Đang khởi tạo DataLoader...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    print("Số batch Train:", len(train_loader))

    print("Số batch Validation:", len(val_loader))

    # --------------------------------------------------------
    # BƯỚC 7: KIỂM TRA MỘT BATCH
    # --------------------------------------------------------

    print("\n[7] Kiểm tra dữ liệu đầu vào...")

    images, currency_labels, denomination_labels = next(
        iter(train_loader)
    )

    print("Kích thước ảnh:", images.shape)

    print("Nhãn loại tiền:", currency_labels.shape)

    print("Nhãn mệnh giá:", denomination_labels.shape)

    # --------------------------------------------------------
    # BƯỚC 8: KHỞI TẠO MÔ HÌNH
    # --------------------------------------------------------

    print("\n[8] Đang khởi tạo ResNet18...")

    model = ResNetMultiTask(
        num_currencies=num_currencies,
        num_denominations=num_denominations,
        pretrained=USE_PRETRAINED
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    # --------------------------------------------------------
    # BƯỚC 9: LƯU MAPPING VÀO THƯ MỤC KẾT QUẢ
    # --------------------------------------------------------

    with open(
        MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as source_file:

        mapping = json.load(source_file)

    with open(
        OUTPUT_DIR / "class_mapping.json",
        "w",
        encoding="utf-8"
    ) as output_file:

        json.dump(
            mapping,
            output_file,
            ensure_ascii=False,
            indent=4
        )

    # --------------------------------------------------------
    # BƯỚC 10: HUẤN LUYỆN
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BẮT ĐẦU HUẤN LUYỆN")
    print("=" * 70)

    best_accuracy = -1.0

    history = []

    for epoch in range(EPOCHS):

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion
        )

        # Accuracy trung bình của hai nhiệm vụ
        val_accuracy = (
            val_metrics["currency_accuracy"]
            + val_metrics["denomination_accuracy"]
        ) / 2

        scheduler.step(val_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]

        record = {

            "epoch": epoch + 1,

            "learning_rate": current_lr,

            "train_loss": train_metrics["loss"],

            "train_currency_accuracy": (
                train_metrics["currency_accuracy"]
            ),

            "train_denomination_accuracy": (
                train_metrics["denomination_accuracy"]
            ),

            "val_loss": val_metrics["loss"],

            "val_currency_accuracy": (
                val_metrics["currency_accuracy"]
            ),

            "val_denomination_accuracy": (
                val_metrics["denomination_accuracy"]
            ),

            "val_average_accuracy": val_accuracy

        }

        history.append(record)

        print("\n" + "-" * 70)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}]"
        )

        print(
            f"Train Loss: {train_metrics['loss']:.4f}"
        )

        print(
            f"Train Currency Accuracy: "
            f"{train_metrics['currency_accuracy'] * 100:.2f}%"
        )

        print(
            f"Train Denomination Accuracy: "
            f"{train_metrics['denomination_accuracy'] * 100:.2f}%"
        )

        print(
            f"Validation Loss: {val_metrics['loss']:.4f}"
        )

        print(
            f"Validation Currency Accuracy: "
            f"{val_metrics['currency_accuracy'] * 100:.2f}%"
        )

        print(
            f"Validation Denomination Accuracy: "
            f"{val_metrics['denomination_accuracy'] * 100:.2f}%"
        )

        print(
            f"Validation Average Accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Learning Rate: {current_lr:.7f}"
        )

        # ----------------------------------------------------
        # LƯU MODEL TỐT NHẤT
        # ----------------------------------------------------

        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_metrics,
                currency_classes,
                denomination_classes,
                "best_model.pth"
            )

            print("Đã lưu best_model.pth")

        # ----------------------------------------------------
        # LƯU MODEL CUỐI CÙNG CỦA EPOCH
        # ----------------------------------------------------

        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            val_metrics,
            currency_classes,
            denomination_classes,
            "last_model.pth"
        )

        # Cập nhật CSV sau mỗi epoch
        save_training_history(history)

    # --------------------------------------------------------
    # BƯỚC 11: LƯU TỔNG KẾT
    # --------------------------------------------------------

    final_metrics = {

        "model": "ResNet18",

        "device": str(DEVICE),

        "epochs": EPOCHS,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "train_ratio": TRAIN_RATIO,

        "random_seed": RANDOM_SEED,

        "num_currencies": num_currencies,

        "num_denominations": num_denominations,

        "total_images": len(samples),

        "train_images": len(train_samples),

        "validation_images": len(val_samples),

        "best_validation_average_accuracy": best_accuracy,

        "best_validation_average_accuracy_percent": (
            best_accuracy * 100
        ),

        "currency_classes": currency_classes,

        "denomination_classes": denomination_classes

    }

    with open(
        OUTPUT_DIR / "training_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_metrics,
            file,
            ensure_ascii=False,
            indent=4
        )

    # --------------------------------------------------------
    # HOÀN TẤT
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("HOÀN TẤT HUẤN LUYỆN RESNET18")

    print("=" * 70)

    print(
        f"Validation Accuracy tốt nhất: "
        f"{best_accuracy * 100:.2f}%"
    )

    print("Thư mục kết quả:", OUTPUT_DIR)

    print("\nCác file đã lưu:")

    for file_path in sorted(OUTPUT_DIR.iterdir()):

        if file_path.is_file():

            print("-", file_path.name)

    print("=" * 70)


# ============================================================
# 17. CHẠY CHƯƠNG TRÌNH
# ============================================================

if __name__ == "__main__":

    main()
