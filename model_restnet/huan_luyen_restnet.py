
import json
import random
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "asset"
MAPPING_DIR = PROJECT_DIR / "asset_restnet"
MAPPING_PATH = MAPPING_DIR / "class_mapping.json"

# Thư mục kết quả nằm ngoài model_resnet
OUTPUT_DIR = PROJECT_DIR / "ket_qua_resnet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CẤU HÌNH HUẤN LUYỆN
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
NUM_WORKERS = 0

USE_PRETRAINED = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. CẤU HÌNH NHÃN
# ============================================================

CURRENCY_MAPPING = {
    "americandollar": "USD",
    "chineseyuan": "CNY",
    "vietnamesedong": "VND"
}

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# 4. KHỞI TẠO RANDOM SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 5. CHUẨN HÓA TÊN MỆNH GIÁ
# ============================================================

def normalize_denomination(folder_name):
    import re

    name = folder_name.strip()

    match = re.fullmatch(
        r"(\d+)\s*(?:dollar|doolar)?",
        name,
        flags=re.IGNORECASE
    )

    if not match:
        return None

    return str(int(match.group(1)))


# ============================================================
# 6. ĐỌC CLASS MAPPING
# ============================================================

def load_class_mapping():

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file mapping: {MAPPING_PATH}"
        )

    with open(MAPPING_PATH, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    currency_classes = mapping["currency_classes"]
    denomination_classes = mapping["denomination_classes"]

    currency_to_idx = {
        name: index
        for index, name in enumerate(currency_classes)
    }

    denomination_to_idx = {
        name: index
        for index, name in enumerate(denomination_classes)
    }

    return (
        currency_classes,
        denomination_classes,
        currency_to_idx,
        denomination_to_idx
    )


# ============================================================
# 7. QUÉT DỮ LIỆU ẢNH
# ============================================================

def scan_dataset(
    currency_to_idx,
    denomination_to_idx
):

    samples = []

    for currency_folder in DATA_DIR.iterdir():

        if not currency_folder.is_dir():
            continue

        currency_code = CURRENCY_MAPPING.get(
            currency_folder.name.casefold()
        )

        if currency_code is None:
            continue

        currency_idx = currency_to_idx.get(currency_code)

        if currency_idx is None:
            continue

        for denomination_folder in currency_folder.iterdir():

            if not denomination_folder.is_dir():
                continue

            # Bỏ qua thư mục Test Set
            if denomination_folder.name.strip().casefold() == "test set":
                continue

            denomination_value = normalize_denomination(
                denomination_folder.name
            )

            if denomination_value is None:
                continue

            denomination_label = (
                f"{currency_code}_{denomination_value}"
            )

            denomination_idx = denomination_to_idx.get(
                denomination_label
            )

            if denomination_idx is None:
                continue

            for image_path in denomination_folder.rglob("*"):

                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                samples.append({
                    "path": image_path,
                    "currency": currency_idx,
                    "denomination": denomination_idx
                })

    return samples


# ============================================================
# 8. CHIA TRAIN / VALIDATION
# ============================================================

def stratified_split(samples):

    groups = {}

    for sample in samples:

        key = (
            sample["currency"],
            sample["denomination"]
        )

        groups.setdefault(key, []).append(sample)

    train_samples = []
    val_samples = []

    rng = random.Random(RANDOM_SEED)

    for group in groups.values():

        rng.shuffle(group)

        if len(group) == 1:
            train_samples.extend(group)
            continue

        split_index = int(len(group) * TRAIN_RATIO)

        split_index = max(
            1,
            min(split_index, len(group) - 1)
        )

        train_samples.extend(group[:split_index])
        val_samples.extend(group[split_index:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples


# ============================================================
# 9. DATA TRANSFORM
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomRotation(7),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 10. DATASET
# ============================================================

class MoneyDataset(Dataset):

    def __init__(self, samples, transform=None):

        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image_path = sample["path"]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

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
# 11. XÂY DỰNG MÔ HÌNH RESNET18
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

        # Bỏ lớp phân loại mặc định
        self.backbone.fc = nn.Identity()

        # Đầu ra phân loại tiền tệ
        self.currency_head = nn.Linear(
            input_features,
            num_currencies
        )

        # Đầu ra phân loại mệnh giá
        self.denomination_head = nn.Linear(
            input_features,
            num_denominations
        )

    def forward(self, x):

        features = self.backbone(x)

        currency_output = self.currency_head(features)

        denomination_output = self.denomination_head(features)

        return currency_output, denomination_output


# ============================================================
# 12. HUẤN LUYỆN MỘT EPOCH
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

        loss = currency_loss + denomination_loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

        currency_predictions = currency_output.argmax(dim=1)

        denomination_predictions = denomination_output.argmax(dim=1)

        correct_currency += (
            currency_predictions == currency_labels
        ).sum().item()

        correct_denomination += (
            denomination_predictions == denomination_labels
        ).sum().item()

        total += images.size(0)

    return {
        "loss": total_loss / total,
        "currency_accuracy": correct_currency / total,
        "denomination_accuracy": correct_denomination / total
    }


# ============================================================
# 13. ĐÁNH GIÁ VALIDATION
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

            total_loss += loss.item() * images.size(0)

            currency_predictions = currency_output.argmax(dim=1)

            denomination_predictions = denomination_output.argmax(dim=1)

            correct_currency += (
                currency_predictions == currency_labels
            ).sum().item()

            correct_denomination += (
                denomination_predictions == denomination_labels
            ).sum().item()

            total += images.size(0)

    return {
        "loss": total_loss / total,
        "currency_accuracy": correct_currency / total,
        "denomination_accuracy": correct_denomination / total
    }


# ============================================================
# 14. LƯU MODEL
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    filename
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }

    torch.save(
        checkpoint,
        OUTPUT_DIR / filename
    )


# ============================================================
# 15. CHƯƠNG TRÌNH CHÍNH
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    print("=" * 65)
    print("HUẤN LUYỆN MÔ HÌNH RESNET18")
    print("=" * 65)

    print(f"Thiết bị: {DEVICE}")
    print(f"Thư mục dữ liệu: {DATA_DIR}")
    print(f"Thư mục kết quả: {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Đọc mapping
    # --------------------------------------------------------

    (
        currency_classes,
        denomination_classes,
        currency_to_idx,
        denomination_to_idx
    ) = load_class_mapping()

    num_currencies = len(currency_classes)

    num_denominations = len(denomination_classes)

    print("\nSố lớp tiền tệ:", num_currencies)

    print("Số lớp mệnh giá:", num_denominations)

    # --------------------------------------------------------
    # Quét dữ liệu
    # --------------------------------------------------------

    print("\nĐang quét dữ liệu ảnh...")

    samples = scan_dataset(
        currency_to_idx,
        denomination_to_idx
    )

    if len(samples) == 0:
        raise RuntimeError(
            "Không tìm thấy ảnh hợp lệ. "
            "Hãy kiểm tra thư mục asset và class_mapping.json."
        )

    print("Tổng số ảnh:", len(samples))

    # --------------------------------------------------------
    # Chia dữ liệu
    # --------------------------------------------------------

    train_samples, val_samples = stratified_split(samples)

    print("Số ảnh Train:", len(train_samples))

    print("Số ảnh Validation:", len(val_samples))

    # --------------------------------------------------------
    # Khởi tạo Dataset
    # --------------------------------------------------------

    train_dataset = MoneyDataset(
        train_samples,
        transform=train_transform
    )

    val_dataset = MoneyDataset(
        val_samples,
        transform=val_transform
    )

    # --------------------------------------------------------
    # Khởi tạo DataLoader
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Khởi tạo mô hình
    # --------------------------------------------------------

    print("\nĐang khởi tạo ResNet18...")

    model = ResNetMultiTask(
        num_currencies=num_currencies,
        num_denominations=num_denominations,
        pretrained=USE_PRETRAINED
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    # --------------------------------------------------------
    # Lưu mapping vào thư mục kết quả
    # --------------------------------------------------------

    shutil.copy2(
        MAPPING_PATH,
        OUTPUT_DIR / "class_mapping.json"
    )

    # --------------------------------------------------------
    # Huấn luyện
    # --------------------------------------------------------

    best_accuracy = 0.0

    history = []

    print("\nBẮT ĐẦU HUẤN LUYỆN")
    print("=" * 65)

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

        # Accuracy trung bình của hai đầu ra
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
            "train_currency_accuracy": train_metrics["currency_accuracy"],
            "train_denomination_accuracy": train_metrics["denomination_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_currency_accuracy": val_metrics["currency_accuracy"],
            "val_denomination_accuracy": val_metrics["denomination_accuracy"],
            "val_average_accuracy": val_accuracy
        }

        history.append(record)

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

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

        print(f"Learning Rate: {current_lr:.7f}")

        # Lưu model tốt nhất
        if val_accuracy > best_accuracy:

            best_accuracy = val_accuracy

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_metrics,
                "best_model.pth"
            )

            print("Đã lưu best_model.pth")

        # Lưu model của epoch gần nhất
        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            val_metrics,
            "last_model.pth"
        )

    # --------------------------------------------------------
    # Lưu lịch sử huấn luyện CSV
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Lưu tổng kết huấn luyện JSON
    # --------------------------------------------------------

    final_metrics = {
        "model": "ResNet18",
        "device": str(DEVICE),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_currencies": num_currencies,
        "num_denominations": num_denominations,
        "total_images": len(samples),
        "train_images": len(train_samples),
        "validation_images": len(val_samples),
        "best_validation_average_accuracy": best_accuracy,
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
    # KẾT THÚC
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("HOÀN TẤT HUẤN LUYỆN RESNET18")
    print("=" * 65)

    print(
        f"Accuracy validation tốt nhất: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(f"Thư mục kết quả: {OUTPUT_DIR}")

    print("\nCác file đã lưu:")

    for file_path in OUTPUT_DIR.iterdir():

        if file_path.is_file():
            print(f"- {file_path.name}")

    print("=" * 65)


if __name__ == "__main__":
    main()
