
import json
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_DIR / "anh_kiem_tra" / "Bo_test_3"

MODEL_DIR = PROJECT_DIR / "ket_qua_resnet"
MODEL_PATH = MODEL_DIR / "best_model.pth"

MAPPING_PATH = PROJECT_DIR / "asset_restnet" / "class_mapping.json"

# OUTPUT_CSV = MODEL_DIR / "ket_qua_du_doan_Bo_test_01.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "ket_qua_du_doan_restnet_test_03.csv"

IMG_SIZE = (224, 224)

SUPPORTED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# 2. ĐỌC CLASS MAPPING
# ==========================================

if not MAPPING_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy file mapping: {MAPPING_PATH}"
    )

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

currency_classes = mapping["currency_classes"]
denomination_classes = mapping["denomination_classes"]

num_currencies = len(currency_classes)
num_denominations = len(denomination_classes)


# ==========================================
# 3. XÂY DỰNG MÔ HÌNH RESNET18
# ==========================================

class ResNet18Model(nn.Module):

    def __init__(self, num_currencies, num_denominations):
        super().__init__()

        self.backbone = models.resnet18(weights=None)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()

        self.currency_head = nn.Linear(
            in_features,
            num_currencies
        )

        self.denomination_head = nn.Linear(
            in_features,
            num_denominations
        )

    def forward(self, x):

        features = self.backbone(x)

        currency_output = self.currency_head(features)

        denomination_output = self.denomination_head(features)

        return currency_output, denomination_output


# ==========================================
# 4. KHỞI TẠO VÀ NẠP TRỌNG SỐ
# ==========================================

model = ResNet18Model(
    num_currencies,
    num_denominations
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy mô hình: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

# Hỗ trợ checkpoint lưu trực tiếp state_dict
# hoặc lưu trong một dictionary
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint

# Loại bỏ tiền tố module. nếu có
state_dict = {
    key.replace("module.", "", 1): value
    for key, value in state_dict.items()
}

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("Da tai mo hinh thanh cong!")
print("Thiet bi:", DEVICE)


# ==========================================
# 5. TIỀN XỬ LÝ ẢNH
# ==========================================

transform = transforms.Compose([

    transforms.Resize(IMG_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==========================================
# 6. HÀM DỰ ĐOÁN MỘT ẢNH
# ==========================================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        currency_output, denomination_output = model(
            image_tensor
        )

        currency_prob = torch.softmax(
            currency_output, dim=1
        )

        denomination_prob = torch.softmax(
            denomination_output, dim=1
        )

        currency_conf, currency_idx = torch.max(
            currency_prob, dim=1
        )

        denomination_conf, denomination_idx = torch.max(
            denomination_prob, dim=1
        )

    currency_name = currency_classes[
        currency_idx.item()
    ]

    denomination_name = denomination_classes[
        denomination_idx.item()
    ]

    return {
        "loai_tien_du_doan": currency_name,
        "menh_gia_du_doan": denomination_name,
        "do_tin_cay_loai_tien": round(
            currency_conf.item() * 100, 2
        ),
        "do_tin_cay_menh_gia": round(
            denomination_conf.item() * 100, 2
        )
    }


# ==========================================
# 7. DUYỆT ẢNH TRONG THƯ MỤC KIỂM TRA
# ==========================================

def predict_folder():

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh: {TEST_DIR}"
        )

    image_paths = sorted([
        path
        for path in TEST_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if len(image_paths) == 0:
        print("Khong tim thay anh trong thu muc kiem tra!")
        return

    results = []

    print("\nBat dau du doan...")
    print("Tong so anh:", len(image_paths))

    for index, image_path in enumerate(image_paths, start=1):

        try:

            prediction = predict_image(image_path)

            results.append({

                "ten_anh": image_path.name,

                "duong_dan_anh": str(
                    image_path.relative_to(PROJECT_DIR)
                ),

                "loai_tien_du_doan":
                    prediction["loai_tien_du_doan"],

                "menh_gia_du_doan":
                    prediction["menh_gia_du_doan"],

                "do_tin_cay_loai_tien":
                    prediction["do_tin_cay_loai_tien"],

                "do_tin_cay_menh_gia":
                    prediction["do_tin_cay_menh_gia"]

            })

            print(
                f"[{index}/{len(image_paths)}] "
                f"{image_path.name} -> "
                f"{prediction['menh_gia_du_doan']}"
            )

        except Exception as error:

            print(
                f"Loi khi xu ly {image_path.name}: {error}"
            )

    # ======================================
    # 8. XUẤT KẾT QUẢ RA FILE CSV
    # ======================================

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "ten_anh",
        "duong_dan_anh",
        "loai_tien_du_doan",
        "menh_gia_du_doan",
        "do_tin_cay_loai_tien",
        "do_tin_cay_menh_gia"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    print("\n================================")
    print("HOAN THANH DU DOAN!")
    print("So anh duoc du doan:", len(results))
    print("File CSV:", OUTPUT_CSV)
    print("================================")


# ==========================================
# 9. CHẠY CHƯƠNG TRÌNH
# ==========================================

if __name__ == "__main__":

    predict_folder()
