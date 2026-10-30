
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

# ==========================================
# 1. KHAI BÁO ĐƯỜNG DẪN
# ==========================================

DATASET_PATH = Path("asset_finally")
OUTPUT_PATH = Path("asset_overview")

OUTPUT_PATH.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".webp", ".bmp", ".gif", ".tif", ".tiff"
}

# ==========================================
# 2. THU THẬP THÔNG TIN ẢNH
# ==========================================

image_data = []

for image_path in DATASET_PATH.rglob("*"):

    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    try:
        with Image.open(image_path) as img:

            width, height = img.size

            image_data.append({
                "filename": image_path.name,
                "folder": image_path.parent.name,
                "path": str(image_path),
                "format": img.format,
                "width": width,
                "height": height,
                "resolution": width * height,
                "mode": img.mode,
                "size_kb": round(
                    image_path.stat().st_size / 1024, 2
                ),
                "aspect_ratio": round(width / height, 3)
            })

    except Exception as e:
        print(f"Không đọc được ảnh: {image_path} - {e}")

# Chuyển dữ liệu thành DataFrame
df = pd.DataFrame(image_data)

if df.empty:
    print("Không tìm thấy ảnh trong thư mục!")
    raise SystemExit

# ==========================================
# 3. TỔNG QUAN DATASET
# ==========================================

print("\n========== TỔNG QUAN DATASET ==========")

print("Tổng số ảnh:", len(df))

print("Tổng số thư mục chứa ảnh:",
      df["folder"].nunique())

print("Tổng dung lượng:",
      round(df["size_kb"].sum() / 1024, 2), "MB")

print("\n========== THỐNG KÊ KÍCH THƯỚC ==========")

print(df[["width", "height", "resolution",
          "size_kb", "aspect_ratio"]].describe())

# ==========================================
# 4. THỐNG KÊ SỐ LƯỢNG ẢNH THEO THƯ MỤC
# ==========================================

folder_counts = (
    df.groupby("folder")
    .size()
    .reset_index(name="image_count")
    .sort_values("image_count", ascending=False)
)

print("\n========== SỐ LƯỢNG ẢNH THEO THƯ MỤC ==========")
print(folder_counts.to_string(index=False))

# ==========================================
# 5. THỐNG KÊ ĐỊNH DẠNG ẢNH
# ==========================================

format_counts = (
    df["format"]
    .value_counts()
    .reset_index()
)

format_counts.columns = ["format", "count"]

print("\n========== ĐỊNH DẠNG ẢNH ==========")
print(format_counts.to_string(index=False))

# ==========================================
# 6. THỐNG KÊ CHẾ ĐỘ MÀU
# ==========================================

mode_counts = df["mode"].value_counts()

print("\n========== CHẾ ĐỘ MÀU ==========")
print(mode_counts)

# ==========================================
# 7. LƯU FILE CSV
# ==========================================

df.to_csv(
    OUTPUT_PATH / "image_details.csv",
    index=False,
    encoding="utf-8-sig"
)

folder_counts.to_csv(
    OUTPUT_PATH / "folder_statistics.csv",
    index=False,
    encoding="utf-8-sig"
)

format_counts.to_csv(
    OUTPUT_PATH / "format_statistics.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 8. TRỰC QUAN HÓA DỮ LIỆU
# ==========================================

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# Biểu đồ 1: Số lượng ảnh theo thư mục
sns.barplot(
    data=folder_counts,
    x="folder",
    y="image_count",
    ax=axes[0, 0]
)

axes[0, 0].set_title("So luong anh theo thu muc")
axes[0, 0].set_xlabel("Thu muc")
axes[0, 0].set_ylabel("So luong anh")
axes[0, 0].tick_params(axis="x", rotation=45)

# Biểu đồ 2: Tỷ lệ định dạng ảnh
axes[0, 1].pie(
    format_counts["count"],
    labels=format_counts["format"],
    autopct="%1.1f%%",
    startangle=90
)

axes[0, 1].set_title("Ty le dinh dang anh")

# Biểu đồ 3: Phân bố chiều rộng và chiều cao
sns.scatterplot(
    data=df,
    x="width",
    y="height",
    hue="folder",
    ax=axes[1, 0],
    legend=False
)

axes[1, 0].set_title("Phan bo kich thuoc anh")
axes[1, 0].set_xlabel("Width")
axes[1, 0].set_ylabel("Height")

# Biểu đồ 4: Phân bố dung lượng ảnh
sns.histplot(
    data=df,
    x="size_kb",
    bins=30,
    kde=True,
    ax=axes[1, 1]
)

axes[1, 1].set_title("Phan bo dung luong anh")
axes[1, 1].set_xlabel("Dung luong (KB)")
axes[1, 1].set_ylabel("So luong")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH / "dataset_overview.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nĐÃ HOÀN THÀNH THỐNG KÊ DATASET!")
print("Kết quả được lưu tại:", OUTPUT_PATH)
