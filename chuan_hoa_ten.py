# Input folder : asset
# Output folder : asset_finally  
''' Chuẩn hóa tên asset'''
import os
import shutil
from pathlib import Path

# Thư mục ảnh gốc
SOURCE_FOLDER = Path("asset")

# Thư mục lưu ảnh sau khi đổi tên
OUTPUT_FOLDER = Path("asset_finally")

# Các định dạng ảnh được hỗ trợ
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


def rename_and_copy_images(source, destination):
    # Tạo thư mục đích
    destination.mkdir(parents=True, exist_ok=True)

    # Lấy danh sách ảnh và sắp xếp theo tên
    images = sorted(
        [
            file for file in source.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda x: x.name.lower()
    )

    # Sao chép và đổi tên ảnh
    for index, image in enumerate(images, start=1):
        new_name = f"{index}{image.suffix.lower()}"

        output_path = destination / new_name

        shutil.copy2(image, output_path)

    print(f"Đã xử lý: {source}")


# Duyệt toàn bộ thư mục con
for current_folder, subfolders, files in os.walk(SOURCE_FOLDER):

    current_folder = Path(current_folder)

    # Giữ nguyên cấu trúc thư mục
    relative_path = current_folder.relative_to(SOURCE_FOLDER)
    output_path = OUTPUT_FOLDER / relative_path

    rename_and_copy_images(current_folder, output_path)

print("HOÀN THÀNH!")