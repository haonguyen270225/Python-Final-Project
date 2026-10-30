
from pathlib import Path
import uuid

# Thư mục gốc
BASE_DIR = Path(__file__).resolve().parent
ROOT_FOLDER = BASE_DIR / "anh_kiem_tra"


def rename_files(folder):
    # Lấy tất cả file trực tiếp trong thư mục
    files = sorted(
        [f for f in folder.iterdir() if f.is_file()],
        key=lambda x: x.name.lower()
    )

    # Bước 1: Đổi sang tên tạm để tránh trùng tên
    temp_files = []

    for file in files:
        temp_name = folder / f"temp_{uuid.uuid4().hex}{file.suffix}"
        file.rename(temp_name)
        temp_files.append((temp_name, file.suffix))

    # Bước 2: Đổi sang tên số thứ tự
    for index, (temp_file, extension) in enumerate(
        temp_files, start=1
    ):
        new_name = folder / f"{index}{extension.lower()}"
        temp_file.rename(new_name)

    print(f"Đã đổi tên {len(files)} file trong: {folder.name}")


# Duyệt từng thư mục con
for folder in ROOT_FOLDER.iterdir():
    if folder.is_dir():
        rename_files(folder)

print("\nHOÀN THÀNH ĐỔI TÊN FILE!")
