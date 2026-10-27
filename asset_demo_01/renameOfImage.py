# ============================================ Chuẩn hóa tên tiền Mỹ ============================================
from pathlib import Path
import cv2

# Lấy thư mục chứa file renameOfImage.py
BASE_DIR = Path(__file__).resolve().parent

# Thư mục ảnh gốc và thư mục kết quả
SOURCE_DIR = BASE_DIR / "AmericanDollar"
OUTPUT_DIR = BASE_DIR / "AmericanDollar_Rename"

# Các thư mục mệnh giá
DENOMINATIONS = [
    "1_usd",
    "2_usd",
    "5_usd",
    "3_usd",
    "10_usd",
    "20_usd",
    "50_usd",
    "100_usd"
]

# Các định dạng ảnh được hỗ trợ
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

# Tạo thư mục đầu ra
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

total_images = 0

for denomination in DENOMINATIONS:
    input_folder = SOURCE_DIR / denomination
    output_folder = OUTPUT_DIR / denomination

    if not input_folder.is_dir():
        print(f"Không tìm thấy thư mục: {input_folder}")
        continue

    output_folder.mkdir(parents=True, exist_ok=True)

    # Sắp xếp tên ảnh để đánh số ổn định
    image_files = sorted(
        file for file in input_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    index = 1

    for image_path in image_files:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Không đọc được ảnh: {image_path.name}")
            continue

        # Chuẩn hóa tên ảnh, lưu dưới dạng JPG
        new_name = f"{denomination}_{index}.jpg"
        output_path = output_folder / new_name

        success = cv2.imwrite(str(output_path), image)

        if success:
            print(f"Đã xử lý: {image_path.name} -> {new_name}")
            index += 1
            total_images += 1

print("\n================================")
print("HOÀN TẤT CHUẨN HÓA TÊN ẢNH")
print(f"Tổng số ảnh đã xử lý: {total_images}")
print(f"Thư mục đầu ra: {OUTPUT_DIR}")


# ============================= Chuẩn hóa tên ảnh cho tiền Trung Quốc =============================
# from pathlib import Path
# import cv2

# # Lấy thư mục chứa file renameOfImage.py
# BASE_DIR = Path(__file__).resolve().parent

# # Thư mục ảnh gốc và thư mục kết quả
# SOURCE_DIR = BASE_DIR / "ChineseYuan"
# OUTPUT_DIR = BASE_DIR / "ChineseYuan_Rename"

# # Các thư mục mệnh giá
# DENOMINATIONS = [
#     "1_cny",
#     "5_cny",
#     "10_cny",
#     "20_cny",
#     "50_cny",
#     "100_cny",

# ]

# # Các định dạng ảnh được hỗ trợ
# IMAGE_EXTENSIONS = {
#     ".jpg", ".jpeg", ".png", ".bmp",
#     ".webp", ".tif", ".tiff"
# }

# # Tạo thư mục đầu ra
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# total_images = 0

# for denomination in DENOMINATIONS:
#     input_folder = SOURCE_DIR / denomination
#     output_folder = OUTPUT_DIR / denomination

#     if not input_folder.is_dir():
#         print(f"Không tìm thấy thư mục: {input_folder}")
#         continue

#     output_folder.mkdir(parents=True, exist_ok=True)

#     # Sắp xếp tên ảnh để đánh số ổn định
#     image_files = sorted(
#         file for file in input_folder.iterdir()
#         if file.is_file()
#         and file.suffix.lower() in IMAGE_EXTENSIONS
#     )

#     index = 1

#     for image_path in image_files:
#         image = cv2.imread(str(image_path))

#         if image is None:
#             print(f"Không đọc được ảnh: {image_path.name}")
#             continue

#         # Chuẩn hóa tên ảnh, lưu dưới dạng JPG
#         new_name = f"{denomination}_{index}.jpg"
#         output_path = output_folder / new_name

#         success = cv2.imwrite(str(output_path), image)

#         if success:
#             print(f"Đã xử lý: {image_path.name} -> {new_name}")
#             index += 1
#             total_images += 1

# print("\n================================")
# print("HOÀN TẤT CHUẨN HÓA TÊN ẢNH")
# print(f"Tổng số ảnh đã xử lý: {total_images}")
# print(f"Thư mục đầu ra: {OUTPUT_DIR}")

# ============================ Chuẩn hóa tên ảnh cho tiền Việt Nam =============================
# from pathlib import Path
# import cv2

# # Lấy thư mục chứa file renameOfImage.py
# BASE_DIR = Path(__file__).resolve().parent

# # Thư mục ảnh gốc và thư mục kết quả
# SOURCE_DIR = BASE_DIR / "VietnameseDong"
# OUTPUT_DIR = BASE_DIR / "VietnameseDong_Rename"

# # Các thư mục mệnh giá
# DENOMINATIONS = [
#     "1000_vnd",
#     "2000_vnd",
#     "5000_vnd",
#     "10000_vnd",
#     "20000_vnd",
#     "50000_vnd",
#     "100000_vnd",
#     "200000_vnd",
#     "500000_vnd"
# ]

# # Các định dạng ảnh được hỗ trợ
# IMAGE_EXTENSIONS = {
#     ".jpg", ".jpeg", ".png", ".bmp",
#     ".webp", ".tif", ".tiff"
# }

# # Tạo thư mục đầu ra
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# total_images = 0

# for denomination in DENOMINATIONS:
#     input_folder = SOURCE_DIR / denomination
#     output_folder = OUTPUT_DIR / denomination

#     if not input_folder.is_dir():
#         print(f"Không tìm thấy thư mục: {input_folder}")
#         continue

#     output_folder.mkdir(parents=True, exist_ok=True)

#     # Sắp xếp tên ảnh để đánh số ổn định
#     image_files = sorted(
#         file for file in input_folder.iterdir()
#         if file.is_file()
#         and file.suffix.lower() in IMAGE_EXTENSIONS
#     )

#     index = 1

#     for image_path in image_files:
#         image = cv2.imread(str(image_path))

#         if image is None:
#             print(f"Không đọc được ảnh: {image_path.name}")
#             continue

#         # Chuẩn hóa tên ảnh, lưu dưới dạng JPG
#         new_name = f"{denomination}_{index}.jpg"
#         output_path = output_folder / new_name

#         success = cv2.imwrite(str(output_path), image)

#         if success:
#             print(f"Đã xử lý: {image_path.name} -> {new_name}")
#             index += 1
#             total_images += 1

# print("\n================================")
# print("HOÀN TẤT CHUẨN HÓA TÊN ẢNH")
# print(f"Tổng số ảnh đã xử lý: {total_images}")
# print(f"Thư mục đầu ra: {OUTPUT_DIR}")
