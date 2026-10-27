
import os
import cv2
import numpy as np
from pathlib import Path

# ==========================================
# 1. KHAI BÁO THƯ MỤC
# ==========================================

# Thư mục chứa ảnh tiền giấy ban đầu
INPUT_DIR = Path("asset_demo_02_anh_cai_thien")

# Thư mục lưu ảnh sau khi chuẩn hóa
OUTPUT_DIR = Path("asset_demo_02_anh_chuan_hoa")

# Kích thước đầu vào của MobileNetV2
IMAGE_SIZE = (224, 224)

# Tạo thư mục đầu ra nếu chưa tồn tại
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. HÀM CHUẨN HÓA ẢNH
# ==========================================

def chuan_hoa_anh(image_path):
    """
    Đọc và chuẩn hóa một ảnh tiền giấy:
    - Đọc ảnh từ đường dẫn.
    - Chuyển BGR sang RGB.
    - Resize về 224 x 224.
    - Chuẩn hóa pixel về khoảng [-1, 1].
    """

    # Đọc ảnh bằng OpenCV
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    # Chuyển từ BGR sang RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Thay đổi kích thước ảnh
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Chuyển sang kiểu float32
    image = image.astype(np.float32)

    # Chuẩn hóa pixel từ [0, 255] về [-1, 1]
    image = image / 127.5 - 1.0

    return image


# ==========================================
# 3. XỬ LÝ TOÀN BỘ ẢNH TRONG THƯ MỤC
# ==========================================

extensions = {".jpg", ".jpeg", ".png", ".bmp"}

image_paths = [
    path for path in INPUT_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in extensions
]

if not image_paths:
    print("Không tìm thấy ảnh. Hãy kiểm tra lại INPUT_DIR.")
else:
    thanh_cong = 0
    that_bai = 0

    for image_path in image_paths:
        try:
            # Chuẩn hóa ảnh
            image = chuan_hoa_anh(image_path)

            # Giữ nguyên cấu trúc thư mục con
            relative_path = image_path.relative_to(INPUT_DIR)
            output_path = OUTPUT_DIR / relative_path

            # Tạo thư mục con tương ứng
            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # Đổi pixel về [0, 255] để lưu ảnh JPEG/PNG
            image_to_save = np.clip(
                (image + 1.0) * 127.5,
                0,
                255
            ).astype(np.uint8)

            # Chuyển RGB về BGR để lưu bằng OpenCV
            image_to_save = cv2.cvtColor(
                image_to_save,
                cv2.COLOR_RGB2BGR
            )

            # Lưu ảnh đã resize
            if cv2.imwrite(str(output_path), image_to_save):
                thanh_cong += 1
            else:
                that_bai += 1
                print(f"Không lưu được: {image_path}")

        except Exception as e:
            that_bai += 1
            print(f"Lỗi xử lý {image_path}: {e}")

    print("===== KẾT QUẢ XỬ LÝ =====")
    print(f"Tổng số ảnh: {len(image_paths)}")
    print(f"Thành công: {thanh_cong}")
    print(f"Thất bại: {that_bai}")
    print(f"Thư mục đầu ra: {OUTPUT_DIR.resolve()}")
