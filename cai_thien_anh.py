
import cv2
import numpy as np
from pathlib import Path

# =====================================================
# 1. KHAI BÁO THƯ MỤC
# =====================================================

INPUT_DIR = Path("asset_demo_01")
OUTPUT_DIR = Path("asset_demo_02_anh_cai_thien")

# Các định dạng ảnh được hỗ trợ
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

# =====================================================
# 2. HÀM CẢI THIỆN ẢNH
# =====================================================

def cai_thien_anh(image):
    """
    Cải thiện ảnh tiền giấy bằng:
    - Khử nhiễu nhẹ.
    - Tăng tương phản cục bộ bằng CLAHE.
    - Làm sắc nét nhẹ.

    Ảnh đầu vào và đầu ra đều sử dụng BGR của OpenCV.
    """

    # Bước 1: Khử nhiễu màu nhẹ
    # Giá trị càng lớn thì khử nhiễu càng mạnh,
    # nhưng có thể làm mất chi tiết nếu đặt quá cao.
    image_khu_nhieu = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=3,
        hColor=3,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Bước 2: Chuyển sang không gian màu LAB
    # Chỉ tăng tương phản trên kênh độ sáng L
    # để hạn chế làm thay đổi màu sắc tiền giấy.
    lab = cv2.cvtColor(image_khu_nhieu, cv2.COLOR_BGR2LAB)

    L, A, B = cv2.split(lab)

    # Bước 3: Tăng tương phản cục bộ bằng CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    L_cai_thien = clahe.apply(L)

    # Ghép lại các kênh LAB
    lab_cai_thien = cv2.merge((L_cai_thien, A, B))

    # Chuyển trở lại BGR
    anh_tang_tuong_phan = cv2.cvtColor(
        lab_cai_thien,
        cv2.COLOR_LAB2BGR
    )

    # Bước 4: Làm sắc nét nhẹ bằng Unsharp Mask
    anh_lam_mo = cv2.GaussianBlur(
        anh_tang_tuong_phan,
        (0, 0),
        sigmaX=1.0
    )

    anh_sac_net = cv2.addWeighted(
        anh_tang_tuong_phan,
        1.25,
        anh_lam_mo,
        -0.25,
        0
    )

    return anh_sac_net


# =====================================================
# 3. KIỂM TRA THƯ MỤC ĐẦU VÀO
# =====================================================

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục đầu vào: {INPUT_DIR.resolve()}\n"
        "Hãy kiểm tra lại đường dẫn INPUT_DIR."
    )

# Tạo thư mục đầu ra nếu chưa tồn tại
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 4. DUYỆT VÀ XỬ LÝ TẤT CẢ ẢNH
# =====================================================

tong_anh = 0
thanh_cong = 0
that_bai = 0

for image_path in INPUT_DIR.rglob("*"):

    # Bỏ qua thư mục và file không phải ảnh
    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    tong_anh += 1

    # Đọc ảnh
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[LỖI] Không đọc được ảnh: {image_path}")
        that_bai += 1
        continue

    try:
        # Cải thiện ảnh
        image_cai_thien = cai_thien_anh(image)

        # Giữ nguyên cấu trúc thư mục con
        relative_path = image_path.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / relative_path

        # Tạo thư mục con tương ứng
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Lưu ảnh đã cải thiện
        success = cv2.imwrite(
            str(output_path),
            image_cai_thien
        )

        if success:
            thanh_cong += 1
            print(f"[OK] {relative_path}")
        else:
            that_bai += 1
            print(f"[LỖI] Không lưu được: {output_path}")

    except Exception as error:
        that_bai += 1
        print(f"[LỖI] {image_path}: {error}")


# =====================================================
# 5. THỐNG KÊ KẾT QUẢ
# =====================================================

print("\n========== KẾT QUẢ XỬ LÝ ==========")
print(f"Tổng số ảnh tìm thấy : {tong_anh}")
print(f"Số ảnh thành công    : {thanh_cong}")
print(f"Số ảnh thất bại      : {that_bai}")
print(f"Thư mục kết quả      : {OUTPUT_DIR.resolve()}")
print("===================================")
