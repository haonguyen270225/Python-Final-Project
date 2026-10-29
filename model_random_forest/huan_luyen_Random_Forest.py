from pathlib import Path

code = r'''"""
Cấu hình và chuẩn hóa dữ liệu asset cho mô hình Random Forest.

Nguồn:
    D:\\Python-Final-Project\\asset\\
        AmericanDollar\\1 Dollar\\...
        ChineseYuan\\10\\...
        VietnameseDong\\000200\\...

Đầu ra:
    D:\\Python-Final-Project\\asset_Random_Forest\\
        USD_1\\...
        USD_2\\...
        CNY_10\\...
        VND_200\\...

Chạy từ thư mục gốc dự án:
    .\\.venv\\Scripts\\python.exe .\\model_RandomForest\\cau_hinh_du_lieu_Random_Forest.py

Script chỉ sao chép ảnh; không sửa/xóa ảnh trong asset gốc.
Nhãn được suy ra từ tên thư mục, nên cần kiểm tra nội dung ảnh và tên thư mục nguồn.
"""'''

from pathlib import Path
import csv
import re
import shutil
import sys
import warnings


# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

ASSET_DIR = BASE_DIR / "asset"
OUTPUT_DIR = BASE_DIR / "asset_Random_Forest"

# False: giữ dữ liệu cũ và thêm/cập nhật file mới.
# True: xóa TOÀN BỘ asset_Random_Forest trước khi tạo lại.
# Chỉ bật True nếu chắc chắn muốn dựng lại từ đầu.
RESET_OUTPUT = False

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

CURRENCY_MAP = {
    "AmericanDollar": "USD",
    "ChineseYuan": "CNY",
    "VietnameseDong": "VND",
}


def get_denomination(folder_name: str):
    """
    Lấy mệnh giá từ tên thư mục.
    Ví dụ:
        '1 Dollar' -> '1'
        '2 Doolar' -> '2'
        '000200'    -> '200'
        '050000'    -> '50000'
    Thư mục không có số như 'Test Set' sẽ được bỏ qua.
    """
    match = re.search(r"\d+", folder_name)
    if not match:
        return None

    try:
        return str(int(match.group(0)))
    except ValueError:
        return None


def get_unique_destination(destination: Path) -> Path:
    """Tránh ghi đè khi nhiều ảnh trùng tên trong cùng một lớp."""
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent
    counter = 1

    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def main():
    print("=" * 68)
    print("CHUẨN HÓA DỮ LIỆU ASSET CHO RANDOM FOREST")
    print("=" * 68)
    print(f"Thư mục nguồn : {ASSET_DIR}")
    print(f"Thư mục đầu ra: {OUTPUT_DIR}")

    if not ASSET_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục asset: {ASSET_DIR}\n"
            "Hãy kiểm tra cấu trúc dự án và vị trí file Python."
        )

    if OUTPUT_DIR.resolve() == ASSET_DIR.resolve():
        raise RuntimeError("Thư mục đầu ra không được trùng thư mục asset nguồn.")

    if RESET_OUTPUT and OUTPUT_DIR.exists():
        print(
            "\nRESET_OUTPUT=True: đang xóa thư mục asset_Random_Forest cũ..."
        )
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    skipped_folder_count = 0
    unreadable_path_count = 0
    class_counts = {}
    manifest_path = OUTPUT_DIR / "dataset_manifest.csv"
    manifest_rows = []

    for currency_folder_name, currency_code in CURRENCY_MAP.items():
        currency_dir = ASSET_DIR / currency_folder_name

        if not currency_dir.exists():
            warnings.warn(f"Không tìm thấy thư mục nguồn: {currency_dir}")
            continue

        for denomination_dir in sorted(currency_dir.iterdir()):
            if not denomination_dir.is_dir():
                continue

            denomination = get_denomination(denomination_dir.name)
            if denomination is None:
                skipped_folder_count += 1
                print(
                    f"Bỏ qua thư mục không có mệnh giá: "
                    f"{denomination_dir.relative_to(ASSET_DIR)}"
                )
                continue

            class_name = f"{currency_code}_{denomination}"
            class_output_dir = OUTPUT_DIR / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)

            image_paths = sorted(
                path for path in denomination_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            if not image_paths:
                print(
                    f"Cảnh báo: không tìm thấy ảnh trong "
                    f"{denomination_dir.relative_to(ASSET_DIR)}"
                )
                continue

            copied_this_class = 0

            for source_path in image_paths:
                try:
                    # Dùng tên file nguồn; nếu trùng tên thì thêm hậu tố _1, _2...
                    destination_path = get_unique_destination(
                        class_output_dir / source_path.name
                    )
                    shutil.copy2(source_path, destination_path)

                    manifest_rows.append({
                        "label": class_name,
                        "currency": currency_code,
                        "denomination": denomination,
                        "source_path": str(source_path.relative_to(BASE_DIR)),
                        "output_path": str(destination_path.relative_to(BASE_DIR)),
                    })

                    copied_count += 1
                    copied_this_class += 1

                except OSError as exc:
                    unreadable_path_count += 1
                    print(f"Lỗi sao chép {source_path}: {exc}")

            class_counts[class_name] = (
                class_counts.get(class_name, 0) + copied_this_class
            )
            print(f"{class_name:<14} : {copied_this_class:>5} ảnh")

    if copied_count == 0:
        raise RuntimeError(
            "Không sao chép được ảnh nào. Hãy kiểm tra tên thư mục "
            "nguồn và định dạng ảnh."
        )

    # Lưu manifest để truy vết ảnh gốc và nhãn được gán.
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "label", "currency", "denomination", "source_path", "output_path"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Lưu thống kê số ảnh mỗi lớp.
    counts_path = OUTPUT_DIR / "class_counts.csv"
    with counts_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "image_count"])
        for label, count in sorted(class_counts.items()):
            writer.writerow([label, count])

    print("\n" + "=" * 68)
    print("HOÀN TẤT CHUẨN HÓA DỮ LIỆU")
    print("=" * 68)
    print(f"Tổng ảnh đã sao chép : {copied_count}")
    print(f"Tổng lớp có ảnh      : {len(class_counts)}")
    print(f"Thư mục bỏ qua       : {skipped_folder_count}")
    print(f"Lỗi sao chép         : {unreadable_path_count}")
    print(f"Thư mục dữ liệu mới  : {OUTPUT_DIR}")
    print(f"File manifest        : {manifest_path}")
    print(f"Thống kê theo lớp    : {counts_path}")

    print("\nVí dụ cấu trúc đầu ra:")
    for class_name in sorted(class_counts)[:10]:
        print(f"  asset_Random_Forest/{class_name}/")
    if len(class_counts) > 10:
        print("  ...")

    print(
        "\nLưu ý: file này chỉ chuẩn hóa/sao chép dữ liệu, "
        "chưa huấn luyện Random Forest."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nLỖI: {error}", file=sys.stderr)
        raise

out = Path("/mnt/data/cau_hinh_du_lieu_Random_Forest.py")
out.write_text(code, encoding="utf-8")
print(f"Đã tạo file: {out} ({out.stat().st_size:,} bytes)")
