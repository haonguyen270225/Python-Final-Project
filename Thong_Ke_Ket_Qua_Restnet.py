
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================

THU_MUC_GOC = Path(__file__).resolve().parent

# Thư mục chứa các file CSV ResNet
THU_MUC_CSV = THU_MUC_GOC / "model_restnet"

# Thư mục lưu kết quả thống kê
THU_MUC_KET_QUA = (
    THU_MUC_GOC / "ket_qua_thong_ke" / "ResNet"
)

THU_MUC_KET_QUA.mkdir(
    parents=True,
    exist_ok=True
)

# Tự động tìm tất cả file CSV
DANH_SACH_CSV = sorted(
    THU_MUC_CSV.glob("*.csv")
)

# Tên các cột CSV
COT_TEN_ANH = "ten_anh"
COT_DUONG_DAN = "duong_dan_anh"
COT_LOAI_TIEN = "loai_tien_du_doan"
COT_MENH_GIA = "menh_gia_du_doan"

COT_TIN_CAY_LOAI_TIEN = "do_tin_cay_loai_tien"
COT_TIN_CAY_MENH_GIA = "do_tin_cay_menh_gia"

CAC_COT_CAN_THIET = [
    COT_TEN_ANH,
    COT_DUONG_DAN,
    COT_LOAI_TIEN,
    COT_MENH_GIA,
    COT_TIN_CAY_LOAI_TIEN,
    COT_TIN_CAY_MENH_GIA
]


# ==========================================
# 2. KIỂM TRA THƯ MỤC VÀ FILE CSV
# ==========================================

if not THU_MUC_CSV.exists():
    raise FileNotFoundError(
        f"Không tìm thấy thư mục CSV: {THU_MUC_CSV}"
    )

if len(DANH_SACH_CSV) == 0:
    raise FileNotFoundError(
        f"Không tìm thấy file CSV nào trong: {THU_MUC_CSV}"
    )

print("=" * 60)
print("THỐNG KÊ KẾT QUẢ DỰ ĐOÁN RESNET")
print("=" * 60)

print("\nCác file CSV được tìm thấy:")

for file in DANH_SACH_CSV:
    print("-", file.name)

print("\nTổng số file CSV:", len(DANH_SACH_CSV))


# ==========================================
# 3. ĐỌC VÀ KIỂM TRA DỮ LIỆU
# ==========================================

du_lieu_mo_hinh = {}

for duong_dan_csv in DANH_SACH_CSV:

    ten_mo_hinh = duong_dan_csv.stem

    print("\n" + "=" * 60)
    print("ĐANG ĐỌC:", duong_dan_csv.name)
    print("=" * 60)

    df = pd.read_csv(
        duong_dan_csv,
        encoding="utf-8-sig"
    )

    # Chuẩn hóa tên cột
    df.columns = df.columns.str.strip()

    # Kiểm tra các cột cần thiết
    for cot in CAC_COT_CAN_THIET:

        if cot not in df.columns:
            raise ValueError(
                f"File {duong_dan_csv.name} thiếu cột: {cot}\n"
                f"Các cột hiện có: {df.columns.tolist()}"
            )

    if df.empty:
        raise ValueError(
            f"File {duong_dan_csv.name} không có dữ liệu."
        )

    # Chuẩn hóa tên ảnh
    df[COT_TEN_ANH] = (
        df[COT_TEN_ANH]
        .fillna("Không xác định")
        .astype(str)
        .str.strip()
    )

    # Chuẩn hóa đường dẫn ảnh
    df[COT_DUONG_DAN] = (
        df[COT_DUONG_DAN]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Chuẩn hóa loại tiền
    df[COT_LOAI_TIEN] = (
        df[COT_LOAI_TIEN]
        .fillna("Không xác định")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Chuẩn hóa mệnh giá
    df[COT_MENH_GIA] = (
        df[COT_MENH_GIA]
        .fillna("Không xác định")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Chuyển độ tin cậy loại tiền sang kiểu số
    df[COT_TIN_CAY_LOAI_TIEN] = pd.to_numeric(
        df[COT_TIN_CAY_LOAI_TIEN],
        errors="coerce"
    )

    # Chuyển độ tin cậy mệnh giá sang kiểu số
    df[COT_TIN_CAY_MENH_GIA] = pd.to_numeric(
        df[COT_TIN_CAY_MENH_GIA],
        errors="coerce"
    )

    # Tạo nhãn kết hợp loại tiền và mệnh giá
    df["Loại tiền - Mệnh giá"] = (
        df[COT_LOAI_TIEN]
        + " - "
        + df[COT_MENH_GIA]
    )

    # Thêm tên file CSV để phục vụ so sánh
    df["Mô hình"] = ten_mo_hinh

    du_lieu_mo_hinh[ten_mo_hinh] = df

    print("Tổng số ảnh:", len(df))

    print("\n5 dòng dữ liệu đầu tiên:")
    print(df.head().to_string(index=False))


# ==========================================
# 4. THỐNG KÊ KẾT QUẢ TỪNG FILE CSV
# ==========================================

danh_sach_ket_qua = []

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    tong_so_anh = len(df)

    # Độ tin cậy loại tiền
    tin_cay_loai_tien = (
        df[COT_TIN_CAY_LOAI_TIEN].dropna()
    )

    # Độ tin cậy mệnh giá
    tin_cay_menh_gia = (
        df[COT_TIN_CAY_MENH_GIA].dropna()
    )

    ket_qua = {
        "File CSV": ten_mo_hinh,
        "Tổng số ảnh": tong_so_anh,

        "Độ tin cậy loại tiền TB (%)":
            round(tin_cay_loai_tien.mean(), 2),

        "Độ tin cậy loại tiền cao nhất (%)":
            round(tin_cay_loai_tien.max(), 2),

        "Độ tin cậy loại tiền thấp nhất (%)":
            round(tin_cay_loai_tien.min(), 2),

        "Độ tin cậy mệnh giá TB (%)":
            round(tin_cay_menh_gia.mean(), 2),

        "Độ tin cậy mệnh giá cao nhất (%)":
            round(tin_cay_menh_gia.max(), 2),

        "Độ tin cậy mệnh giá thấp nhất (%)":
            round(tin_cay_menh_gia.min(), 2),

        "Số ảnh có đủ 2 độ tin cậy":
            df[
                [
                    COT_TIN_CAY_LOAI_TIEN,
                    COT_TIN_CAY_MENH_GIA
                ]
            ].notna().all(axis=1).sum()
    }

    danh_sach_ket_qua.append(ket_qua)

    print("\n========== THỐNG KÊ:", ten_mo_hinh, "==========")

    for ten_chi_so, gia_tri in ket_qua.items():
        print(f"{ten_chi_so}: {gia_tri}")


# ==========================================
# 5. TẠO BẢNG SO SÁNH CÁC FILE CSV
# ==========================================

bang_ket_qua = pd.DataFrame(
    danh_sach_ket_qua
)

print("\n" + "=" * 60)
print("BẢNG SO SÁNH CÁC FILE CSV RESNET")
print("=" * 60)

print(bang_ket_qua.to_string(index=False))

bang_ket_qua.to_csv(
    THU_MUC_KET_QUA / "bang_so_sanh_cac_file.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 6. THỐNG KÊ SỐ LƯỢNG THEO LOẠI TIỀN
# ==========================================

danh_sach_loai_tien = []

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    thong_ke = df[COT_LOAI_TIEN].value_counts()

    for loai_tien, so_luong in thong_ke.items():

        danh_sach_loai_tien.append({
            "File CSV": ten_mo_hinh,
            "Loại tiền": loai_tien,
            "Số lượng ảnh": so_luong
        })

bang_loai_tien = pd.DataFrame(
    danh_sach_loai_tien
)

print("\n========== THỐNG KÊ LOẠI TIỀN ==========")
print(bang_loai_tien.to_string(index=False))

bang_loai_tien.to_csv(
    THU_MUC_KET_QUA / "thong_ke_loai_tien.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 7. THỐNG KÊ SỐ LƯỢNG THEO MỆNH GIÁ
# ==========================================

danh_sach_menh_gia = []

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    thong_ke = df[COT_MENH_GIA].value_counts()

    for menh_gia, so_luong in thong_ke.items():

        danh_sach_menh_gia.append({
            "File CSV": ten_mo_hinh,
            "Mệnh giá": menh_gia,
            "Số lượng ảnh": so_luong
        })

bang_menh_gia = pd.DataFrame(
    danh_sach_menh_gia
)

print("\n========== THỐNG KÊ MỆNH GIÁ ==========")
print(bang_menh_gia.to_string(index=False))

bang_menh_gia.to_csv(
    THU_MUC_KET_QUA / "thong_ke_menh_gia.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 8. THỐNG KÊ LOẠI TIỀN - MỆNH GIÁ
# ==========================================

danh_sach_nhan = []

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    thong_ke = df["Loại tiền - Mệnh giá"].value_counts()

    for nhan, so_luong in thong_ke.items():

        danh_sach_nhan.append({
            "File CSV": ten_mo_hinh,
            "Loại tiền - Mệnh giá": nhan,
            "Số lượng ảnh": so_luong
        })

bang_nhan = pd.DataFrame(danh_sach_nhan)

print("\n========== THỐNG KÊ LOẠI TIỀN - MỆNH GIÁ ==========")
print(bang_nhan.to_string(index=False))

bang_nhan.to_csv(
    THU_MUC_KET_QUA / "thong_ke_loai_tien_menh_gia.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 9. GỘP DỮ LIỆU CÁC FILE CSV
# ==========================================

df_tong_hop = pd.concat(
    du_lieu_mo_hinh.values(),
    ignore_index=True
)

df_tong_hop.to_csv(
    THU_MUC_KET_QUA / "du_lieu_tong_hop.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTổng số ảnh của tất cả file:", len(df_tong_hop))


# ==========================================
# 10. CẤU HÌNH BIỂU ĐỒ
# ==========================================

sns.set_theme(style="whitegrid")

# ==========================================
# 11. BIỂU ĐỒ ĐỘ TIN CẬY LOẠI TIỀN
# ==========================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=bang_ket_qua,
    x="File CSV",
    y="Độ tin cậy loại tiền TB (%)"
)

plt.title(
    "SO SÁNH ĐỘ TIN CẬY DỰ ĐOÁN LOẠI TIỀN",
    fontsize=14
)

plt.xlabel("File CSV")
plt.ylabel("Độ tin cậy trung bình (%)")

plt.ylim(0, 100)
plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    THU_MUC_KET_QUA / "so_sanh_do_tin_cay_loai_tien.png",
    dpi=300
)

plt.show()


# ==========================================
# 12. BIỂU ĐỒ ĐỘ TIN CẬY MỆNH GIÁ
# ==========================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=bang_ket_qua,
    x="File CSV",
    y="Độ tin cậy mệnh giá TB (%)"
)

plt.title(
    "SO SÁNH ĐỘ TIN CẬY DỰ ĐOÁN MỆNH GIÁ",
    fontsize=14
)

plt.xlabel("File CSV")
plt.ylabel("Độ tin cậy trung bình (%)")

plt.ylim(0, 100)
plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    THU_MUC_KET_QUA / "so_sanh_do_tin_cay_menh_gia.png",
    dpi=300
)

plt.show()


# ==========================================
# 13. BIỂU ĐỒ SỐ LƯỢNG THEO LOẠI TIỀN
# ==========================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=bang_loai_tien,
    x="Loại tiền",
    y="Số lượng ảnh",
    hue="File CSV"
)

plt.title(
    "SO SÁNH SỐ LƯỢNG DỰ ĐOÁN THEO LOẠI TIỀN",
    fontsize=14
)

plt.xlabel("Loại tiền")
plt.ylabel("Số lượng ảnh")

plt.tight_layout()

plt.savefig(
    THU_MUC_KET_QUA / "so_sanh_loai_tien.png",
    dpi=300
)

plt.show()


# ==========================================
# 14. BIỂU ĐỒ SỐ LƯỢNG THEO MỆNH GIÁ
# ==========================================

plt.figure(figsize=(14, 7))

sns.barplot(
    data=bang_menh_gia,
    x="Mệnh giá",
    y="Số lượng ảnh",
    hue="File CSV"
)

plt.title(
    "SO SÁNH SỐ LƯỢNG DỰ ĐOÁN THEO MỆNH GIÁ",
    fontsize=14
)

plt.xlabel("Mệnh giá")
plt.ylabel("Số lượng ảnh")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    THU_MUC_KET_QUA / "so_sanh_menh_gia.png",
    dpi=300
)

plt.show()


# ==========================================
# 15. BIỂU ĐỒ LOẠI TIỀN - MỆNH GIÁ
# ==========================================

plt.figure(figsize=(16, 7))

sns.barplot(
    data=bang_nhan,
    x="Loại tiền - Mệnh giá",
    y="Số lượng ảnh",
    hue="File CSV"
)

plt.title(
    "SO SÁNH SỐ LƯỢNG THEO LOẠI TIỀN VÀ MỆNH GIÁ",
    fontsize=14
)

plt.xlabel("Loại tiền - Mệnh giá")
plt.ylabel("Số lượng ảnh")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    THU_MUC_KET_QUA / "so_sanh_loai_tien_menh_gia.png",
    dpi=300
)

plt.show()


# ==========================================
# 16. BIỂU ĐỒ PHÂN BỐ ĐỘ TIN CẬY
# ==========================================

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    # Biểu đồ độ tin cậy loại tiền
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x=COT_TIN_CAY_LOAI_TIEN,
        bins=20,
        kde=True
    )

    plt.title(
        f"PHÂN BỐ ĐỘ TIN CẬY LOẠI TIỀN - {ten_mo_hinh}",
        fontsize=13
    )

    plt.xlabel("Độ tin cậy loại tiền (%)")
    plt.ylabel("Số lượng ảnh")

    plt.xlim(0, 100)
    plt.tight_layout()

    plt.savefig(
        THU_MUC_KET_QUA /
        f"{ten_mo_hinh}_phan_bo_do_tin_cay_loai_tien.png",
        dpi=300
    )

    plt.show()

    # Biểu đồ độ tin cậy mệnh giá
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x=COT_TIN_CAY_MENH_GIA,
        bins=20,
        kde=True
    )

    plt.title(
        f"PHÂN BỐ ĐỘ TIN CẬY MỆNH GIÁ - {ten_mo_hinh}",
        fontsize=13
    )

    plt.xlabel("Độ tin cậy mệnh giá (%)")
    plt.ylabel("Số lượng ảnh")

    plt.xlim(0, 100)
    plt.tight_layout()

    plt.savefig(
        THU_MUC_KET_QUA /
        f"{ten_mo_hinh}_phan_bo_do_tin_cay_menh_gia.png",
        dpi=300
    )

    plt.show()


# ==========================================
# 17. XUẤT BÁO CÁO CHI TIẾT
# ==========================================

with open(
    THU_MUC_KET_QUA / "bao_cao_thong_ke.txt",
    "w",
    encoding="utf-8"
) as tep:

    tep.write(
        "BÁO CÁO THỐNG KÊ KẾT QUẢ DỰ ĐOÁN RESNET\n"
    )

    tep.write("=" * 60 + "\n")

    tep.write(
        f"\nTổng số file CSV: {len(DANH_SACH_CSV)}\n"
    )

    tep.write(
        f"Tổng số ảnh: {len(df_tong_hop)}\n"
    )

    tep.write(
        "\nLưu ý: Độ tin cậy là điểm confidence do mô hình "
        "dự đoán cung cấp, không phải Accuracy thực tế.\n"
    )

    tep.write(
        "Chưa có nhãn thực tế nên chưa thể tính "
        "Accuracy, Precision, Recall và F1-score.\n"
    )

    for ten_mo_hinh, df in du_lieu_mo_hinh.items():

        tep.write("\n" + "=" * 60 + "\n")
        tep.write(f"FILE CSV: {ten_mo_hinh}\n")
        tep.write("=" * 60 + "\n")

        tep.write(f"Tổng số ảnh: {len(df)}\n")

        # Thống kê loại tiền
        tep.write("\n--- ĐỘ TIN CẬY LOẠI TIỀN ---\n")

        tep.write(
            f"Trung bình: "
            f"{df[COT_TIN_CAY_LOAI_TIEN].mean():.2f}%\n"
        )

        tep.write(
            f"Cao nhất: "
            f"{df[COT_TIN_CAY_LOAI_TIEN].max():.2f}%\n"
        )

        tep.write(
            f"Thấp nhất: "
            f"{df[COT_TIN_CAY_LOAI_TIEN].min():.2f}%\n"
        )

        # Thống kê mệnh giá
        tep.write("\n--- ĐỘ TIN CẬY MỆNH GIÁ ---\n")

        tep.write(
            f"Trung bình: "
            f"{df[COT_TIN_CAY_MENH_GIA].mean():.2f}%\n"
        )

        tep.write(
            f"Cao nhất: "
            f"{df[COT_TIN_CAY_MENH_GIA].max():.2f}%\n"
        )

        tep.write(
            f"Thấp nhất: "
            f"{df[COT_TIN_CAY_MENH_GIA].min():.2f}%\n"
        )

        # Thống kê loại tiền
        tep.write("\n--- SỐ LƯỢNG THEO LOẠI TIỀN ---\n")

        tep.write(
            df[COT_LOAI_TIEN].value_counts().to_string()
        )

        # Thống kê mệnh giá
        tep.write("\n\n--- SỐ LƯỢNG THEO MỆNH GIÁ ---\n")

        tep.write(
            df[COT_MENH_GIA].value_counts().to_string()
        )

        # Thống kê loại tiền - mệnh giá
        tep.write(
            "\n\n--- LOẠI TIỀN - MỆNH GIÁ ---\n"
        )

        tep.write(
            df["Loại tiền - Mệnh giá"]
            .value_counts()
            .to_string()
        )

        tep.write("\n")


# ==========================================
# 18. KẾT THÚC
# ==========================================

print("\n" + "=" * 60)
print("HOÀN THÀNH THỐNG KÊ RESNET!")
print("=" * 60)

print("\nThư mục chứa kết quả:")
print(THU_MUC_KET_QUA)

print("\nCác tệp kết quả đã tạo:")

for file in sorted(THU_MUC_KET_QUA.iterdir()):
    if file.is_file():
        print("-", file.name)

print("\nTổng số file CSV đã thống kê:", len(DANH_SACH_CSV))
print("Tổng số ảnh đã thống kê:", len(df_tong_hop))
