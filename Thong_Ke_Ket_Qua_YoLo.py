
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ==========================================

THU_MUC_GOC = Path(__file__).resolve().parent

# Thư mục chứa 3 file CSV đầu vào
THU_MUC_CSV = THU_MUC_GOC / "model_YoLo"

# Thư mục lưu kết quả thống kê
THU_MUC_KET_QUA = (
    THU_MUC_GOC / "ket_qua_thong_ke" / "YoLo"
)

THU_MUC_KET_QUA.mkdir(
    parents=True,
    exist_ok=True
)

DANH_SACH_MO_HINH = {
    "Mô hình 1": THU_MUC_CSV / "ket_qua_du_doan_yolo_test_01.csv",
    "Mô hình 2": THU_MUC_CSV / "ket_qua_du_doan_yolo_test_02.csv",
    "Mô hình 3": THU_MUC_CSV / "ket_qua_du_doan_yolo_test_03.csv"
}


# Tên các cột trong file CSV YOLO
COT_TEN_ANH = "ten_anh"
COT_LOAI_TIEN = "loai_tien_du_doan"
COT_MENH_GIA = "menh_gia_du_doan"
COT_DO_TIN_CAY = "do_tin_cay_phan_tram"
COT_DUONG_DAN = "duong_dan_anh_goc"
COT_TRANG_THAI = "trang_thai"

cac_cot_can_thiet = [
    COT_TEN_ANH,
    COT_LOAI_TIEN,
    COT_MENH_GIA,
    COT_DO_TIN_CAY,
    COT_DUONG_DAN,
    COT_TRANG_THAI
]


# ==========================================
# 2. ĐỌC VÀ KIỂM TRA DỮ LIỆU
# ==========================================

du_lieu_mo_hinh = {}

for ten_mo_hinh, duong_dan_csv in DANH_SACH_MO_HINH.items():

    if not duong_dan_csv.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp CSV: {duong_dan_csv}"
        )

    df = pd.read_csv(
        duong_dan_csv,
        encoding="utf-8-sig"
    )

    # Loại bỏ khoảng trắng trong tên cột
    df.columns = df.columns.str.strip()

    # Kiểm tra các cột
    for cot in cac_cot_can_thiet:
        if cot not in df.columns:
            raise ValueError(
                f"Tệp {duong_dan_csv.name} thiếu cột: {cot}\n"
                f"Các cột hiện có: {df.columns.tolist()}"
            )

    # Chuyển độ tin cậy thành kiểu số
    df[COT_DO_TIN_CAY] = pd.to_numeric(
        df[COT_DO_TIN_CAY],
        errors="coerce"
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
    )

    # Tạo nhãn kết hợp loại tiền và mệnh giá
    df["Loại tiền - Mệnh giá"] = (
        df[COT_LOAI_TIEN]
        + " - "
        + df[COT_MENH_GIA]
    )

    if df.empty:
        raise ValueError(
            f"Tệp {duong_dan_csv.name} không có dữ liệu."
        )

    du_lieu_mo_hinh[ten_mo_hinh] = df

    print(f"\n========== {ten_mo_hinh} ==========")
    print("Tổng số ảnh:", len(df))

    print("\nCác cột dữ liệu:")
    print(df.columns.tolist())

    print("\n5 dòng dữ liệu đầu tiên:")
    print(df.head().to_string(index=False))


# ==========================================
# 3. THỐNG KÊ KẾT QUẢ TỪNG MÔ HÌNH
# ==========================================

danh_sach_ket_qua = []

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    tong_so_anh = len(df)

    # Chuẩn hóa trạng thái để nhận cả Thanh cong và Thành công
    trang_thai = (
        df[COT_TRANG_THAI]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Số ảnh xử lý thành công
    so_anh_thanh_cong = trang_thai.isin(
        ["thanh cong", "thành công", "success"]
    ).sum()

    # Số ảnh xử lý thất bại
    so_anh_that_bai = tong_so_anh - so_anh_thanh_cong

    # Thống kê độ tin cậy
    do_tin_cay = df[COT_DO_TIN_CAY].dropna()

    do_tin_cay_trung_binh = do_tin_cay.mean()
    do_tin_cay_cao_nhat = do_tin_cay.max()
    do_tin_cay_thap_nhat = do_tin_cay.min()

    ti_le_thanh_cong = (
        so_anh_thanh_cong / tong_so_anh * 100
        if tong_so_anh > 0 else 0
    )

    ket_qua = {
        "Mô hình": ten_mo_hinh,
        "Tổng số ảnh": tong_so_anh,
        "Số ảnh thành công": so_anh_thanh_cong,
        "Số ảnh thất bại": so_anh_that_bai,
        "Tỷ lệ thành công (%)": round(ti_le_thanh_cong, 2),
        "Độ tin cậy trung bình (%)": round(do_tin_cay_trung_binh, 2),
        "Độ tin cậy cao nhất (%)": round(do_tin_cay_cao_nhat, 2),
        "Độ tin cậy thấp nhất (%)": round(do_tin_cay_thap_nhat, 2)
    }

    danh_sach_ket_qua.append(ket_qua)

    print(f"\n========== THỐNG KÊ {ten_mo_hinh} ==========")

    for ten_chi_so, gia_tri in ket_qua.items():
        print(f"{ten_chi_so}: {gia_tri}")


# ==========================================
# 4. TẠO BẢNG SO SÁNH 3 MÔ HÌNH
# ==========================================

bang_ket_qua = pd.DataFrame(danh_sach_ket_qua)

print("\n========== BẢNG SO SÁNH 3 MÔ HÌNH YOLO ==========")

print(bang_ket_qua.to_string(index=False))

bang_ket_qua.to_csv(
    THU_MUC_KET_QUA / "bang_so_sanh_3_mo_hinh.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 5. THỐNG KÊ SỐ LƯỢNG THEO LOẠI TIỀN
# ==========================================

danh_sach_loai_tien = []

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    thong_ke = df[COT_LOAI_TIEN].value_counts()

    for loai_tien, so_luong in thong_ke.items():

        danh_sach_loai_tien.append({
            "Mô hình": ten_mo_hinh,
            "Loại tiền": loai_tien,
            "Số lượng ảnh": so_luong
        })

bang_loai_tien = pd.DataFrame(danh_sach_loai_tien)

print("\n========== THỐNG KÊ THEO LOẠI TIỀN ==========")
print(bang_loai_tien.to_string(index=False))

bang_loai_tien.to_csv(
    THU_MUC_KET_QUA / "thong_ke_loai_tien.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 6. THỐNG KÊ SỐ LƯỢNG THEO MỆNH GIÁ
# ==========================================

danh_sach_menh_gia = []

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    thong_ke = df[COT_MENH_GIA].value_counts()

    for menh_gia, so_luong in thong_ke.items():

        danh_sach_menh_gia.append({
            "Mô hình": ten_mo_hinh,
            "Mệnh giá": menh_gia,
            "Số lượng ảnh": so_luong
        })

bang_menh_gia = pd.DataFrame(danh_sach_menh_gia)

print("\n========== THỐNG KÊ THEO MỆNH GIÁ ==========")
print(bang_menh_gia.to_string(index=False))

bang_menh_gia.to_csv(
    THU_MUC_KET_QUA / "thong_ke_menh_gia.csv",
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 7. THỐNG KÊ THEO LOẠI TIỀN VÀ MỆNH GIÁ
# ==========================================

danh_sach_nhan = []

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    thong_ke = df["Loại tiền - Mệnh giá"].value_counts()

    for nhan, so_luong in thong_ke.items():

        danh_sach_nhan.append({
            "Mô hình": ten_mo_hinh,
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
# 8. CẤU HÌNH BIỂU ĐỒ
# ==========================================

sns.set_theme(style="whitegrid")


# ==========================================
# 9. BIỂU ĐỒ SO SÁNH ĐỘ TIN CẬY
# ==========================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=bang_ket_qua,
    x="Mô hình",
    y="Độ tin cậy trung bình (%)"
)

plt.title(
    "SO SÁNH ĐỘ TIN CẬY TRUNG BÌNH CỦA 3 MÔ HÌNH YOLO",
    fontsize=14
)

plt.xlabel("Mô hình")
plt.ylabel("Độ tin cậy trung bình (%)")
plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    THU_MUC_KET_QUA / "so_sanh_do_tin_cay.png",
    dpi=300
)

plt.show()


# ==========================================
# 10. BIỂU ĐỒ SO SÁNH TRẠNG THÁI DỰ ĐOÁN
# ==========================================

bang_trang_thai = bang_ket_qua.melt(
    id_vars="Mô hình",
    value_vars=[
        "Số ảnh thành công",
        "Số ảnh thất bại"
    ],
    var_name="Trạng thái",
    value_name="Số lượng ảnh"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=bang_trang_thai,
    x="Mô hình",
    y="Số lượng ảnh",
    hue="Trạng thái"
)

plt.title(
    "SO SÁNH TRẠNG THÁI XỬ LÝ ẢNH CỦA 3 MÔ HÌNH YOLO",
    fontsize=14
)

plt.xlabel("Mô hình")
plt.ylabel("Số lượng ảnh")

plt.tight_layout()

plt.savefig(
    THU_MUC_KET_QUA / "so_sanh_trang_thai_du_doan.png",
    dpi=300
)

plt.show()


# ==========================================
# 11. BIỂU ĐỒ SO SÁNH SỐ LƯỢNG LOẠI TIỀN
# ==========================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=bang_loai_tien,
    x="Loại tiền",
    y="Số lượng ảnh",
    hue="Mô hình"
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
# 12. BIỂU ĐỒ SO SÁNH SỐ LƯỢNG MỆNH GIÁ
# ==========================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=bang_menh_gia,
    x="Mệnh giá",
    y="Số lượng ảnh",
    hue="Mô hình"
)

plt.title(
    "SO SÁNH SỐ LƯỢNG DỰ ĐOÁN THEO MỆNH GIÁ",
    fontsize=14
)

plt.xlabel("Mệnh giá")
plt.ylabel("Số lượng ảnh")

plt.tight_layout()

plt.savefig(
    THU_MUC_KET_QUA / "so_sanh_menh_gia.png",
    dpi=300
)

plt.show()


# ==========================================
# 13. BIỂU ĐỒ SO SÁNH LOẠI TIỀN - MỆNH GIÁ
# ==========================================

plt.figure(figsize=(14, 7))

sns.barplot(
    data=bang_nhan,
    x="Loại tiền - Mệnh giá",
    y="Số lượng ảnh",
    hue="Mô hình"
)

plt.title(
    "SO SÁNH SỐ LƯỢNG DỰ ĐOÁN THEO LOẠI TIỀN VÀ MỆNH GIÁ",
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
# 14. BIỂU ĐỒ PHÂN BỐ ĐỘ TIN CẬY
# ==========================================

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x=COT_DO_TIN_CAY,
        bins=20,
        kde=True
    )

    plt.title(
        f"PHÂN BỐ ĐỘ TIN CẬY - {ten_mo_hinh}",
        fontsize=14
    )

    plt.xlabel("Độ tin cậy (%)")
    plt.ylabel("Số lượng ảnh")
    plt.xlim(0, 100)

    plt.tight_layout()

    ten_file = (
        ten_mo_hinh.lower()
        .replace(" ", "_")
        .replace("ô", "o")
        .replace("hình", "hinh")
    )

    plt.savefig(
        THU_MUC_KET_QUA / f"{ten_file}_phan_bo_do_tin_cay.png",
        dpi=300
    )

    plt.show()


# ==========================================
# 15. XUẤT BÁO CÁO CHI TIẾT
# ==========================================

with open(
    THU_MUC_KET_QUA / "bao_cao_thong_ke.txt",
    "w",
    encoding="utf-8"
) as tep:

    tep.write("BÁO CÁO THỐNG KÊ KẾT QUẢ 3 MÔ HÌNH YOLO\n")

    tep.write("\nLưu ý:")
    tep.write(
        "\nCác tệp CSV hiện tại chưa có nhãn thực tế."
    )
    tep.write(
        "\nDo đó chưa thể tính Accuracy, Precision, Recall, "
        "F1-score và Confusion Matrix.\n"
    )

    for ten_mo_hinh, df in du_lieu_mo_hinh.items():

        tep.write(f"\n========== {ten_mo_hinh} ==========\n")

        tep.write(f"Tổng số ảnh: {len(df)}\n")

        trang_thai = (
            df[COT_TRANG_THAI]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        thanh_cong = trang_thai.isin(
            ["thanh cong", "thành công", "success"]
        ).sum()

        tep.write(f"Số ảnh xử lý thành công: {thanh_cong}\n")

        tep.write(
            f"Số ảnh xử lý thất bại: {len(df) - thanh_cong}\n"
        )

        tep.write(
            f"Tỷ lệ thành công: "
            f"{thanh_cong / len(df) * 100:.2f}%\n"
            if len(df) > 0 else "Tỷ lệ thành công: 0%\n"
        )

        tep.write(
            f"Độ tin cậy trung bình: "
            f"{df[COT_DO_TIN_CAY].mean():.2f}%\n"
        )

        tep.write(
            f"Độ tin cậy cao nhất: "
            f"{df[COT_DO_TIN_CAY].max():.2f}%\n"
        )

        tep.write(
            f"Độ tin cậy thấp nhất: "
            f"{df[COT_DO_TIN_CAY].min():.2f}%\n"
        )

        tep.write("\nThống kê theo loại tiền:\n")
        tep.write(
            df[COT_LOAI_TIEN].value_counts().to_string()
        )

        tep.write("\n\nThống kê theo mệnh giá:\n")
        tep.write(
            df[COT_MENH_GIA].value_counts().to_string()
        )

        tep.write("\n\nThống kê theo loại tiền - mệnh giá:\n")
        tep.write(
            df["Loại tiền - Mệnh giá"].value_counts().to_string()
        )

        tep.write("\n")


# ==========================================
# 16. KẾT THÚC
# ==========================================

print("\n==========================================")
print("HOÀN THÀNH THỐNG KÊ 3 MÔ HÌNH YOLO!")
print("==========================================")

print("\nThư mục chứa kết quả:")
print(THU_MUC_KET_QUA)

print("\nCác tệp kết quả đã tạo:")
print("1. bang_so_sanh_3_mo_hinh.csv")
print("2. thong_ke_loai_tien.csv")
print("3. thong_ke_menh_gia.csv")
print("4. thong_ke_loai_tien_menh_gia.csv")
print("5. so_sanh_do_tin_cay.png")
print("6. so_sanh_trang_thai_du_doan.png")
print("7. so_sanh_loai_tien.png")
print("8. so_sanh_menh_gia.png")
print("9. so_sanh_loai_tien_menh_gia.png")
print("10. bao_cao_thong_ke.txt")
print("11. Các biểu đồ phân bố độ tin cậy của từng mô hình")
