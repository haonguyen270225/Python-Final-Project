from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================

# Thư mục gốc của dự án Python-Final-Project
THU_MUC_GOC = Path(__file__).resolve().parent

# Thư mục chứa 3 file CSV kết quả dự đoán Random Forest
THU_MUC_CSV = THU_MUC_GOC / "model_random_forest"

# Thư mục lưu các bảng thống kê, biểu đồ và báo cáo
THU_MUC_KET_QUA = (
    THU_MUC_GOC / "ket_qua_thong_ke" / "Random_Forest"
)
THU_MUC_KET_QUA.mkdir(parents=True, exist_ok=True)


# Danh sách 3 file CSV cần thống kê
DANH_SACH_MO_HINH = {
    "Mô hình 1": THU_MUC_CSV / "ket_qua_du_doan_Random_Forest_01.csv",
    "Mô hình 2": THU_MUC_CSV / "ket_qua_du_doan_Random_Forest_02.csv",
    "Mô hình 3": THU_MUC_CSV / "ket_qua_du_doan_Random_Forest_03.csv",
}


# =========================================================
# 2. KHAI BÁO TÊN CỘT TRONG FILE CSV RANDOM FOREST
# =========================================================

# Các tên cột được lấy đúng theo cấu trúc CSV mà chương trình
# dự đoán Random Forest đang xuất ra.
COT_TEN_ANH = "image_name"
COT_DUONG_DAN = "image_path"
COT_LOAI_TIEN = "currency"
COT_MENH_GIA = "denomination"
COT_DO_TIN_CAY = "confidence_percentage"
COT_TRANG_THAI = "status"

CAC_COT_CAN_THIET = [
    COT_TEN_ANH,
    COT_DUONG_DAN,
    COT_LOAI_TIEN,
    COT_MENH_GIA,
    COT_DO_TIN_CAY,
    COT_TRANG_THAI,
]


# =========================================================
# 3. HÀM ĐỌC VÀ CHUẨN HÓA FILE CSV
# =========================================================

def doc_du_lieu_csv(ten_mo_hinh, duong_dan_csv):
    """Đọc, kiểm tra và chuẩn hóa dữ liệu của một file CSV."""

    if not duong_dan_csv.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp CSV:\n{duong_dan_csv}"
        )

    df = pd.read_csv(
        duong_dan_csv,
        encoding="utf-8-sig"
    )

    # Loại bỏ khoảng trắng thừa trong tên cột
    df.columns = df.columns.astype(str).str.strip()

    # Kiểm tra các cột bắt buộc
    cot_thieu = [
        cot for cot in CAC_COT_CAN_THIET
        if cot not in df.columns
    ]

    if cot_thieu:
        raise ValueError(
            f"Tệp {duong_dan_csv.name} thiếu cột: {cot_thieu}\n"
            f"Các cột hiện có: {df.columns.tolist()}"
        )

    if df.empty:
        raise ValueError(
            f"Tệp {duong_dan_csv.name} không có dữ liệu."
        )

    # -----------------------------------------------------
    # Chuyển confidence_percentage thành kiểu số.
    # Ví dụ: "100.00%" -> 100.00
    #        "75.00%"  -> 75.00
    # -----------------------------------------------------
    df[COT_DO_TIN_CAY] = (
        df[COT_DO_TIN_CAY]
        .astype("string")
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    df[COT_DO_TIN_CAY] = pd.to_numeric(
        df[COT_DO_TIN_CAY],
        errors="coerce"
    )

    # Nếu confidence_percentage bị trống nhưng confidence có dữ liệu,
    # sử dụng confidence (dạng 0-1) và chuyển sang phần trăm.
    if "confidence" in df.columns:
        confidence_goc = pd.to_numeric(
            df["confidence"],
            errors="coerce"
        )

        mask_can_bo_sung = (
            df[COT_DO_TIN_CAY].isna()
            & confidence_goc.notna()
        )

        df.loc[mask_can_bo_sung, COT_DO_TIN_CAY] = (
            confidence_goc.loc[mask_can_bo_sung] * 100
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

    # Chuẩn hóa trạng thái
    df[COT_TRANG_THAI] = (
        df[COT_TRANG_THAI]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Tạo nhãn kết hợp loại tiền và mệnh giá
    df["Loại tiền - Mệnh giá"] = (
        df[COT_LOAI_TIEN]
        + " - "
        + df[COT_MENH_GIA]
    )

    print(f"\n========== {ten_mo_hinh} ==========")
    print("File:", duong_dan_csv.name)
    print("Tổng số ảnh:", len(df))
    print("Các cột dữ liệu:", df.columns.tolist())
    print("\n5 dòng dữ liệu đầu tiên:")
    print(df.head().to_string(index=False))

    return df


# =========================================================
# 4. ĐỌC DỮ LIỆU CỦA 3 MÔ HÌNH
# =========================================================

du_lieu_mo_hinh = {}

for ten_mo_hinh, duong_dan_csv in DANH_SACH_MO_HINH.items():
    du_lieu_mo_hinh[ten_mo_hinh] = doc_du_lieu_csv(
        ten_mo_hinh,
        duong_dan_csv
    )


# =========================================================
# 5. THỐNG KÊ KẾT QUẢ TỪNG MÔ HÌNH
# =========================================================

danh_sach_ket_qua = []

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    tong_so_anh = len(df)

    # Các trạng thái được xem là xử lý thành công
    trang_thai_thanh_cong = [
        "thanh cong",
        "thành công",
        "success",
    ]

    so_anh_thanh_cong = df[COT_TRANG_THAI].isin(
        trang_thai_thanh_cong
    ).sum()

    # Số ảnh xử lý thất bại
    so_anh_that_bai = tong_so_anh - so_anh_thanh_cong

    # Thống kê độ tin cậy, bỏ qua các giá trị không hợp lệ
    do_tin_cay = df[COT_DO_TIN_CAY].dropna()

    if not do_tin_cay.empty:
        do_tin_cay_trung_binh = do_tin_cay.mean()
        do_tin_cay_cao_nhat = do_tin_cay.max()
        do_tin_cay_thap_nhat = do_tin_cay.min()
    else:
        do_tin_cay_trung_binh = float("nan")
        do_tin_cay_cao_nhat = float("nan")
        do_tin_cay_thap_nhat = float("nan")

    ti_le_thanh_cong = (
        so_anh_thanh_cong / tong_so_anh * 100
        if tong_so_anh > 0 else 0
    )

    ket_qua = {
        "Mô hình": ten_mo_hinh,
        "Tổng số ảnh": tong_so_anh,
        "Số ảnh thành công": int(so_anh_thanh_cong),
        "Số ảnh thất bại": int(so_anh_that_bai),
        "Tỷ lệ thành công (%)": round(ti_le_thanh_cong, 2),
        "Độ tin cậy trung bình (%)": round(do_tin_cay_trung_binh, 2),
        "Độ tin cậy cao nhất (%)": round(do_tin_cay_cao_nhat, 2),
        "Độ tin cậy thấp nhất (%)": round(do_tin_cay_thap_nhat, 2),
    }

    danh_sach_ket_qua.append(ket_qua)

    print(f"\n========== THỐNG KÊ {ten_mo_hinh} ==========")
    for ten_chi_so, gia_tri in ket_qua.items():
        print(f"{ten_chi_so}: {gia_tri}")


# =========================================================
# 6. TẠO BẢNG SO SÁNH 3 MÔ HÌNH
# =========================================================

bang_ket_qua = pd.DataFrame(danh_sach_ket_qua)

print("\n========== BẢNG SO SÁNH 3 MÔ HÌNH RANDOM FOREST ==========")
print(bang_ket_qua.to_string(index=False))

bang_ket_qua.to_csv(
    THU_MUC_KET_QUA / "bang_so_sanh_3_mo_hinh.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 7. THỐNG KÊ SỐ LƯỢNG THEO LOẠI TIỀN
# =========================================================

danh_sach_loai_tien = []

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    thong_ke = df[COT_LOAI_TIEN].value_counts()

    for loai_tien, so_luong in thong_ke.items():
        danh_sach_loai_tien.append({
            "Mô hình": ten_mo_hinh,
            "Loại tiền": loai_tien,
            "Số lượng ảnh": int(so_luong),
        })

bang_loai_tien = pd.DataFrame(danh_sach_loai_tien)

print("\n========== THỐNG KÊ THEO LOẠI TIỀN ==========")
print(bang_loai_tien.to_string(index=False))

bang_loai_tien.to_csv(
    THU_MUC_KET_QUA / "thong_ke_loai_tien.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 8. THỐNG KÊ SỐ LƯỢNG THEO MỆNH GIÁ
# =========================================================

danh_sach_menh_gia = []

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    thong_ke = df[COT_MENH_GIA].value_counts()

    for menh_gia, so_luong in thong_ke.items():
        danh_sach_menh_gia.append({
            "Mô hình": ten_mo_hinh,
            "Mệnh giá": menh_gia,
            "Số lượng ảnh": int(so_luong),
        })

bang_menh_gia = pd.DataFrame(danh_sach_menh_gia)

print("\n========== THỐNG KÊ THEO MỆNH GIÁ ==========")
print(bang_menh_gia.to_string(index=False))

bang_menh_gia.to_csv(
    THU_MUC_KET_QUA / "thong_ke_menh_gia.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 9. THỐNG KÊ THEO LOẠI TIỀN VÀ MỆNH GIÁ
# =========================================================

danh_sach_nhan = []

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    thong_ke = df["Loại tiền - Mệnh giá"].value_counts()

    for nhan, so_luong in thong_ke.items():
        danh_sach_nhan.append({
            "Mô hình": ten_mo_hinh,
            "Loại tiền - Mệnh giá": nhan,
            "Số lượng ảnh": int(so_luong),
        })

bang_nhan = pd.DataFrame(danh_sach_nhan)

print("\n========== THỐNG KÊ LOẠI TIỀN - MỆNH GIÁ ==========")
print(bang_nhan.to_string(index=False))

bang_nhan.to_csv(
    THU_MUC_KET_QUA / "thong_ke_loai_tien_menh_gia.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 10. CẤU HÌNH BIỂU ĐỒ
# =========================================================

sns.set_theme(style="whitegrid")


# =========================================================
# 11. BIỂU ĐỒ SO SÁNH ĐỘ TIN CẬY
# =========================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=bang_ket_qua,
    x="Mô hình",
    y="Độ tin cậy trung bình (%)"
)

plt.title(
    "SO SÁNH ĐỘ TIN CẬY TRUNG BÌNH CỦA 3 MÔ HÌNH RANDOM FOREST",
    fontsize=13
)
plt.xlabel("Mô hình")
plt.ylabel("Độ tin cậy trung bình (%)")
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig(
    THU_MUC_KET_QUA / "so_sanh_do_tin_cay.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# =========================================================
# 12. BIỂU ĐỒ SO SÁNH TRẠNG THÁI DỰ ĐOÁN
# =========================================================

bang_trang_thai = bang_ket_qua.melt(
    id_vars="Mô hình",
    value_vars=[
        "Số ảnh thành công",
        "Số ảnh thất bại",
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
    "SO SÁNH TRẠNG THÁI XỬ LÝ ẢNH CỦA 3 MÔ HÌNH RANDOM FOREST",
    fontsize=13
)
plt.xlabel("Mô hình")
plt.ylabel("Số lượng ảnh")
plt.tight_layout()

plt.savefig(
    THU_MUC_KET_QUA / "so_sanh_trang_thai_du_doan.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# =========================================================
# 13. BIỂU ĐỒ SO SÁNH SỐ LƯỢNG THEO LOẠI TIỀN
# =========================================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=bang_loai_tien,
    x="Loại tiền",
    y="Số lượng ảnh",
    hue="Mô hình"
)

plt.title(
    "SO SÁNH SỐ LƯỢNG DỰ ĐOÁN THEO LOẠI TIỀN",
    fontsize=13
)
plt.xlabel("Loại tiền")
plt.ylabel("Số lượng ảnh")
plt.tight_layout()

plt.savefig(
    THU_MUC_KET_QUA / "so_sanh_loai_tien.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# =========================================================
# 14. BIỂU ĐỒ SO SÁNH SỐ LƯỢNG THEO MỆNH GIÁ
# =========================================================

plt.figure(figsize=(12, 6))

sns.barplot(
    data=bang_menh_gia,
    x="Mệnh giá",
    y="Số lượng ảnh",
    hue="Mô hình"
)

plt.title(
    "SO SÁNH SỐ LƯỢNG DỰ ĐOÁN THEO MỆNH GIÁ",
    fontsize=13
)
plt.xlabel("Mệnh giá")
plt.ylabel("Số lượng ảnh")
plt.tight_layout()

plt.savefig(
    THU_MUC_KET_QUA / "so_sanh_menh_gia.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# =========================================================
# 15. BIỂU ĐỒ SO SÁNH LOẠI TIỀN - MỆNH GIÁ
# =========================================================

plt.figure(figsize=(14, 7))

sns.barplot(
    data=bang_nhan,
    x="Loại tiền - Mệnh giá",
    y="Số lượng ảnh",
    hue="Mô hình"
)

plt.title(
    "SO SÁNH SỐ LƯỢNG DỰ ĐOÁN THEO LOẠI TIỀN VÀ MỆNH GIÁ",
    fontsize=13
)
plt.xlabel("Loại tiền - Mệnh giá")
plt.ylabel("Số lượng ảnh")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    THU_MUC_KET_QUA / "so_sanh_loai_tien_menh_gia.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()


# =========================================================
# 16. BIỂU ĐỒ PHÂN BỐ ĐỘ TIN CẬY CỦA TỪNG MÔ HÌNH
# =========================================================

for ten_mo_hinh, df in du_lieu_mo_hinh.items():

    du_lieu_do_tin_cay = df[COT_DO_TIN_CAY].dropna()

    if du_lieu_do_tin_cay.empty:
        print(
            f"Bỏ qua biểu đồ độ tin cậy của {ten_mo_hinh}: "
            "không có dữ liệu hợp lệ."
        )
        continue

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=du_lieu_do_tin_cay,
        bins=20,
        kde=(len(du_lieu_do_tin_cay) > 1)
    )

    plt.title(
        f"PHÂN BỐ ĐỘ TIN CẬY - {ten_mo_hinh}",
        fontsize=13
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
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()


# =========================================================
# 17. XUẤT BÁO CÁO CHI TIẾT
# =========================================================

duong_dan_bao_cao = THU_MUC_KET_QUA / "bao_cao_thong_ke.txt"

with open(
    duong_dan_bao_cao,
    "w",
    encoding="utf-8"
) as tep:

    tep.write("BÁO CÁO THỐNG KÊ KẾT QUẢ 3 MÔ HÌNH RANDOM FOREST\n")
    tep.write("=" * 65 + "\n")

    tep.write("\nLƯU Ý:\n")
    tep.write(
        "Các tệp CSV hiện tại chứa kết quả dự đoán, chưa có nhãn thực tế "
        "(Ground Truth) để đối chiếu.\n"
    )
    tep.write(
        "Do đó, báo cáo chỉ thống kê số lượng ảnh, trạng thái xử lý "
        "và độ tin cậy dự đoán.\n"
    )
    tep.write(
        "Chưa thể tính chính xác Accuracy, Precision, Recall, F1-score "
        "và Confusion Matrix nếu không có nhãn thực tế.\n"
    )

    for ten_mo_hinh, df in du_lieu_mo_hinh.items():

        tong_so_anh = len(df)
        thanh_cong = df[COT_TRANG_THAI].isin(
            ["thanh cong", "thành công", "success"]
        ).sum()

        do_tin_cay = df[COT_DO_TIN_CAY].dropna()

        tep.write(f"\n\n========== {ten_mo_hinh} ==========\n")
        tep.write(f"Tổng số ảnh: {tong_so_anh}\n")
        tep.write(f"Số ảnh xử lý thành công: {int(thanh_cong)}\n")
        tep.write(
            f"Số ảnh xử lý thất bại: {tong_so_anh - int(thanh_cong)}\n"
        )

        ti_le = (
            thanh_cong / tong_so_anh * 100
            if tong_so_anh > 0 else 0
        )
        tep.write(f"Tỷ lệ xử lý thành công: {ti_le:.2f}%\n")

        if not do_tin_cay.empty:
            tep.write(
                f"Độ tin cậy trung bình: {do_tin_cay.mean():.2f}%\n"
            )
            tep.write(
                f"Độ tin cậy cao nhất: {do_tin_cay.max():.2f}%\n"
            )
            tep.write(
                f"Độ tin cậy thấp nhất: {do_tin_cay.min():.2f}%\n"
            )
        else:
            tep.write("Không có dữ liệu độ tin cậy hợp lệ.\n")

        tep.write("\nThống kê theo loại tiền:\n")
        tep.write(df[COT_LOAI_TIEN].value_counts().to_string())

        tep.write("\n\nThống kê theo mệnh giá:\n")
        tep.write(df[COT_MENH_GIA].value_counts().to_string())

        tep.write("\n\nThống kê theo loại tiền - mệnh giá:\n")
        tep.write(
            df["Loại tiền - Mệnh giá"].value_counts().to_string()
        )
        tep.write("\n")


# =========================================================
# 18. KẾT THÚC
# =========================================================

print("\n" + "=" * 60)
print("HOÀN THÀNH THỐNG KÊ 3 MÔ HÌNH RANDOM FOREST!")
print("=" * 60)

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
print("10. Các biểu đồ phân bố độ tin cậy của từng mô hình")
print("11. bao_cao_thong_ke.txt")
