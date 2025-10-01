import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import date

# =============================
# App config & title
# =============================
st.set_page_config(page_title="PACKING TRACKER", page_icon="📁", layout="wide")
st.title("PACKING TRACKER")

# =============================
# Helpers
# =============================
def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).upper().strip())

def _suggest(cols, candidates):
    cols_n = [_norm(c) for c in cols]
    for cand in candidates:
        n = _norm(cand)
        if n in cols_n:
            return cols[cols_n.index(n)]
    return cols[0] if cols else None

# =============================
# Cached processing (no widgets)
# =============================
@st.cache_data(show_spinner=False)
def build_joined_df(
    resi_bytes: bytes,
    sku_bytes: bytes,
    mapping_tuple: tuple,
    special_bytes: bytes | None,
    handling_bytes: bytes | None,
) -> pd.DataFrame:
    """
    - Mapping header
    - Durasi antar scan (detik) per (NAMA, TANGGAL); last scan = 0 detik
    - Join RESI x SKU
    - SKU spesial & handling khusus dari file (opsional) atau default
    - PESANAN (Satuan/Spesial/Biasa/Campuran)
    - STATUS per RESI: (durasi_detik + bonus_handling) <= (total_qty*30)
    """

    # ---------- read base files ----------
    resi_tgl, resi_jam, resi_nama, resi_no, sku_no, sku_sku, sku_qty = mapping_tuple
    df_resi = pd.read_excel(BytesIO(resi_bytes))
    df_sku  = pd.read_excel(BytesIO(sku_bytes))

    df_resi.columns = df_resi.columns.str.strip().str.upper()
    df_sku.columns  = df_sku.columns.str.strip().str.upper()

    df_resi = df_resi.rename(columns={
        resi_tgl: "TANGGAL SCAN",
        resi_jam: "JAM SCAN",
        resi_nama:"NAMA",
        resi_no:  "NO RESI",
    })
    df_sku = df_sku.rename(columns={
        sku_no:  "NO RESI",
        sku_sku: "SKU",
        sku_qty: "QTY",
    })

    df_resi["NO RESI"] = df_resi["NO RESI"].astype(str).str.strip()
    df_sku["NO RESI"]  = df_sku["NO RESI"].astype(str).str.strip()
    df_sku["QTY"] = pd.to_numeric(df_sku["QTY"], errors="coerce").fillna(0).astype(int)

    # ---------- read optional special & handling ----------
    # defaults
    special_set_norm_default = {_norm(x) for x in ["C222-AK328-2","C222-AK328-8","C222-AK328-7","C222-AK328-1","C222-AK328-5"]}
    handling_map_norm_default = {_norm("BM-AKS28-1"): 60}

    def read_special_set(b: bytes | None) -> set[str]:
        if not b:
            return special_set_norm_default
        d = pd.read_excel(BytesIO(b))
        d.columns = d.columns.str.strip().str.upper()
        sku_col = _suggest(d.columns.tolist(), ["SKU SPESIAL"])
        return {_norm(x) for x in d[sku_col].dropna().astype(str).tolist()}

    def read_handling_map(b: bytes | None) -> dict[str, float]:
        if not b:
            return handling_map_norm_default
        d = pd.read_excel(BytesIO(b))
        d.columns = d.columns.str.strip().str.upper()
        sku_col = _suggest(d.columns.tolist(), ["SKU KHUSUS"])
        bonus_col = _suggest(d.columns.tolist(), ["BONUS_DETIK","BONUS SEC","BONUS(DETIK)","BONUS_SEC","BONUS"])
        d["_SKU_NORM"] = d[sku_col].astype(str).map(_norm)
        d["_BONUS"] = pd.to_numeric(d[bonus_col], errors="coerce").fillna(0).astype(float)
        return dict(d[["_SKU_NORM","_BONUS"]].values)

    special_set_norm = read_special_set(special_bytes)
    handling_map_norm = read_handling_map(handling_bytes)

    # ---------- durasi antar scan (detik) ----------
    df_resi["TANGGAL SCAN"] = pd.to_datetime(df_resi["TANGGAL SCAN"], errors="coerce")
    jam_str = df_resi["JAM SCAN"].astype(str).str.strip().str.replace(r"^(\d{1,2}:\d{2})$", r"\1:00", regex=True)
    jam_parsed = pd.to_datetime(jam_str, format="%H:%M:%S", errors="coerce")

    df_resi["TGL_NORM"]  = df_resi["TANGGAL SCAN"].dt.normalize()
    df_resi["SCAN_DT"]   = pd.to_datetime(df_resi["TGL_NORM"].dt.strftime("%Y-%m-%d") + " " + jam_parsed.dt.strftime("%H:%M:%S"), errors="coerce")

    df_resi = df_resi.sort_values(["NAMA","TGL_NORM","SCAN_DT"])
    df_resi["SCAN_NEXT"] = df_resi.groupby(["NAMA","TGL_NORM"])["SCAN_DT"].shift(-1)
    df_resi["DURASI_SEC"] = (df_resi["SCAN_NEXT"] - df_resi["SCAN_DT"]).dt.total_seconds().fillna(0).clip(lower=0)
    df_resi["DURASI"] = (df_resi["DURASI_SEC"] // 60).astype(int)

    df_resi = df_resi.drop(columns=["SCAN_NEXT","TGL_NORM","SCAN_DT"], errors="ignore")

    # ---------- join ----------
    df = pd.merge(df_resi, df_sku, on="NO RESI", how="inner")
    df["SKU_NORM"] = df["SKU"].astype(str).map(_norm)

    # ---------- aggregate per resi ----------
    sku_unik = df.groupby("NO RESI")["SKU_NORM"].nunique().reset_index(name="SKU_UNIK")
    qty_total = df.groupby("NO RESI")["QTY"].sum().reset_index(name="QTY_TOTAL")
    only_sku_norm = df.groupby("NO RESI")["SKU_NORM"].apply(lambda s: next(iter(set(s))) if len(set(s))==1 else None).reset_index(name="ONLY_SKU_NORM")
    dur_resi_sec = df.groupby("NO RESI")["DURASI_SEC"].first().reset_index(name="DURASI_PER_RESI_SEC")

    def sum_bonus(sec_series) -> float:
        return float(sum(handling_map_norm.get(x, 0.0) for x in set(sec_series)))

    bonus_sec = df.groupby("NO RESI")["SKU_NORM"].apply(sum_bonus).reset_index(name="BONUS_SEC")

    agg = sku_unik.merge(qty_total, on="NO RESI") \
                  .merge(only_sku_norm, on="NO RESI") \
                  .merge(dur_resi_sec, on="NO RESI") \
                  .merge(bonus_sec, on="NO RESI")

    # ---------- pesanan & status ----------
    def classify(sku_u: int, qty_t: int, only_norm: str) -> str:
        if pd.isna(sku_u):
            return "Tidak Diketahui"
        if sku_u == 1:
            return "Spesial" if (only_norm in special_set_norm) else "Satuan"
        return "Biasa" if qty_t <= 3 else "Campuran"

    agg["PESANAN_RES"] = [classify(su, qt, osku) for su, qt, osku in zip(agg["SKU_UNIK"], agg["QTY_TOTAL"], agg["ONLY_SKU_NORM"])]

    actual_sec  = agg["DURASI_PER_RESI_SEC"].fillna(0).astype(float) 
    allowed_sec = (agg["QTY_TOTAL"].fillna(0).astype(float) * 30.0) + agg["BONUS_SEC"].fillna(0).astype(float)
    agg["STATUS_RES"] = ["OK" if a <= b else "LAMBAT" for a, b in zip(actual_sec, allowed_sec)]

    # ---------- map back & cleanup ----------
    df = df.merge(agg[["NO RESI","PESANAN_RES","STATUS_RES"]], on="NO RESI", how="left")
    df = df.rename(columns={"PESANAN_RES":"PESANAN","STATUS_RES":"STATUS"})
    df = df.drop(columns=["DURASI_SEC","SKU_NORM"], errors="ignore")

    preferred = ["TANGGAL SCAN","JAM SCAN","NAMA","NO RESI","PESANAN","STATUS","DURASI","SKU","QTY"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]

# =============================
# Upload
# =============================
u1, u2 = st.columns(2)
with u1:
    resi_file = st.file_uploader("Upload RESI DATA.xlsx", type=["xlsx"])
with u2:
    sku_file  = st.file_uploader("Upload SKU DATA.xlsx",  type=["xlsx"])

u3, u4 = st.columns(2)
with u3:
    special_file  = st.file_uploader("Upload SKU Spesial", type=["xlsx"])
with u4:
    handling_file = st.file_uploader("Upload SKU Handling Khusus", type=["xlsx"])

with st.sidebar:
    if st.button("♻️ Clear cache data"):
        st.cache_data.clear()
        st.success("Cache dibersihkan.")

# =============================
# Main
# =============================
if resi_file and sku_file:
    # header mapping (no cache)
    df_resi_raw = pd.read_excel(resi_file)
    df_sku_raw  = pd.read_excel(sku_file)
    resi_cols = df_resi_raw.columns.str.strip().str.upper().tolist()
    sku_cols  = df_sku_raw.columns.str.strip().str.upper().tolist()

    resi_tgl_def = _suggest(resi_cols, ["TANGGAL SCAN","Tgl scan"])
    resi_jam_def = _suggest(resi_cols, ["JAM SCAN","Jam scan"])
    resi_nama_def= _suggest(resi_cols, ["NAMA","Nama"])
    resi_no_def  = _suggest(resi_cols, ["NO RESI","No Resi"])

    sku_no_def   = _suggest(sku_cols,  ["NO RESI", "NO_RESI"])
    sku_sku_def  = _suggest(sku_cols,  ["SKU","KODE SKU"])
    sku_qty_def  = _suggest(sku_cols,  ["QTY","KUANTITAS"])

    with st.expander("🔧 Penyesuaian Header"):
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**RESI DATA.xlsx**")
            resi_tgl = st.selectbox("Kolom Tanggal", options=resi_cols, index=resi_cols.index(resi_tgl_def))
            resi_jam = st.selectbox("Kolom Jam",     options=resi_cols, index=resi_cols.index(resi_jam_def))
            resi_nama= st.selectbox("Kolom Nama",    options=resi_cols, index=resi_cols.index(resi_nama_def))
            resi_no  = st.selectbox("Kolom No Resi", options=resi_cols, index=resi_cols.index(resi_no_def))
        with cB:
            st.markdown("**SKU DATA.xlsx**")
            sku_no   = st.selectbox("Kolom No Resi (SKU)", options=sku_cols, index=sku_cols.index(sku_no_def))
            sku_sku  = st.selectbox("Kolom SKU",           options=sku_cols, index=sku_cols.index(sku_sku_def))
            sku_qty  = st.selectbox("Kolom QTY",           options=sku_cols, index=sku_cols.index(sku_qty_def))

    mapping_tuple = (resi_tgl, resi_jam, resi_nama, resi_no, sku_no, sku_sku, sku_qty)

    df_base = build_joined_df(
        resi_file.getvalue(),
        sku_file.getvalue(),
        mapping_tuple,
        special_file.getvalue() if special_file else None,
        handling_file.getvalue() if handling_file else None,
        )

    # Filters
    st.subheader("Filter")
    f1, f2, f3 = st.columns([1,1,1])
    t_series = pd.to_datetime(df_base["TANGGAL SCAN"], errors="coerce").dt.date
    valid_dates = t_series.dropna()
    if len(valid_dates) > 0:
        min_date, max_date = valid_dates.min(), valid_dates.max()
        default_date = max_date
    else:
        today = date.today()
        min_date = max_date = default_date = today

    with f1:
        sel_date = st.date_input("Tanggal", value=default_date, min_value=min_date, max_value=max_date)
    with f2:
        mode_nama = st.selectbox("Nama", ["Semua","Pilih..."], index=0)
        sel_names = st.multiselect("Pilih Nama QC", options=sorted(df_base["NAMA"].dropna().unique().tolist()),
                                   placeholder="Pilih satu atau lebih nama") if mode_nama=="Pilih..." else []
    with f3:
        jenis = st.selectbox("Jenis", ["Semua","Satuan","Spesial","Biasa","Campuran"], index=0)

    df = df_base.copy()
    df = df[t_series == sel_date]
    if sel_names:
        df = df[df["NAMA"].isin(sel_names)]
    if jenis != "Semua":
        df = df[df["PESANAN"] == jenis]

    # Metrics per resi unik
    resi_level = df[["NO RESI","PESANAN","STATUS"]].drop_duplicates(subset="NO RESI")
    total_resi = resi_level["NO RESI"].nunique()
    ok_count   = int((resi_level["STATUS"] == "OK").sum())
    bad_count  = int((resi_level["STATUS"] == "LAMBAT").sum())
    st.metric("TOTAL RESI", f"{total_resi:,}")

    # Status color
    def color_status(v: str) -> str:
        if v=="OK": return "color: green; font-weight: 700;"
        if v=="LAMBAT": return "color: red; font-weight: 700;"
        return ""
    styled = df.style.map(color_status, subset=["STATUS"])

    st.dataframe(styled, width="stretch")

    # Summary per resi
    pesanan_counts = resi_level["PESANAN"].value_counts(dropna=False).to_dict()
    summary = pd.DataFrame([
        {"METRIC":"Tanggal", "VALUE": str(sel_date)},
        {"METRIC":"Total Resi", "VALUE": int(total_resi)},
        {"METRIC":"Jumlah Pesanan - Satuan",   "VALUE": int(pesanan_counts.get("Satuan",0))},
        {"METRIC":"Jumlah Pesanan - Spesial",  "VALUE": int(pesanan_counts.get("Spesial",0))},
        {"METRIC":"Jumlah Pesanan - Biasa",    "VALUE": int(pesanan_counts.get("Biasa",0))},
        {"METRIC":"Jumlah Pesanan - Campuran", "VALUE": int(pesanan_counts.get("Campuran",0))},
        {"METRIC":"Jumlah Status - OK",        "VALUE": ok_count},
        {"METRIC":"Jumlah Status - LAMBAT",    "VALUE": bad_count},
    ])
    if total_resi != ok_count + bad_count:
        st.warning("Perhatian: total resi ≠ OK + LAMBAT (cek data/hitungan).")

    # Export
    xlsx_buffer = BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
        sheet = "PACKING"
        startrow = 2
        df.to_excel(writer, index=False, sheet_name=sheet, startrow=startrow)
        ws = writer.sheets[sheet]
        ws.write(0, 0, "Total Resi")
        ws.write(0, 1, int(total_resi))

        if "STATUS" in df.columns:
            status_col = df.columns.get_loc("STATUS")
            nrows = len(df)
            workbook = writer.book
            fmt_ok  = workbook.add_format({"font_color": "green", "bold": True})
            fmt_bad = workbook.add_format({"font_color": "red",   "bold": True})
            first_data_row = startrow + 1
            last_data_row  = startrow + nrows
            ws.conditional_format(first_data_row, status_col, last_data_row, status_col,
                                  {"type":"text","criteria":"containing","value":"OK","format":fmt_ok})
            ws.conditional_format(first_data_row, status_col, last_data_row, status_col,
                                  {"type":"text","criteria":"containing","value":"LAMBAT","format":fmt_bad})

        summary.to_excel(writer, index=False, sheet_name="SUMMARY")

    st.download_button(
        label="Download Excel",
        data=xlsx_buffer.getvalue(),
        file_name="PACKING TRACKER.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("Silakan upload kedua file terlebih dahulu.")