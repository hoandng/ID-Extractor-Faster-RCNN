import json
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────
# Cau hinh trang
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "CCCD Extractor",
    layout     = "wide",
)

# ─────────────────────────────────────────────────────────
# CSS tuy chinh
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Bo vien cho cac card ket qua */
    .result-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 4px solid #1f77b4;
    }
    .result-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
    .result-value {
        font-size: 18px;
        font-weight: 600;
        color: #1a1a2e;
    }
    /* Badge trang thai */
    .badge-success {
        background: #d4edda; color: #155724;
        padding: 3px 10px; border-radius: 12px;
        font-size: 13px; font-weight: 600;
    }
    .badge-error {
        background: #f8d7da; color: #721c24;
        padding: 3px 10px; border-radius: 12px;
        font-size: 13px; font-weight: 600;
    }
    /* An hamburger menu mac dinh cua Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Load model (cache de khong load lai moi lan)
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Load CCCDPipeline 1 lan duy nhat, cache lai.
    st.cache_resource: phu hop cho object ton RAM nhu model PyTorch.
    """
    try:
        from src.inference import CCCDPipeline
        pipe = CCCDPipeline(
            card_model   = "weights/card/best.pth",
            corner_model = "weights/corner/best.pth",
            field_model  = "weights/field/best.pth",
        )
        return pipe, None
    except FileNotFoundError as e:
        return None, f"Khong tim thay weights: {e}"
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────
# Ten hien thi cac truong
# ─────────────────────────────────────────────────────────
FIELD_DISPLAY = {
    "id":        "Số CCCD",
    "hoten":     "Họ và tên",
    "ngaysinh":  "Ngày sinh",
    "gioitinh":  "Giới tính",
    "quoctich":  "Quốc tịch",
    "quequan":   "Quê quán",
    "diachi":    "Nơi thường trú",
    "giatriden": "Có giá trị đến",
}

# Thu tu hien thi cac truong
FIELD_ORDER = [
    "id", "hoten", "ngaysinh", "gioitinh",
    "quoctich", "quequan", "diachi", "giatriden",
]


# ─────────────────────────────────────────────────────────
# Trang 1: OCR
# ─────────────────────────────────────────────────────────
def page_ocr():
    st.title("")
    st.caption("Tải ảnh CCCD lên để trích xuất thông tin tự động")

    # Load model
    with st.spinner("Đang tải model..."):
        pipe, err = load_pipeline()

    if err:
        st.error(f"❌ Lỗi tải model: {err}")
        st.info("Hãy chắc chắn đã train xong và có file weights/*/best.pth")
        return

    st.success("Model sẵn sàng", icon="✅")
    st.divider()

    # ── Khu vuc upload ──────────────────────────────────
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("Upload ảnh")
        uploaded = st.file_uploader(
            label       = "Kéo thả hoặc click để chọn ảnh",
            type        = ["jpg", "jpeg", "png"],
            label_visibility = "collapsed",
        )

        if uploaded:
            # Hien thi anh preview
            img_pil = Image.open(uploaded)
            st.image(img_pil, caption="Ảnh đã upload", use_container_width=True)

            # Nut xu ly
            if st.button("Trích xuất thông tin",
                         type="primary", use_container_width=True):
                # Chuyen PIL -> numpy BGR cho OpenCV
                img_np  = np.array(img_pil.convert("RGB"))
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                with st.spinner("Đang xử lý..."):
                    t0     = time.time()
                    result = pipe.run(img_bgr)
                    elapsed = time.time() - t0

                # Luu ket qua vao session_state de hien thi ben phai
                st.session_state["result"]  = result
                st.session_state["elapsed"] = elapsed
                st.session_state["img_pil"] = img_pil

    # ── Hien thi ket qua ─────────────────────────────────
    with col_result:
        st.subheader("Kết quả")

        if "result" not in st.session_state:
            st.info("Upload ảnh và nhấn **Trích xuất** để xem kết quả")
            return

        result  = st.session_state["result"]
        elapsed = st.session_state["elapsed"]
        status  = result.get("status")

        # Hien thi trang thai
        if status == "success":
            st.markdown('<span class="badge-success"> Thành công</span>',
                        unsafe_allow_html=True)
            st.caption(f"⏱ Thời gian xử lý: {elapsed:.2f}s")
            st.divider()

            data = result.get("data", {})
            for key in FIELD_ORDER:
                val   = data.get(key, "—")
                label = FIELD_DISPLAY.get(key, key)
                st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">{label}</div>
                        <div class="result-value">{val}</div>
                    </div>
                """, unsafe_allow_html=True)

            st.divider()

            # Nut copy JSON
            st.download_button(
                label     = "Tải kết quả JSON",
                data      = json.dumps(data, ensure_ascii=False, indent=2),
                file_name = "cccd_result.json",
                mime      = "application/json",
            )

        elif status == "no_card":
            st.markdown('<span class="badge-error">❌ Không tìm thấy thẻ CCCD</span>',
                        unsafe_allow_html=True)
            st.warning("Hãy chụp rõ hơn, đảm bảo thẻ nằm trong khung ảnh.")

        elif status == "corners_missing":
            found = result.get("found", [])
            st.markdown('<span class="badge-error">❌ Không xác định được góc thẻ</span>',
                        unsafe_allow_html=True)
            st.warning(f"Chỉ tìm được: {found}. Hãy chụp thẻ ở góc độ thẳng hơn.")

        elif status == "warp_failed":
            st.markdown('<span class="badge-error">❌ Không thể căn chỉnh ảnh</span>',
                        unsafe_allow_html=True)

        else:
            st.error(f"Lỗi không xác định: {status}")
            if "message" in result:
                st.code(result["message"])

# ─────────────────────────────────────────────────────────
# Trang 2: Training Results
# ─────────────────────────────────────────────────────────
def page_training():
    st.title("Training Results")
    st.caption("Xem kết quả và đồ thị quá trình huấn luyện")

    # Danh sach model
    MODELS = {
        "Card Detector":   "weights/card",
        "Corner Detector": "weights/corner",
        "Field Detector":  "weights/field",
    }

    for model_name, weights_dir in MODELS.items():
        wd = Path(weights_dir)
        st.subheader(f"🔹 {model_name}")

        col1, col2, col3 = st.columns(3)

        # Kiem tra file ton tai
        has_best    = (wd / "best.pth").exists()
        has_plot    = (wd / "results.png").exists()
        has_kfold   = (wd / "kfold_results.png").exists()
        has_summary = (wd / "kfold_summary.txt").exists()

        col1.metric("best.pth",        "✅" if has_best    else "❌ Chưa train")
        col2.metric("results.png",     "✅" if has_plot    else "❌ Chưa có")
        col3.metric("kfold_results",   "✅" if has_kfold   else "— Không dùng K-Fold")

        # Hien thi do thi
        if has_plot or has_kfold:
            tab_labels = []
            if has_plot:  tab_labels.append("Epoch Chart")
            if has_kfold: tab_labels.append("K-Fold Results")
            if has_summary: tab_labels.append("K-Fold Summary")

            tabs = st.tabs(tab_labels)
            t = 0

            if has_plot:
                with tabs[t]:
                    st.image(str(wd / "results.png"),
                             use_container_width=True)
                t += 1

            if has_kfold:
                with tabs[t]:
                    st.image(str(wd / "kfold_results.png"),
                             use_container_width=True)
                t += 1

            if has_summary:
                with tabs[t]:
                    txt = (wd / "kfold_summary.txt").read_text(encoding="utf-8")
                    st.code(txt, language=None)

        else:
            st.info(f"Chưa có kết quả. Chạy: `python train.py configs/"
                    f"{model_name.split()[0].lower()}.yaml`")

        # K-Fold: hien thi ket qua tung fold
        fold_dirs = sorted(wd.glob("fold_*"))
        if fold_dirs:
            with st.expander(f"Xem chi tiết {len(fold_dirs)} fold"):
                cols = st.columns(min(len(fold_dirs), 3))
                for i, fold_dir in enumerate(fold_dirs):
                    fold_plot = fold_dir / "results.png"
                    if fold_plot.exists():
                        with cols[i % 3]:
                            st.image(str(fold_plot),
                                     caption=fold_dir.name,
                                     use_container_width=True)

        st.divider()


# ─────────────────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.title("ID Extractor")
        st.caption("Hệ thống trích xuất thông tin CCCD tự động")
        st.divider()

        page = st.radio(
            "Chọn chức năng",
            options = ["Trích xuất OCR", "Kết quả Training"],
            label_visibility = "collapsed",
        )

    if page == "Trích xuất OCR":
        page_ocr()
    else:
        page_training()


if __name__ == "__main__":
    main()
