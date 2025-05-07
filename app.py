import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
import gdown

# -------------------- Streamlit Config & Styling --------------------
st.set_page_config(page_title="Chessboard to FEN", layout="wide")

# Apply modern glass-style UI
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .stApp {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 1rem;
        padding: 2rem;
        backdrop-filter: blur(10px);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f0f0f0 !important;
    }
    button {
        border-radius: 8px;
        background: #1e3c72;
        background: linear-gradient(to right, #2a5298, #1e3c72);
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease-in-out;
    }
    button:hover {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        transform: scale(1.05);
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

st.title("♟️ Chessboard Image to FEN Generator")

# -------------------- Model Loading --------------------
@st.cache_resource
def load_model_from_drive():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1bWy_hMxsENMd9xgMP9buoDWjCLN3gvEt"
        gdown.download(url, model_path, quiet=False)
    return YOLO(model_path)

model = load_model_from_drive()

client = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="hiXB6MTQOH0hlxROddmK"
)

TEMP_DIR = tempfile.mkdtemp()
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# -------------------- Helper Functions --------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]    # Top-left
    rect[2] = pts[np.argmax(s)]    # Bottom-right
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left
    return rect

def generate_fen(board):
    fen_rows = []
    for row in board:
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell == "Blank":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                piece_map = {
                    'Black_Pawn': 'p', 'Black_Rook': 'r', 'Black_Knight': 'n',
                    'Black_Bishop': 'b', 'Black_Queen': 'q', 'Black_King': 'k',
                    'White_Pawn': 'P', 'White_Rook': 'R', 'White_Knight': 'N',
                    'White_Bishop': 'B', 'White_Queen': 'Q', 'White_King': 'K'
                }
                fen_row += piece_map.get(cell, '?')
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    return "/".join(fen_rows) + " w - - 0 1"

# -------------------- Main App --------------------
uploaded_file = st.file_uploader("📷 Upload a chessboard image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)[:, :, ::-1]  # RGB to BGR
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if 'fen' not in st.session_state:
        with st.spinner("🔍 Detecting chessboard..."):
            with tempfile.NamedTemporaryFile(suffix=".jpg", dir=TEMP_DIR, delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, img_np)
                result = client.infer(tmp_path, model_id="chessboard-segmentation/1")

            points = np.array([(p["x"], p["y"]) for p in result["predictions"][0]["points"]], dtype=np.float32)
            hull = cv2.convexHull(points).astype(np.int32)
            ordered_corners = order_points(hull.reshape(-1, 2)).astype(np.float32)

            dst_points = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
            warped = cv2.warpPerspective(img_np, matrix, (1000, 1000))

            board_layout = []
            rows, cols = 8, 8
            cell_size = 1000 // rows
            for i in range(rows):
                row_layout = []
                for j in range(cols):
                    x_start, y_start = j * cell_size, i * cell_size
                    cell = warped[y_start:y_start + cell_size, x_start:x_start + cell_size]
                    result = model.predict(source=cell, verbose=False)
                    class_idx = result[0].probs.top1
                    class_name = model.names[class_idx]
                    row_layout.append(class_name)
                board_layout.append(row_layout)

            fen = generate_fen(board_layout)
            st.session_state.fen = fen

    # -------------------- Output FEN --------------------
    st.subheader("Generated FEN:")
    st.code(st.session_state.fen, language="text")
    st.success("✅ FEN copied! (Use the built-in copy icon in the top-right of the box)", icon="✅")

    # -------------------- Lichess Button --------------------
    if st.button("🔗 View in Lichess", type="primary"):
        lichess_url = f"https://lichess.org/analysis/{st.session_state.fen}"
        st.markdown(
            f"""<script>window.open("{lichess_url}", "_blank");</script>""",
            unsafe_allow_html=True,
        )
