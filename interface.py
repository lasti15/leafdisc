import streamlit as st
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from analysis import analyse_image
import requests
import base64

st.set_page_config(layout="wide")

if "radius_average" not in st.session_state:
    st.session_state.radius_average = []
avg_size = []

# ---- Add logo ----
logo_url = "https://www.rothamsted.ac.uk/sites/default/files/rothamsted-logo.png"  # or a URL like "https://example.com/logo.png"
response = requests.get(logo_url)
logo = Image.open(BytesIO(response.content)).convert("RGBA")  # keep transparency

# ---- Convert black pixels to white ----
data = np.array(logo)  # shape: (H, W, 4)
r, g, b, a = data.T

# Define black pixels (all channels near 0)
black_areas = (r < 50) & (g < 50) & (b < 50)

# Set black pixels to white
data[..., :3][black_areas.T] = [255, 255, 255]

# Convert back to PIL
logo_white = Image.fromarray(data)

# Encode image as base64
buffer = BytesIO()
logo_white.save(buffer, format="PNG")
logo_base64 = base64.b64encode(buffer.getvalue()).decode()

# ---- Display logo in top-right corner ----
st.markdown(
    f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            position: relative;
        }}
        .top-right-logo {{
            position: absolute;
            top: 1rem;
            right: 1rem;
        }}
        .top-right-logo img {{
            width: 120px;
        }}
    </style>
    <div class="top-right-logo">
        <img src="data:image/png;base64,{logo_base64}">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Leaf disk damage estimator")

# ---- File uploader (drag & drop) ----
uploaded_files = st.sidebar.file_uploader(
    "Drag and drop images here",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True
)

# ---- Session state to track current image ----
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if uploaded_files:
    # Keep images sorted by filename
    uploaded_files.sort(key=lambda f: f.name.lower())

    # Sidebar
    st.sidebar.header("Images")

    if uploaded_files:
        uploaded_files.sort(key=lambda f: f.name.lower())

        options = [f.name for f in uploaded_files]
        selected_name = st.sidebar.selectbox("Select image", options)

        st.session_state.current_index = options.index(selected_name)


    # Main panel
    current_file = uploaded_files[st.session_state.current_index]
    current_file.seek(0)  # reset file pointer before reading
    pil_img = Image.open(current_file).convert("RGB")

    st.write(f"### {current_file.name} ({st.session_state.current_index + 1}/{len(uploaded_files)})")

    # Roundness slider
    roundness_limit = st.slider("Roundness", 0.0, 1.0, 0.3, 0.05)
    n_disks = st.number_input("Expected n leaf disks", min_value=1, max_value=None, value="min")
    acceptable_radius_variation = st.slider("Accepted radius variation (% less than median)", 0.0, 1.0, 0.2,0.05)

    # Convert to NumPy array for OpenCV blur
    img_np = np.array(pil_img)

    results, img, st.session_state.radius_average = analyse_image(img_np, roundness_limit, acceptable_radius_variation, st.session_state.radius_average, n_disks)

    # Size slider
    max_radius = round(int(min(pil_img.size)/5),-2)
    disk_limit = st.slider("Disk Size", 0, max_radius, int(np.median(st.session_state.radius_average)), 1)

    print(st.session_state.radius_average)

    # Convert back to PIL for Streamlit display
    img_pil = Image.fromarray(img)
    st.image(img_pil, width='stretch')
else:
    st.write("Drag & drop images into the sidebar to begin.")
