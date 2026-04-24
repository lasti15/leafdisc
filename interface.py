import streamlit as st
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from analysis import analyse_image, new_analyse_image
import requests
import base64

st.set_page_config(layout="wide")

if "all_results" not in st.session_state:
    st.session_state.all_results = {}

if "radius_average" not in st.session_state:
    st.session_state.radius_average = []

if "roundness_limit" not in st.session_state:
    st.session_state.roundness_limit = 0.3

if "acceptable_radius_variation" not in st.session_state:
    st.session_state.acceptable_radius_variation = 0.2

if "n_disks" not in st.session_state:
    st.session_state.n_disks = 30

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
st.sidebar.markdown(
    f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            position: relative;
        }}
        .top-left-logo {{
            position: absolute;
            top: 1rem;
            right: 1rem;
        }}
        .top-left-logo img {{
            width: 240px;
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

    uploaded_files.sort(key=lambda f: f.name.lower())

    if st.sidebar.button("Calculate all images"):
        progress = st.progress(0, text="Analysing images...")
        for idx, current_file in enumerate(uploaded_files):
            current_file.seek(0)  # reset file pointer before reading
            pil_img = Image.open(current_file).convert("RGB")
            img_np = np.array(pil_img)
            max_radius = round(int(min(pil_img.size) / 5), -2)
            if st.session_state.radius_average == []:
                start_limit = min(230, max_radius)
            else:
                start_limit = int(np.median(st.session_state.radius_average))

            results, img, st.session_state.radius_average = new_analyse_image(img_np, current_file.name, start_limit,
                                                                              st.session_state.roundness_limit,
                                                                              st.session_state.acceptable_radius_variation,
                                                                              st.session_state.radius_average,
                                                                              st.session_state.n_disks)
            st.session_state.all_results[current_file.name] = results

            progress.progress(
                (idx + 1) / len(uploaded_files),
                text=f"Analysing images... {idx + 1}/{len(uploaded_files)}"
            )
        progress.empty()

    else:

        if st.sidebar.button("Next"):
            st.session_state.current_index = (st.session_state.current_index + 1) % len(uploaded_files)

        options = [f.name for f in uploaded_files]
        selected_name = st.sidebar.selectbox(
            "Select image",
            options,
            index=st.session_state.current_index
        )

        st.session_state.current_index = options.index(selected_name)

        # Main panel

        current_file = uploaded_files[st.session_state.current_index]
        current_file.seek(0)  # reset file pointer before reading
        pil_img = Image.open(current_file).convert("RGB")
        st.write(f"### {current_file.name} ({st.session_state.current_index + 1}/{len(uploaded_files)})")

        image_container = st.container()  # will appear first (top)
        slider_container = st.container()

        # Roundness slider
        with slider_container:
            st.session_state.roundness_limit = st.slider("Roundness", 0.0, 1.0, st.session_state.roundness_limit, 0.05)
            st.session_state.n_disks = st.number_input("Expected n leaf disks", min_value=1, max_value=None,
                                                       value=st.session_state.n_disks)
            st.session_state.acceptable_radius_variation = st.slider("Accepted radius variation (% less than median)",
                                                                     0.0, 1.0,
                                                                     st.session_state.acceptable_radius_variation, 0.05)

        # Convert to NumPy array for OpenCV blur
        img_np = np.array(pil_img)

        # Size slider
        max_radius = round(int(min(pil_img.size) / 5), -2)

        if st.session_state.radius_average == []:
            start_limit = np.min([230, max_radius])
        else:
            start_limit = int(np.median(st.session_state.radius_average))

        with image_container:
            disk_limit = st.slider("Select approx starting size", 0, max_radius, start_limit, 1)

        # results, img, st.session_state.radius_average = analyse_image(img_np, roundness_limit, acceptable_radius_variation, st.session_state.radius_average, n_disks)
        results, img, st.session_state.radius_average = new_analyse_image(img_np, current_file.name, disk_limit,
                                                                          st.session_state.roundness_limit,
                                                                          st.session_state.acceptable_radius_variation,
                                                                          st.session_state.radius_average,
                                                                          st.session_state.n_disks)
        st.session_state.all_results[current_file.name] = results

        # Convert back to PIL for Streamlit display
        img_pil = Image.fromarray(img)
        with image_container:
            if st.button("Retry"):
                st.rerun()
            st.image(img_pil, width='stretch')

    if st.session_state.all_results:
        import pandas as pd
        import io

        import itertools

        all_rows = list(itertools.chain.from_iterable(st.session_state.all_results.values()))
        df = pd.DataFrame(all_rows)
        # df = pd.DataFrame(st.session_state.all_results.values())
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.sidebar.download_button(
            label=f"Download All Results ({len(df)} disks, {df['img_name'].nunique()} images)",
            data=csv_buffer.getvalue(),
            file_name="results.csv",
            mime="text/csv"
        )



else:
    st.write("Drag & drop images into the sidebar to begin.")
