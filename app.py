import streamlit as st
import os
import cv2
import numpy as np
import random
import math
import zipfile
import pandas as pd
from io import BytesIO
from tempfile import TemporaryDirectory

# --- Filter Functions ---

def rotate_image_3d(img, angle_x=0, angle_y=0, angle_z=0):
    h, w = img.shape[:2]
    ax = math.radians(angle_x)
    ay = math.radians(angle_y)
    az = math.radians(angle_z)
    f = w
    cx, cy = w / 2, h / 2
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    corners = np.array([[-cx, -cy, 0], [w - cx, -cy, 0], [-cx, h - cy, 0], [w - cx, h - cy, 0]])
    rotated = corners @ R.T
    projected = rotated.copy()
    projected[:, 0] = (f * rotated[:, 0]) / (f + rotated[:, 2]) + cx
    projected[:, 1] = (f * rotated[:, 1]) / (f + rotated[:, 2]) + cy
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_pts = np.float32(projected[:, :2])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

def rotate_2d(img):
    angle = random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def random_crop(img):
    h, w = img.shape[:2]
    scale = random.uniform(0.7, 0.95)
    new_h, new_w = int(h * scale), int(w * scale)
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    return cv2.resize(img[top:top + new_h, left:left + new_w], (w, h))

def color_jitter(img):
    img = img.astype(np.float32)
    if random.random() < 0.5:
        img *= random.uniform(0.8, 1.2)
    img = np.clip(img, 0, 255)
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= random.uniform(0.7, 1.3)
    hsv[..., 0] += random.uniform(-10, 10)
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def perspective_warp(img):
    h, w = img.shape[:2]
    shift = random.randint(5, 20)
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_pts = np.float32([[shift, 0], [w - shift, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_augmentations(img, filters):
    if '3D Rotation' in filters:
        img = rotate_image_3d(img, random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10))
    if '2D Rotation' in filters:
        img = rotate_2d(img)
    if 'Random Crop' in filters:
        img = random_crop(img)
    if 'Color Jitter' in filters:
        img = color_jitter(img)
    if 'Sharpen' in filters:
        img = sharpen_image(img)
    if 'Perspective Warp' in filters:
        img = perspective_warp(img)
    return img

def extract_images(files):
    images = []
    for uploaded_file in files:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as z:
                for file in z.namelist():
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        data = z.read(file)
                        img_np = np.frombuffer(data, np.uint8)
                        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                        if img is not None:
                            images.append((file, img))
        else:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                images.append((uploaded_file.name, img))
    return images

# --- Streamlit App ---
st.set_page_config(page_title="Image Augmentation Tool", layout="centered")
st.title("🧪 Augmentation with Preview + Auto-Labeling")

uploaded_files = st.file_uploader("Upload images or ZIP", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)
num_variants = st.slider("Number of Variants per Image", 1, 20, 5)

filters = st.multiselect("Select Filters", ["3D Rotation", "2D Rotation", "Random Crop", "Color Jitter", "Sharpen", "Perspective Warp"], default=["3D Rotation", "2D Rotation"])
preview_toggle = st.checkbox("🔍 Show before/after preview")

if uploaded_files and st.button("Run Augmentation"):
    with TemporaryDirectory() as tempdir:
        images = extract_images(uploaded_files)
        labels = []

        if not images:
            st.error("No valid images found.")
        else:
            for filename, img in images:
                label = os.path.splitext(os.path.basename(filename))[0].split("_")[0]
                base_name = os.path.splitext(os.path.basename(filename))[0]

                for i in range(num_variants):
                    aug = apply_augmentations(img.copy(), filters)
                    new_name = f"{base_name}_aug_{i}.jpg"
                    out_path = os.path.join(tempdir, new_name)
                    cv2.imwrite(out_path, aug)
                    labels.append((new_name, label))

                    # Show preview (first image and first variant only)
                    if preview_toggle and i == 0:
                        st.markdown(f"**Original vs Augmented → `{filename}`**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
                        with col2:
                            st.image(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB), caption="Augmented", use_column_width=True)

            # Save labels.csv
            df = pd.DataFrame(labels, columns=["filename", "label"])
            label_path = os.path.join(tempdir, "labels.csv")
            df.to_csv(label_path, index=False)

            # Create ZIP
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zipf:
                for file in os.listdir(tempdir):
                    zipf.write(os.path.join(tempdir, file), arcname=file)
            zip_buffer.seek(0)

            st.success(f"✅ Generated {num_variants} variant(s) per image.")
            st.download_button("📦 Download Augmented Images + Labels", data=zip_buffer, file_name="augmented_dataset.zip", mime="application/zip")
else:
    st.info("📂 Upload images and select filters to begin.")
