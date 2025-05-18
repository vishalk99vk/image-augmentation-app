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

def rotate_2d(img, angle=None):
    if angle is None:
        angle = random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def random_crop(img, scale=None, top=None, left=None):
    h, w = img.shape[:2]
    if scale is None:
        scale = random.uniform(0.7, 0.95)
    new_h, new_w = int(h * scale), int(w * scale)
    if top is None:
        top = random.randint(0, h - new_h)
    if left is None:
        left = random.randint(0, w - new_w)
    return cv2.resize(img[top:top + new_h, left:left + new_w], (w, h))

def color_jitter(img, brightness_factor=None, hue_shift=None, saturation_factor=None):
    img = img.astype(np.float32)
    if brightness_factor is None:
        brightness_factor = random.uniform(0.8, 1.2)
    if random.random() < 0.5:
        img *= brightness_factor
    img = np.clip(img, 0, 255)
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    if saturation_factor is None:
        saturation_factor = random.uniform(0.7, 1.3)
    hsv[..., 1] *= saturation_factor
    if hue_shift is None:
        hue_shift = random.uniform(-10, 10)
    hsv[..., 0] += hue_shift
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def perspective_warp(img, shift=None):
    h, w = img.shape[:2]
    if shift is None:
        shift = random.randint(5, 20)
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_pts = np.float32([[shift, 0], [w - shift, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

# Apply filters with optional parameters (for copying)
def apply_augmentations_with_params(img, filters, params=None):
    # params is dict of filter_name: parameters used in that filter to reproduce augmentation
    if params is None:
        params = {}

    if '3D Rotation' in filters:
        p = params.get('3D Rotation')
        if p is None:
            angles = (random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10))
        else:
            angles = p
        img = rotate_image_3d(img, *angles)
        params['3D Rotation'] = angles

    if '2D Rotation' in filters:
        p = params.get('2D Rotation')
        if p is None:
            angle = random.uniform(-10, 10)
        else:
            angle = p
        img = rotate_2d(img, angle)
        params['2D Rotation'] = angle

    if 'Random Crop' in filters:
        p = params.get('Random Crop')
        if p is None:
            scale = random.uniform(0.7, 0.95)
            top = None
            left = None
        else:
            scale, top, left = p
        img = random_crop(img, scale, top, left)
        params['Random Crop'] = (scale, top, left)

    if 'Color Jitter' in filters:
        p = params.get('Color Jitter')
        if p is None:
            brightness_factor = random.uniform(0.8, 1.2)
            hue_shift = random.uniform(-10, 10)
            saturation_factor = random.uniform(0.7, 1.3)
        else:
            brightness_factor, hue_shift, saturation_factor = p
        img = color_jitter(img, brightness_factor, hue_shift, saturation_factor)
        params['Color Jitter'] = (brightness_factor, hue_shift, saturation_factor)

    if 'Sharpen' in filters:
        # Sharpen has no params
        img = sharpen_image(img)

    if 'Perspective Warp' in filters:
        p = params.get('Perspective Warp')
        if p is None:
            shift = random.randint(5, 20)
        else:
            shift = p
        img = perspective_warp(img, shift)
        params['Perspective Warp'] = shift

    return img, params

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
st.title("🧪 Augmentation with Preview + Auto-Labeling + Reference Filters")

uploaded_ref_files = st.file_uploader("Upload reference images (optional)", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)
uploaded_client_files = st.file_uploader("Upload client images", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)

num_variants_ref = st.slider("Number of Variants per Reference Image", 1, 10, 3)
filters = st.multiselect("Select Filters", ["3D Rotation", "2D Rotation", "Random Crop", "Color Jitter", "Sharpen", "Perspective Warp"], default=["3D Rotation", "2D Rotation"])
preview_toggle = st.checkbox("🔍 Show before/after preview")

if uploaded_client_files and st.button("Run Augmentation"):
    with TemporaryDirectory() as tempdir:
        client_images = extract_images(uploaded_client_files)
        ref_images = extract_images(uploaded_ref_files) if uploaded_ref_files else []

        labels = []

        if not client_images:
            st.error("No valid client images found.")
        else:
            # If no ref images, just apply filters directly on client images
            if not ref_images:
                st.info("No reference images uploaded, applying filters directly on client images.")

                for filename, img in client_images:
                    base_name = os.path.splitext(os.path.basename(filename))[0]
                    for i in range(num_variants_ref):
                        aug, _ = apply_augmentations_with_params(img.copy(), filters, params=None)
                        new_name = f"{base_name}_aug_{i}.jpg"
                        out_path = os.path.join(tempdir, new_name)
                        cv2.imwrite(out_path, aug)
                        label = base_name.split("_")[0]
                        labels.append((new_name, label))

                        if preview_toggle and i == 0:
                            st.markdown(f"**Original vs Augmented → `{filename}`**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
                            with col2:
                                st.image(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB), caption="Augmented", use_column_width=True)

            else:
                # For each ref image, create variants + store their params (filters + params)
                ref_variants = []  # List of (ref_name, variant_img, params)
                for ref_filename, ref_img in ref_images:
                    base_ref_name = os.path.splitext(os.path.basename(ref_filename))[0]
                    for i in range(num_variants_ref):
                        params = {}
                        aug_ref, params = apply_augmentations_with_params(ref_img.copy(), filters, params)
                        variant_name = f"{base_ref_name}_aug_{i}.jpg"
                        ref_variants.append((variant_name, params))

                total_count = 0
                # Now apply each ref variant's filters & params on each client image
                for client_filename, client_img in client_images:
                    base_client_name = os.path.splitext(os.path.basename(client_filename))[0]
                    for variant_name, params in ref_variants:
                        aug_client, _ = apply_augmentations_with_params(client_img.copy(), filters, params)
                        new_name = f"{base_client_name}_ref_{variant_name}"
                        out_path = os.path.join(tempdir, new_name)
                        cv2.imwrite(out_path, aug_client)
                        label = base_client_name.split("_")[0]
                        labels.append((new_name, label))
                        total_count += 1

                        # Show preview for first client image and first variant only
                        if preview_toggle and total_count == 1:
                            st.markdown(f"**Original Client Image vs Augmented Client Image**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(cv2.cvtColor(client_img, cv2.COLOR_BGR2RGB), caption=f"Original {client_filename}", use_column_width=True)
                            with col2:
                                st.image(cv2.cvtColor(aug_client, cv2.COLOR_BGR2RGB), caption=f"Augmented {new_name}", use_column_width=True)

                st.success(f"✅ Generated {total_count} augmented images from {len(client_images)} client images and {len(ref_images)} reference image variants.")

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

            st.download_button("📦 Download Augmented Images + Labels", data=zip_buffer, file_name="augmented_dataset.zip", mime="application/zip")
else:
    st.info("📂 Upload client images and optionally reference images, select filters to begin.")
