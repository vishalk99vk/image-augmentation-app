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

# --- Filter Functions with parameter support ---

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
    angle = angle if angle is not None else random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), angle

def random_crop(img, scale=None, top=None, left=None):
    h, w = img.shape[:2]
    scale = scale if scale is not None else random.uniform(0.7, 0.95)
    new_h, new_w = int(h * scale), int(w * scale)
    top = top if top is not None else random.randint(0, h - new_h)
    left = left if left is not None else random.randint(0, w - new_w)
    cropped = img[top:top + new_h, left:left + new_w]
    resized = cv2.resize(cropped, (w, h))
    return resized, scale, top, left

def color_jitter(img, brightness=None, hue_shift=None, sat_scale=None):
    img = img.astype(np.float32)
    brightness = brightness if brightness is not None else (random.uniform(0.8, 1.2) if random.random() < 0.5 else 1.0)
    img *= brightness
    img = np.clip(img, 0, 255)
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_scale = sat_scale if sat_scale is not None else random.uniform(0.7, 1.3)
    hue_shift = hue_shift if hue_shift is not None else random.uniform(-10, 10)
    hsv[..., 1] *= sat_scale
    hsv[..., 0] += hue_shift
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out, brightness, hue_shift, sat_scale

def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def perspective_warp(img, shift=None):
    h, w = img.shape[:2]
    shift = shift if shift is not None else random.randint(5, 20)
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_pts = np.float32([[shift, 0], [w - shift, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT), shift

def apply_augmentations_with_params(img, filters, params=None):
    # params is dict to store/reuse augmentation parameters
    if params is None:
        params = {}

    if '3D Rotation' in filters:
        if '3d_angle_x' not in params:
            params['3d_angle_x'] = random.uniform(-10, 10)
            params['3d_angle_y'] = random.uniform(-10, 10)
            params['3d_angle_z'] = random.uniform(-10, 10)
        img = rotate_image_3d(img, params['3d_angle_x'], params['3d_angle_y'], params['3d_angle_z'])

    if '2D Rotation' in filters:
        if '2d_angle' not in params:
            img, angle = rotate_2d(img)
            params['2d_angle'] = angle
        else:
            img, _ = rotate_2d(img, params['2d_angle'])

    if 'Random Crop' in filters:
        if not all(k in params for k in ['crop_scale', 'crop_top', 'crop_left']):
            img, scale, top, left = random_crop(img)
            params['crop_scale'], params['crop_top'], params['crop_left'] = scale, top, left
        else:
            img, _, _, _ = random_crop(img, params['crop_scale'], params['crop_top'], params['crop_left'])

    if 'Color Jitter' in filters:
        if not all(k in params for k in ['brightness', 'hue_shift', 'sat_scale']):
            img, brightness, hue_shift, sat_scale = color_jitter(img)
            params['brightness'], params['hue_shift'], params['sat_scale'] = brightness, hue_shift, sat_scale
        else:
            img, _, _, _ = color_jitter(img, params['brightness'], params['hue_shift'], params['sat_scale'])

    if 'Sharpen' in filters:
        img = sharpen_image(img)

    if 'Perspective Warp' in filters:
        if 'persp_shift' not in params:
            img, shift = perspective_warp(img)
            params['persp_shift'] = shift
        else:
            img, _ = perspective_warp(img, params['persp_shift'])

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
st.set_page_config(page_title="Image Augmentation Tool with Reference Filters", layout="centered")
st.title("🧪 Augmentation with Preview + Auto-Labeling + Reference Filter Copy")

st.markdown("**Step 1:** Upload optional reference images (will generate variants and extract filters).")
ref_files = st.file_uploader("Upload Reference Images or ZIP (optional)", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)

st.markdown("**Step 2:** Upload client images.")
client_files = st.file_uploader("Upload Client Images or ZIP", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)

num_variants_ref = st.slider("Number of Variants per Reference Image", 1, 20, 5)
num_variants_client = st.slider("Number of Variants per Client Image (ignored if reference images uploaded)", 1, 20, 5)

filters = st.multiselect(
    "Select Filters",
    ["3D Rotation", "2D Rotation", "Random Crop", "Color Jitter", "Sharpen", "Perspective Warp"],
    default=["3D Rotation", "2D Rotation"]
)

preview_toggle = st.checkbox("🔍 Show before/after preview")

if client_files and st.button("Run Augmentation"):
    with TemporaryDirectory() as tempdir:
        client_images = extract_images(client_files)
        labels = []

        # If reference images uploaded, generate variants and store params
        if ref_files:
            ref_images = extract_images(ref_files)
            if not ref_images:
                st.error("No valid reference images found.")
                st.stop()

            if not client_images:
                st.error("No valid client images found.")
                st.stop()

            st.info(f"Generating {num_variants_ref} variants per reference image to extract filters...")

            ref_variants = []  # (variant_name, params)
            for ref_filename, ref_img in ref_images:
                base_ref_name = os.path.splitext(os.path.basename(ref_filename))[0]
                for i in range(num_variants_ref):
                    params = {}
                    aug_ref, params = apply_augmentations_with_params(ref_img.copy(), filters, params)
                    variant_name = f"{base_ref_name}_aug_{i}.jpg"
                    ref_variants.append((variant_name, params))

            st.info(f"Applying {len(ref_variants)} reference variants to {len(client_images)} client images...")
            total_count = 0
            for client_filename, client_img in client_images:
                base_client_name = os.path.splitext(os.path.basename(client_filename))[0]
                for variant_name, params in ref_variants:
                    aug_client, _ = apply_augmentations_with_params(client_img.copy(), filters, params)
                    out_name = f"{base_client_name}_{variant_name}"
                    out_path = os.path.join(tempdir, out_name)
                    cv2.imwrite(out_path, aug_client)
                    labels.append(out_name)
                    total_count += 1

                    if preview_toggle and total_count <= 10:
                        st.write(f"**Preview:** {base_client_name} with {variant_name}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(cv2.cvtColor(client_img, cv2.COLOR_BGR2RGB), caption="Original Client Image", use_column_width=True)
                        with col2:
                            st.image(cv2.cvtColor(aug_client, cv2.COLOR_BGR2RGB), caption="Augmented Client Image", use_column_width=True)

            st.success(f"Generated {total_count} augmented client images using reference filters.")

        else:
            # No reference images, just generate variants per client image with random params
            total_count = 0
            for client_filename, client_img in client_images:
                base_client_name = os.path.splitext(os.path.basename(client_filename))[0]
                for i in range(num_variants_client):
                    aug_img, _ = apply_augmentations_with_params(client_img.copy(), filters)
                    out_name = f"{base_client_name}_aug_{i}.jpg"
                    out_path = os.path.join(tempdir, out_name)
                    cv2.imwrite(out_path, aug_img)
                    labels.append(out_name)
                    total_count += 1

                    if preview_toggle and total_count <= 10:
                        st.write(f"**Preview:** {out_name}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(cv2.cvtColor(client_img, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
                        with col2:
                            st.image(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB), caption="Augmented", use_column_width=True)

            st.success(f"Generated {total_count} augmented client images.")

        # ZIP download of results
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for file in os.listdir(tempdir):
                zipf.write(os.path.join(tempdir, file), arcname=file)
        zip_buffer.seek(0)
        st.download_button("⬇️ Download Augmented Images ZIP", data=zip_buffer, file_name="augmented_images.zip")

else:
    st.info("Upload client images and optionally reference images, then click 'Run Augmentation'.")

