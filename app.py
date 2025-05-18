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

# --- Your existing filter functions here (unchanged) ---

def rotate_image_3d_with_params(img, angle_x, angle_y, angle_z):
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

def rotate_2d_with_params(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def random_crop_with_params(img, top, left, new_h, new_w):
    h, w = img.shape[:2]
    cropped = img[top:top + new_h, left:left + new_w]
    return cv2.resize(cropped, (w, h))

def color_jitter_with_params(img, brightness_factor, hue_shift, saturation_factor):
    img = img.astype(np.float32)
    img *= brightness_factor
    img = np.clip(img, 0, 255)
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= saturation_factor
    hsv[..., 0] += hue_shift
    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def perspective_warp_with_params(img, shift):
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_pts = np.float32([[shift, 0], [w - shift, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

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

# --- Generate random filter parameters for each filter ---
def generate_random_params_for_filters(filters, img_shape):
    h, w = img_shape[:2]
    params = {}
    if "3D Rotation" in filters:
        params['3D Rotation'] = {
            'angle_x': random.uniform(-10, 10),
            'angle_y': random.uniform(-10, 10),
            'angle_z': random.uniform(-10, 10),
        }
    if "2D Rotation" in filters:
        params['2D Rotation'] = {
            'angle': random.uniform(-10, 10)
        }
    if "Random Crop" in filters:
        scale = random.uniform(0.7, 0.95)
        new_h, new_w = int(h * scale), int(w * scale)
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        params['Random Crop'] = {
            'top': top,
            'left': left,
            'new_h': new_h,
            'new_w': new_w
        }
    if "Color Jitter" in filters:
        params['Color Jitter'] = {
            'brightness_factor': random.uniform(0.8, 1.2),
            'hue_shift': random.uniform(-10, 10),
            'saturation_factor': random.uniform(0.7, 1.3)
        }
    if "Sharpen" in filters:
        # No params for sharpen
        params['Sharpen'] = {}
    if "Perspective Warp" in filters:
        params['Perspective Warp'] = {
            'shift': random.randint(5, 20)
        }
    return params

# --- Apply augmentations with fixed parameters ---
def apply_augmentations_with_params(img, filter_params):
    for filter_name, params in filter_params.items():
        if filter_name == "3D Rotation":
            img = rotate_image_3d_with_params(img, params['angle_x'], params['angle_y'], params['angle_z'])
        elif filter_name == "2D Rotation":
            img = rotate_2d_with_params(img, params['angle'])
        elif filter_name == "Random Crop":
            img = random_crop_with_params(img, params['top'], params['left'], params['new_h'], params['new_w'])
        elif filter_name == "Color Jitter":
            img = color_jitter_with_params(img, params['brightness_factor'], params['hue_shift'], params['saturation_factor'])
        elif filter_name == "Sharpen":
            img = sharpen_image(img)
        elif filter_name == "Perspective Warp":
            img = perspective_warp_with_params(img, params['shift'])
    return img


# --- Streamlit App ---

st.set_page_config(page_title="Image Augmentation with Reference Filters", layout="centered")
st.title("🧪 Augmentation with Reference Filters")

st.markdown("### Step 1: Upload Reference Images (Optional)")
reference_files = st.file_uploader("Upload Reference images or ZIP", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)

st.markdown("### Step 2: Upload Client Images")
client_files = st.file_uploader("Upload Client images or ZIP", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)

filters = st.multiselect("Select Filters (applied to reference images)", ["3D Rotation", "2D Rotation", "Random Crop", "Color Jitter", "Sharpen", "Perspective Warp"], default=["3D Rotation", "2D Rotation"])

num_variants = st.slider("Number of Variants per Reference Image (if no reference images uploaded, applies random params)", 1, 5, 1)
preview_toggle = st.checkbox("🔍 Show before/after preview")

if client_files and st.button("Run Augmentation"):
    with TemporaryDirectory() as tempdir:
        client_images = extract_images(client_files)
        if not client_images:
            st.error("No valid client images found.")
            st.stop()

        # Prepare reference filter params list
        reference_filter_params_list = []

        if reference_files:
            ref_images = extract_images(reference_files)
            if not ref_images:
                st.warning("No valid reference images found. Using random params.")
            else:
                # For each reference image generate random filter params once
                for ref_name, ref_img in ref_images:
                    params = generate_random_params_for_filters(filters, ref_img.shape)
                    reference_filter_params_list.append((ref_name, params))
        else:
            # No reference images, generate N variants of random params
            for i in range(num_variants):
                dummy_shape = client_images[0][1].shape
                params = generate_random_params_for_filters(filters, dummy_shape)
                reference_filter_params_list.append((f"variant_{i}", params))

        labels = []
        count = 0

        for ref_name, filter_params in reference_filter_params_list:
            for client_name, client_img in client_images:
                aug_img = apply_augmentations_with_params(client_img.copy(), filter_params)
                base_name = os.path.splitext(client_name)[0]
                out_name = f"{base_name}_aug_from_{os.path.splitext(ref_name)[0]}.jpg"
                out_path = os.path.join(tempdir, out_name)
                cv2.imwrite(out_path, aug_img)
                label = base_name.split("_")[0]
                labels.append((out_name, label))
                count += 1

                if preview_toggle and count == 1:
                    st.markdown(f"**Preview of Augmentation from Reference '{ref_name}' applied on Client '{client_name}'**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cv2.cvtColor(client_img, cv2.COLOR_BGR2RGB), caption="Original Client Image", use_column_width=True)
                    with col2:
                        st.image(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB), caption=f"Augmented Client Image\n(Ref: {ref_name})", use_column_width=True)

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

        st.success(f"✅ Generated {count} augmented images (client images x reference filter sets).")
        st.download_button("📦 Download Augmented Images + Labels", data=zip_buffer, file_name="augmented_dataset.zip", mime="application/zip")
else:
    st.info("📂 Upload client images and optionally reference images, then select filters to begin.")
