import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
import io

def pil_to_cv2(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def get_corner_stats(image, patch_size=50):
    w, h = image.size
    corners = {
        "top_left": image.crop((0, 0, patch_size, patch_size)),
        "top_right": image.crop((w - patch_size, 0, w, patch_size)),
        "bottom_left": image.crop((0, h - patch_size, patch_size, h)),
        "bottom_right": image.crop((w - patch_size, h - patch_size, w, h)),
    }
    avg_colors = {k: np.array(v).mean(axis=(0, 1)) for k, v in corners.items()}
    return avg_colors

def match_brightness_contrast_hue(client_img, ref_img):
    # Convert to HSV
    ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    client_hsv = cv2.cvtColor(client_img, cv2.COLOR_BGR2HSV).astype(np.float32)

    ref_mean = ref_hsv.mean(axis=(0, 1))
    client_mean = client_hsv.mean(axis=(0, 1))

    gain = ref_mean / client_mean
    adjusted = client_hsv * gain
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

    return cv2.cvtColor(adjusted, cv2.COLOR_HSV2BGR)

def apply_random_tilt(image):
    angle = random.uniform(-15, 15)
    return image.rotate(angle, resample=Image.BICUBIC, expand=True)

def apply_3d_rotation_sim(image):
    angle = random.uniform(-15, 15)
    w, h = image.size
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    pts2 = np.float32([
        [random.uniform(-15,15), random.uniform(-15,15)],
        [w - random.uniform(-15,15), random.uniform(-15,15)],
        [random.uniform(-15,15), h - random.uniform(-15,15)],
        [w - random.uniform(-15,15), h - random.uniform(-15,15)],
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_cv = pil_to_cv2(image)
    warped = cv2.warpPerspective(img_cv, matrix, (w, h))
    return cv2_to_pil(warped)

st.title("📸 Reference Style Transfer Tool for Client Images")

# Upload images
ref_image_file = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"])
client_image_files = st.file_uploader("Upload Client Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if ref_image_file and client_image_files:
    ref_img = Image.open(ref_image_file).convert("RGB")
    ref_cv = pil_to_cv2(ref_img)

    st.subheader("Reference Image Preview")
    st.image(ref_img, caption="Reference", use_column_width=True)

    # Extract reference color features
    ref_corners = get_corner_stats(ref_img)

    st.subheader("Processed Client Images")
    for uploaded_file in client_image_files:
        client_img = Image.open(uploaded_file).convert("RGB")
        client_cv = pil_to_cv2(client_img)

        # Adjust colors
        adjusted_cv = match_brightness_contrast_hue(client_cv, ref_cv)
        adjusted_img = cv2_to_pil(adjusted_cv)

        # Optional random tilt
        if st.checkbox(f"Apply Tilt to: {uploaded_file.name}", value=True):
            adjusted_img = apply_random_tilt(adjusted_img)

        # Optional 3D perspective
        if st.checkbox(f"Apply 3D Rotation to: {uploaded_file.name}", value=True):
            adjusted_img = apply_3d_rotation_sim(adjusted_img)

        st.image(adjusted_img, caption=f"Processed - {uploaded_file.name}", use_column_width=True)

        # Download button
        buf = io.BytesIO()
        adjusted_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label=f"Download {uploaded_file.name}",
            data=byte_im,
            file_name=f"processed_{uploaded_file.name}",
            mime="image/png"
        )
