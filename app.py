import streamlit as st
import cv2
import numpy as np
import os
import zipfile
from io import BytesIO

# === Tint variations (BGR format for OpenCV)
tints = {
    "warm": (0, 30, 80),
    "cool": (80, 30, 0),
    'daylight': (255, 255, 240),
    'cool_white': (220, 255, 255),
    'warm_white': (255, 240, 200),
}

# === Brightness levels
brightness_factors = {
    "dark": 0.8,
    "normal": 1.2,
    "bright": 1.4
}

alpha = 0.25  # Tint blending strength

# === 3D tilt transformation
def apply_3d_tilt(image, direction="left", tilt_factor=0.1):
    h, w = image.shape[:2]
    dx = int(w * tilt_factor)
    dy = int(h * tilt_factor)

    if direction == "left":
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst = np.float32([[dx, 0], [w, 0], [0, h], [w - dx, h]])
    elif direction == "right":
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst = np.float32([[0, 0], [w - dx, 0], [dx, h], [w, h]])
    elif direction == "up":
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst = np.float32([[0, dy], [w, dy], [0, h], [w, h]])
    elif direction == "down":
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dst = np.float32([[0, 0], [w, 0], [0, h - dy], [w, h - dy]])
    else:
        return image

    M = cv2.getPerspectiveTransform(src, dst)
    tilted = cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return tilted

# === Streamlit UI
st.title("Image Brightness, Tint & 3D Tilt Generator")

uploaded_files = st.file_uploader("Upload image(s)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:

        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.error(f"Failed to load image: {uploaded_file.name}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            base_name = os.path.splitext(uploaded_file.name)[0]

            for brightness_label, brightness in brightness_factors.items():
                bright_image = np.clip(image * brightness, 0, 255).astype(np.uint8)

                for tint_label, tint_color in tints.items():
                    tint_layer = np.full_like(bright_image, tint_color, dtype=np.uint8)
                    tinted_image = cv2.addWeighted(bright_image, 1 - alpha, tint_layer, alpha, 0)

                    # Save original tint + brightness
                    variant_name = f"{base_name}_{brightness_label}_{tint_label}"
                    final_bgr = cv2.cvtColor(tinted_image, cv2.COLOR_RGB2BGR)
                    success, buffer = cv2.imencode(".jpg", final_bgr)
                    if success:
                        zip_file.writestr(f"{variant_name}.jpg", buffer.tobytes())

                    # === Apply 3D tilt variants
                    for direction in ["left", "right", "up", "down"]:
                        tilted_image = apply_3d_tilt(tinted_image, direction=direction, tilt_factor=0.1)
                        final_bgr = cv2.cvtColor(tilted_image, cv2.COLOR_RGB2BGR)
                        success, buffer = cv2.imencode(".jpg", final_bgr)
                        if success:
                            zip_file.writestr(f"{variant_name}_tilt_{direction}.jpg", buffer.tobytes())

    zip_buffer.seek(0)
    st.download_button(
        label="Download All Processed Images as ZIP",
        data=zip_buffer,
        file_name="processed_images.zip",
        mime="application/zip"
    )
