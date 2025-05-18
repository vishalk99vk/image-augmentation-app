import os
import cv2
import numpy as np
import random
import math
import streamlit as st
from PIL import Image
from io import BytesIO

# Your augmentation functions unchanged
def rotate_image_3d(img, angle_x=0, angle_y=0, angle_z=0):
    h, w = img.shape[:2]
    ax = math.radians(angle_x)
    ay = math.radians(angle_y)
    az = math.radians(angle_z)

    f = w
    cx, cy = w / 2, h / 2

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(ax), -math.sin(ax)],
                   [0, math.sin(ax), math.cos(ax)]])

    Ry = np.array([[math.cos(ay), 0, math.sin(ay)],
                   [0, 1, 0],
                   [-math.sin(ay), 0, math.cos(ay)]])

    Rz = np.array([[math.cos(az), -math.sin(az), 0],
                   [math.sin(az), math.cos(az), 0],
                   [0, 0, 1]])

    R = Rz @ Ry @ Rx

    corners = np.array([
        [-cx, -cy, 0],
        [w - cx, -cy, 0],
        [-cx, h - cy, 0],
        [w - cx, h - cy, 0]
    ])

    rotated_corners = corners @ R.T

    projected = rotated_corners.copy()
    projected[:, 0] = (f * rotated_corners[:, 0]) / (f + rotated_corners[:, 2]) + cx
    projected[:, 1] = (f * rotated_corners[:, 1]) / (f + rotated_corners[:, 2]) + cy

    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_pts = np.float32(projected[:, :2])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return warped

def random_crop(img, crop_scale=(0.7, 0.95)):
    h, w = img.shape[:2]
    scale = random.uniform(*crop_scale)
    new_h, new_w = int(h * scale), int(w * scale)

    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)

    cropped = img[top:top + new_h, left:left + new_w]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized

def rotate_2d(img, angle=None):
    if angle is None:
        angle = random.uniform(-10, 10)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def color_jitter(img):
    img = img.astype(np.float32)

    if random.random() < 0.5:
        factor = random.uniform(0.8, 1.2)
        img *= factor

    if random.random() < 0.5:
        mean = np.mean(img, axis=(0,1), keepdims=True)
        factor = random.uniform(0.8, 1.2)
        img = (img - mean) * factor + mean

    img = np.clip(img, 0, 255)

    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    if random.random() < 0.5:
        hsv[..., 1] *= random.uniform(0.7, 1.3)
    if random.random() < 0.3:
        hsv[..., 0] += random.uniform(-10, 10)
        hsv[..., 0] = np.mod(hsv[..., 0], 180)

    hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img.astype(np.uint8)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def apply_random_filters(img):
    temp = img.copy()

    alpha = random.uniform(0.9, 1.1)
    beta = random.randint(-10, 10)
    temp = cv2.convertScaleAbs(temp, alpha=alpha, beta=beta)

    if random.random() < 0.3:
        temp = cv2.GaussianBlur(temp, (5, 5), 0)

    if random.random() < 0.2:
        noise = np.random.normal(0, 5, temp.shape).astype(np.int16)
        temp = temp.astype(np.int16) + noise
        temp = np.clip(temp, 0, 255).astype(np.uint8)

    if random.random() < 0.5:
        angle_x = random.uniform(-10, 10)
        angle_y = random.uniform(-10, 10)
        angle_z = random.uniform(-10, 10)
        temp = rotate_image_3d(temp, angle_x, angle_y, angle_z)

    if random.random() < 0.5:
        temp = rotate_2d(temp)

    if random.random() < 0.4:
        temp = random_crop(temp)

    if random.random() < 0.3:
        hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(0.85, 1.15)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        temp = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if random.random() < 0.3:
        h, w = temp.shape[:2]
        shift = random.randint(5, 20)
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        if random.choice(['x', 'y']) == 'x':
            dst_pts = np.float32([[shift, 0], [w - shift, 0], [0, h], [w, h]])
        else:
            dst_pts = np.float32([[0, shift], [w, 0], [0, h - shift], [w, h]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        temp = cv2.warpPerspective(temp, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    if random.random() < 0.4:
        temp = color_jitter(temp)

    if random.random() < 0.3:
        temp = sharpen_image(temp)

    return temp

# Streamlit UI and logic
def main():
    st.title("Image Augmentation App")

    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    num_variants = st.slider("Number of Variants per Image", min_value=1, max_value=20, value=5)

    if uploaded_files and st.button("Augment Images"):
        output_folder = "augmented_images"
        os.makedirs(output_folder, exist_ok=True)

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files):
            image_bytes = uploaded_file.read()
            npimg = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            if img is None:
                st.warning(f"Failed to load image {uploaded_file.name}")
                continue

            base_name = os.path.splitext(uploaded_file.name)[0]

            for i in range(num_variants):
                augmented = apply_random_filters(img)
                save_path = os.path.join(output_folder, f"{base_name}_aug_{i}.jpg")
                cv2.imwrite(save_path, augmented)

            status_text.text(f"Processed {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
            progress_bar.progress((idx + 1) / len(uploaded_files))

        st.success(f"Augmentation completed! Images saved to '{output_folder}' folder.")

        # Optionally, list the augmented images with download links
        st.subheader("Augmented Images Preview:")
        for file_name in os.listdir(output_folder):
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(output_folder, file_name)
                image = Image.open(img_path)
                st.image(image, caption=file_name, use_column_width=True)

if __name__ == "__main__":
    main()
