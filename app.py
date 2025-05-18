import os
import cv2
import numpy as np
import random
import math
import streamlit as st
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import zipfile
import io

def rotate_image_3d(img, angle_x=0, angle_y=0, angle_z=0):
    h, w = img.shape[:2]
    ax = math.radians(angle_x)
    ay = math.radians(angle_y)
    az = math.radians(angle_z)

    f = w  # focal length (can be tweaked)
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
        angle = random.uniform(-10, 10)  # Angle restricted between -10 and +10
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

def apply_random_filters(img, num_variants=10):
    results = []

    for _ in range(num_variants):
        temp = img.copy()

        # Brightness & contrast
        alpha = random.uniform(0.9, 1.1)
        beta = random.randint(-10, 10)
        temp = cv2.convertScaleAbs(temp, alpha=alpha, beta=beta)

        # Blur
        if random.random() < 0.3:
            temp = cv2.GaussianBlur(temp, (5, 5), 0)

        # Noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 5, temp.shape).astype(np.int16)
            temp = temp.astype(np.int16) + noise
            temp = np.clip(temp, 0, 255).astype(np.uint8)

        # 3D rotation with restricted angles
        if random.random() < 0.5:
            angle_x = random.uniform(-10, 10)
            angle_y = random.uniform(-10, 10)
            angle_z = random.uniform(-10, 10)
            temp = rotate_image_3d(temp, angle_x, angle_y, angle_z)

        # 2D rotation with restricted angles
        if random.random() < 0.5:
            temp = rotate_2d(temp)

        # Random cropping
        if random.random() < 0.4:
            temp = random_crop(temp)

        # Saturation adjustment
        if random.random() < 0.3:
            hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= random.uniform(0.85, 1.15)
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            temp = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Perspective Transform
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

        # Color jitter
        if random.random() < 0.4:
            temp = color_jitter(temp)

        # Sharpening
        if random.random() < 0.3:
            temp = sharpen_image(temp)

        results.append(temp)

    return results

def process_single_image(args):
    img_path, output_folder, prefix, num_variants = args
    try:
        img = cv2.imread(img_path)
        if img is None:
            return f"Failed to load image: {img_path}"

        filtered_versions = apply_random_filters(img, num_variants=num_variants)
        saved_files = []

        for i, variant in enumerate(filtered_versions):
            out_name = f"{prefix}_v{i}.jpg"
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, variant)
            saved_files.append(out_path)

        return f"Processed {img_path} with {len(filtered_versions)} variants."

    except Exception as e:
        return f"Error processing {img_path}: {e}"

def zip_folder(folder_path):
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname=relative_path)
    memory_file.seek(0)
    return memory_file

def main():
    st.title("Image Augmentation with Random Filters")

    reference_folder = st.text_input("Reference Image Folder Path (Optional, must exist)")
    client_folder = st.text_input("Client Image Folder Path (Required, must exist)")
    output_folder = st.text_input("Output Folder Path (Must exist and writable)")

    if st.button("Process Images"):
        # Basic validation
        if not client_folder or not os.path.isdir(client_folder):
            st.error("Client Image Folder path is invalid or not selected.")
            return
        if not output_folder or not os.path.isdir(output_folder):
            st.error("Output Folder path is invalid or not selected.")
            return
        if reference_folder and not os.path.isdir(reference_folder):
            st.warning("Reference folder path invalid or empty, continuing without it.")

        # Get all images in client folder
        client_images = [os.path.join(client_folder, f) for f in os.listdir(client_folder)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not client_images:
            st.warning("No images found in client folder.")
            return

        # Processing params
        num_variants_per_image = 10
        st.info(f"Processing {len(client_images)} images with {num_variants_per_image} variants each...")

        # Use multiprocessing for speed
        pool = Pool(cpu_count())
        args_list = [(img_path, output_folder, f"img{i}", num_variants_per_image) for i, img_path in enumerate(client_images)]

        results = list(tqdm(pool.imap(process_single_image, args_list), total=len(args_list)))
        pool.close()
        pool.join()

        for r in results:
            st.write(r)

        # After processing, zip output folder and provide download button
        zip_data = zip_folder(output_folder)
        st.download_button(
            label="Download All Processed Images (ZIP)",
            data=zip_data,
            file_name="processed_images.zip",
            mime="application/zip"
        )

if __name__ == "__main__":
    main()
