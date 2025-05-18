import os
import cv2
import numpy as np
import random
import math
import streamlit as st
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Your existing augmentation functions (rotate_image_3d, random_crop, rotate_2d, color_jitter, sharpen_image, apply_random_filters) remain the same

def rotate_image_3d(img, angle_x=0, angle_y=0, angle_z=0):
    h, w = img.shape[:2]
    ax = math.radians(angle_x)
    ay = math.radians(angle_y)
    az = math.radians(angle_z)

    f = w  # focal length
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

def apply_random_filters(img, num_variants=10):
    results = []

    for _ in range(num_variants):
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

def main():
    st.title("Image Augmentation with Random Filters")

    st.markdown("""
    Upload or specify folders for:
    - Optional reference images (used for combined processing)
    - Client images (required)
    - Output folder (where augmented images will be saved)
    """)

    # Inputs for folders
    st.write("**Note:** Streamlit does not support selecting folders directly.")
    st.write("You can upload images or specify absolute paths for local runs.")

    # Upload multiple reference images (optional)
    ref_files = st.file_uploader("Upload Reference Images (Optional)", type=['jpg','jpeg','png'], accept_multiple_files=True)

    # Upload multiple client images (required)
    client_files = st.file_uploader("Upload Client Images", type=['jpg','jpeg','png'], accept_multiple_files=True)

    # Input output folder path (must be accessible)
    output_folder = st.text_input("Output Folder Path (must exist and writable)")

    if st.button("Start Augmentation"):

        if not client_files:
            st.error("Please upload client images.")
            return
        if not output_folder:
            st.error("Please specify output folder path.")
            return
        if not os.path.exists(output_folder):
            st.error("Output folder does not exist.")
            return

        # Save uploaded reference images temporarily
        reference_images = []
        if ref_files:
            ref_dir = os.path.join(output_folder, "ref_tmp")
            os.makedirs(ref_dir, exist_ok=True)
            for idx, f in enumerate(ref_files):
                save_path = os.path.join(ref_dir, f"ref_{idx}_{f.name}")
                with open(save_path, "wb") as out_f:
                    out_f.write(f.getbuffer())
                reference_images.append(save_path)

        # Save uploaded client images temporarily
        client_images = []
        client_dir = os.path.join(output_folder, "client_tmp")
        os.makedirs(client_dir, exist_ok=True)
        for idx, f in enumerate(client_files):
            save_path = os.path.join(client_dir, f"client_{idx}_{f.name}")
            with open(save_path, "wb") as out_f:
                out_f.write(f.getbuffer())
            client_images.append(save_path)

        num_variants = 10 if not reference_images else 5
        tasks = []

        if not reference_images:
            st.warning("⚠ No reference images found, applying filters directly to client images...")
            for idx, c_path in enumerate(client_images):
                prefix = f"filtered_{idx}"
                tasks.append((c_path, output_folder, prefix, num_variants))
        else:
            st.info(f"Processing {len(client_images)} client images with {len(reference_images)} reference images...")
            for r_idx, r_path in enumerate(reference_images):
                for c_idx, c_path in enumerate(client_images):
                    prefix = f"ref{r_idx}_client{c_idx}"
                    tasks.append((c_path, output_folder, prefix, 5))

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Multiprocessing can be tricky on some Streamlit deployments; you can run serially if issues arise.
        try:
            with Pool(cpu_count()) as pool:
                results = []
                for i, res in enumerate(pool.imap_unordered(process_single_image, tasks)):
                    results.append(res)
                    progress_bar.progress((i + 1) / len(tasks))
                    status_text.text(res)
        except Exception as e:
            st.error(f"Multiprocessing error: {e}\nTrying sequential processing...")
            results = []
            for i, task in enumerate(tasks):
                res = process_single_image(task)
                results.append(res)
                progress_bar.progress((i + 1) / len(tasks))
                status_text.text(res)

        st.success("Image augmentation completed!")
        st.write("Summary of processing:")
        for r in results:
            st.write(r)

if __name__ == "__main__":
    main()
