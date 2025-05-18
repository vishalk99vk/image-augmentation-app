import streamlit as st
import cv2
import numpy as np
import random
import os

# --- Helper functions ---

def rotate_2d(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def adjust_brightness(img, factor):
    # factor >1 brighter, <1 darker
    img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
    return img

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def random_crop(img, crop_size=(200, 200)):
    h, w = img.shape[:2]
    ch, cw = crop_size
    if ch > h or cw > w:
        return img  # Cannot crop larger than original
    x = random.randint(0, w - cw)
    y = random.randint(0, h - ch)
    cropped = img[y:y+ch, x:x+cw]
    # Resize back to original size (optional)
    cropped_resized = cv2.resize(cropped, (w, h))
    return cropped_resized

def apply_random_filters(img, num_variants=10):
    variants = []
    for _ in range(num_variants):
        augmented = img.copy()
        
        # Random rotation between -30 to 30 degrees
        angle = random.uniform(-30, 30)
        augmented = rotate_2d(augmented, angle)
        
        # Random brightness between 0.7 (darker) to 1.3 (brighter)
        brightness_factor = random.uniform(0.7, 1.3)
        augmented = adjust_brightness(augmented, brightness_factor)
        
        # Randomly sharpen with 50% chance
        if random.random() > 0.5:
            augmented = sharpen_image(augmented)
        
        # Random crop with 30% chance
        if random.random() > 0.7:
            new_h = int(img.shape[0] * 0.8)
            new_w = int(img.shape[1] * 0.8)
            augmented = random_crop(augmented, crop_size=(new_h, new_w))
        
        variants.append(augmented)
    return variants

def process_single_image(img_bytes, num_variants=10):
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Failed to load image")
        return []
    filtered_versions = apply_random_filters(img, num_variants=num_variants)
    return filtered_versions

# --- Streamlit app main ---

def main():
    st.title("Image Augmentation Web App")
    st.write("Upload your client images. Optionally, upload reference images.")

    client_files = st.file_uploader("Upload client images", accept_multiple_files=True, type=['jpg','jpeg','png'])
    reference_files = st.file_uploader("Upload reference images (optional)", accept_multiple_files=True, type=['jpg','jpeg','png'])

    num_variants = st.slider("Number of variants per image", min_value=1, max_value=10, value=5)

    if st.button("Process Images"):
        if not client_files:
            st.error("Please upload at least one client image")
            return

        st.info("Processing images... This may take a while.")

        for i, client_file in enumerate(client_files):
            st.write(f"### Client Image {i+1}: {client_file.name}")
            img_bytes = client_file.read()

            variants = process_single_image(img_bytes, num_variants=num_variants)
            for j, variant in enumerate(variants):
                is_success, buffer = cv2.imencode(".png", variant)
                if is_success:
                    st.image(buffer, caption=f"Variant {j+1}", use_column_width=True)
                    st.download_button(
                        label=f"Download variant {j+1}",
                        data=buffer.tobytes(),
                        file_name=f"{os.path.splitext(client_file.name)[0]}_variant{j+1}.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    main()
