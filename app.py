import streamlit as st
import cv2
import numpy as np
import os

# Dummy placeholder for your actual augmentation function
def apply_random_filters(img, num_variants=10):
    # For testing: just return the original image num_variants times
    return [img.copy() for _ in range(num_variants)]

def process_single_image(img_bytes, num_variants=10):
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Failed to load image")
        return []
    filtered_versions = apply_random_filters(img, num_variants=num_variants)
    return filtered_versions

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
            if not variants:
                st.error(f"No variants returned for image {client_file.name}")
                continue

            for j, variant in enumerate(variants):
                # Encode image to PNG buffer for display and download
                is_success, buffer = cv2.imencode(".png", variant)
                if is_success:
                    st.image(buffer.tobytes(), caption=f"Variant {j+1}", use_container_width=True)
                    st.download_button(
                        label=f"Download variant {j+1}",
                        data=buffer.tobytes(),
                        file_name=f"{os.path.splitext(client_file.name)[0]}_variant{j+1}.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    main()
