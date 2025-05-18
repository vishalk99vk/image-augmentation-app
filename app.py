import streamlit as st
import os
import cv2
import numpy as np
import random
import math
import zipfile
from io import BytesIO
from tempfile import TemporaryDirectory
import torch
from torchvision import transforms
from torchvision.models import vgg19

# Load VGG features once for style transfer
@st.cache_resource
def load_vgg():
    vgg = vgg19(pretrained=True).features.eval()
    for param in vgg.parameters():
        param.requires_grad = False
    return vgg

def image_to_tensor(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def tensor_to_image(tensor):
    unloader = transforms.Compose([
        transforms.Normalize(mean=[-2.12, -2.04, -1.80],
                             std=[4.37, 4.46, 4.44]),
        transforms.ToPILImage()
    ])
    tensor = tensor.cpu().clone().squeeze(0)
    return np.array(unloader(tensor))

# Adaptive Instance Normalization
def adaptive_instance_normalization(content_feat, style_feat):
    content_mean, content_std = content_feat.mean([2, 3], keepdim=True), content_feat.std([2, 3], keepdim=True)
    style_mean, style_std = style_feat.mean([2, 3], keepdim=True), style_feat.std([2, 3], keepdim=True)
    normalized_feat = (content_feat - content_mean) / content_std
    return normalized_feat * style_std + style_mean

def extract_features(img, model, layers):
    features = {}
    x = img
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

def apply_style_transfer(content_img, style_img, alpha=1.0):
    vgg = load_vgg()
    content_tensor = image_to_tensor(content_img)
    style_tensor = image_to_tensor(style_img)

    content_features = extract_features(content_tensor, vgg, layers=['21'])  # relu4_1
    style_features = extract_features(style_tensor, vgg, layers=['21'])

    t = adaptive_instance_normalization(content_features['21'], style_features['21'])
    t = alpha * t + (1 - alpha) * content_features['21']

    return tensor_to_image(t)

# Augmentation functions
def rotate_image_3d(img, angle_x=0, angle_y=0, angle_z=0):
    h, w = img.shape[:2]
    ax, ay, az = math.radians(angle_x), math.radians(angle_y), math.radians(angle_z)
    f, cx, cy = w, w / 2, h / 2

    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx
    corners = np.array([[-cx, -cy, 0], [w - cx, -cy, 0], [-cx, h - cy, 0], [w - cx, h - cy, 0]])
    rotated = corners @ R.T
    projected = rotated.copy()
    projected[:, 0] = (f * rotated[:, 0]) / (f + rotated[:, 2]) + cx
    projected[:, 1] = (f * rotated[:, 1]) / (f + rotated[:, 2]) + cy

    matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [0, h], [w, h]]), projected[:, :2].astype(np.float32))
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

def rotate_2d(img, angle=None):
    if angle is None:
        angle = random.uniform(-10, 10)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

def random_crop(img, crop_scale=(0.7, 0.95)):
    h, w = img.shape[:2]
    scale = random.uniform(*crop_scale)
    nh, nw = int(h * scale), int(w * scale)
    top, left = random.randint(0, h - nh), random.randint(0, w - nw)
    return cv2.resize(img[top:top + nh, left:left + nw], (w, h))

def color_jitter(img):
    img = img.astype(np.float32)
    if random.random() < 0.5:
        img *= random.uniform(0.8, 1.2)
    hsv = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= random.uniform(0.7, 1.3)
    hsv[..., 0] = np.mod(hsv[..., 0] + random.uniform(-10, 10), 180)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_mobile_effects(img):
    temp = img.copy()
    if random.random() < 0.4:
        kernel = np.zeros((5, 5))
        kernel[2, :] = np.ones(5)
        temp = cv2.filter2D(temp, -1, kernel / 5)
    if random.random() < 0.4:
        temp = cv2.convertScaleAbs(temp, alpha=random.uniform(0.8, 1.2), beta=random.randint(-30, 30))
    if random.random() < 0.3:
        h, w = temp.shape[:2]
        x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
        x2, y2 = random.randint(w//2, w), random.randint(h//2, h)
        overlay = temp.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        alpha = random.uniform(0.2, 0.5)
        temp = cv2.addWeighted(overlay, alpha, temp, 1 - alpha, 0)
    return temp

def apply_random_filters(img):
    temp = apply_mobile_effects(img)
    if random.random() < 0.3:
        temp = cv2.GaussianBlur(temp, (5, 5), 0)
    if random.random() < 0.2:
        noise = np.random.normal(0, 5, temp.shape).astype(np.int16)
        temp = np.clip(temp.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.5:
        temp = rotate_image_3d(temp, random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10))
    if random.random() < 0.5:
        temp = rotate_2d(temp)
    if random.random() < 0.4:
        temp = random_crop(temp)
    if random.random() < 0.4:
        temp = color_jitter(temp)
    return temp

# Streamlit UI
st.title("🧠 AI Image Augmentation App")

style_mode = st.radio("Choose Augmentation Type", ["📱 Mobile Simulation", "🎨 Style Transfer (From Reference Image)"])

reference_image = None
if style_mode == "🎨 Style Transfer (From Reference Image)":
    reference_image = st.file_uploader("Upload Reference Style Image", type=["jpg", "jpeg", "png"])

uploaded_files = st.file_uploader("Upload Client Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
num_variants = st.slider("Number of Variants per Image", 1, 10, 3)

if uploaded_files and st.button("Start Augmentation"):
    with TemporaryDirectory() as output_folder:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            base_name = os.path.splitext(uploaded_file.name)[0]

            for i in range(num_variants):
                if style_mode == "📱 Mobile Simulation":
                    augmented = apply_random_filters(img)
                elif style_mode == "🎨 Style Transfer (From Reference Image)" and reference_image:
                    reference_image.seek(0)  # Reset pointer for repeated reads
                    ref_bytes = np.asarray(bytearray(reference_image.read()), dtype=np.uint8)
                    style_img = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)
                    content_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    style_pil = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
                    augmented = apply_style_transfer(content_pil, style_pil)
                    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                else:
                    st.error("Please upload a reference image.")
                    break

                out_path = os.path.join(output_folder, f"{base_name}_aug_{i}.jpg")
                cv2.imwrite(out_path, augmented)

        # ZIP download
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for file_name in os.listdir(output_folder):
                zip_file.write(os.path.join(output_folder, file_name), arcname=file_name)
        zip_buffer.seek(0)
        st.download_button("📥 Download ZIP", zip_buffer, "augmented_images.zip", "application/zip")
