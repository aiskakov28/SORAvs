"""Enhanced feature extraction utilities for video analysis."""

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import os
from PIL import Image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from scenedetect import VideoManager, SceneManager, StatsManager
from scenedetect.detectors import ContentDetector

class FeatureExtractor:
    def __init__(self, model_name='VGG16', layer_name='block5_conv3', use_lpips=True):
        self.model_name = model_name
        self.layer_name = layer_name
        self.use_lpips = use_lpips
        self.feature_extractor = self._build_feature_extractor()

        if use_lpips:
            self.lpips_model = self._build_lpips_model()

        self.object_detector = self._build_object_detector()

    def _build_feature_extractor(self):
        if self.model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False)
            model = Model(inputs=base_model.input,
                         outputs=base_model.get_layer(self.layer_name).output)
            return model
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _build_lpips_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.alexnet(pretrained=True).to(device)
        model.eval()
        return model

    def _build_object_detector(self):
        try:
            prototxt_path = 'models/MobileNetSSD_deploy.prototxt'
            model_path = 'models/MobileNetSSD_deploy.caffemodel'

            if not os.path.exists(prototxt_path) or not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
                print("Object detection model files are missing or incomplete. Disabling object detection.")
                return None

            model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            return model
        except Exception as e:
            print(f"Failed to initialize object detection: {str(e)}")
            print("Object detection will be disabled.")
            return None

    def extract_features(self, frame, target_size=(224, 224)):
        img = cv2.resize(frame, target_size)

        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        features = self.feature_extractor.predict(img, verbose=0)
        return features.flatten()

    def compute_lpips(self, frame1, frame2, target_size=(224, 224)):
        if not self.use_lpips:
            return 0.0

        img1 = cv2.resize(frame1, target_size)
        img2 = cv2.resize(frame2, target_size)

        if img1.shape[2] == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img1_tensor = transform(img1).unsqueeze(0)
        img2_tensor = transform(img2).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img1_tensor = img1_tensor.to(device)
        img2_tensor = img2_tensor.to(device)

        with torch.no_grad():
            feat1 = self.lpips_model.features(img1_tensor).flatten()
            feat2 = self.lpips_model.features(img2_tensor).flatten()

        dist = torch.nn.functional.mse_loss(feat1, feat2).item()
        return dist

    def detect_objects(self, frame):
        if self.object_detector is None:
            return []

        try:
            CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                     "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                     "sofa", "train", "tvmonitor"]

            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5
            )

            self.object_detector.setInput(blob)
            detections = self.object_detector.forward()

            objects = []
            h, w = frame.shape[:2]

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    class_id = int(detections[0, 0, i, 1])

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    objects.append({
                        "class": CLASSES[class_id],
                        "confidence": float(confidence),
                        "box": (startX, startY, endX, endY)
                    })

            return objects
        except Exception as e:
            print(f"Warning: Object detection failed: {str(e)}")
            return []

    def extract_batch_features(self, frames, target_size=(224, 224)):
        if not frames:
            return np.array([])

        features = []
        for frame in frames:
            feature = self.extract_features(frame, target_size)
            features.append(feature)

        return np.array(features)

def compute_color_histogram(frame, bins=256):
    if frame.shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = frame

    hist_r = cv2.calcHist([frame_rgb], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([frame_rgb], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([frame_rgb], [2], None, [bins], [0, 256])

    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()

    hist = np.concatenate([hist_r, hist_g, hist_b])
    return hist

def compute_motion_vectors(frames):
    if len(frames) < 2:
        return np.array([])

    motion_vectors = []
    prev_frame_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

    for i in range(1, len(frames)):
        curr_frame_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, curr_frame_gray,
                                           None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_vectors.append(np.mean(mag))
        prev_frame_gray = curr_frame_gray

    return np.array(motion_vectors)

def extract_frames(video_path, max_frames=100, sample_interval=None):
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return frames

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not sample_interval:
        sample_interval = max(1, total_frames // max_frames)

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames

def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)

    scene_manager.add_detector(ContentDetector(threshold=30.0))

    video_manager.start()

    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()

    scenes = []
    for scene in scene_list:
        scenes.append((scene[0].frame_num, scene[1].frame_num))

    return scenes

def estimate_depth(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        depth_map = np.sqrt(sobelx**2 + sobely**2)

        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        return depth_map
    except Exception as e:
        print(f"Error estimating depth: {str(e)}")
        return None
