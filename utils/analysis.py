from __future__ import annotations

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from deepface import DeepFace
import cv2


class Analysis:
    _HEIGHT_BY_Y = {
        '150cm 미만': 150,
        '150cm 대': 160,
        '160cm 대': 170,
        '170cm 대': 180,
        '180cm 이상': 190,
    }

    # FairFace (multi-task) age config (res34_fair_align_multi_7_20190809.pt)
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # [영유아(0-2), 어린이(3-9), 청소년(10-19), 20대(20-29), 30대(30-39), 40대(40-49), 50대(50-59), 60대(60-69), 70대(70+)]
    _AGE_LABELS_9 = [
        '영유아', '어린이', '청소년', '20대', '30대', '40대', '50대', '60대', '70대'
    ]
    _AGE_BIN_CENTERS_9 = np.array([1, 6, 15, 25, 35, 45, 55, 65, 75], dtype=np.float32)
    _WEIGHTS_PATH = os.path.join('model', 'res34_fair_align_multi_7_20190809.pt')

    def __init__(self, face_recognition):
        self.info = None
        self.state = "idle"

        self.face_recognition = face_recognition

        self._CAMERA_WIDTH, self._CAMERA_HEIGHT, self._WIDTH_RATIO, self._HEIGHT_RATIO, self._CAMERA_CENTER_X, self._CAMERA_CENTER_Y = 0, 0, 0, 0, 0, 0

        # Build FairFace age model once
        self._age_model = None
        self._age_model_ready = False
        try:
            self._age_model = self._build_fairface_multi_model()
            if os.path.exists(self._WEIGHTS_PATH):
                state = torch.load(self._WEIGHTS_PATH, map_location=self._DEVICE)
                # Some checkpoints may have 'state_dict' key
                if isinstance(state, dict) and 'state_dict' in state:
                    self._age_model.load_state_dict(state['state_dict'])
                else:
                    self._age_model.load_state_dict(state)
                self._age_model.to(self._DEVICE)
                self._age_model.eval()
                self._age_model_ready = True
            else:
                print(f"FairFace age weights not found: {self._WEIGHTS_PATH}")
        except Exception as exc:
            print(f"Failed to init FairFace age model: {exc}")

    def init_camera_info(self):
        self._CAMERA_WIDTH = self.face_recognition.cam_width
        self._CAMERA_HEIGHT = self.face_recognition.cam_height
        self._WIDTH_RATIO = self._CAMERA_WIDTH / 640
        self._HEIGHT_RATIO = self._CAMERA_HEIGHT / 480
        self._CAMERA_CENTER_X = self._CAMERA_WIDTH / 2
        self._CAMERA_CENTER_Y = self._CAMERA_HEIGHT / 2

    def face_analysis(self, face_img, face_info: tuple[int, int, int, int]):
        # Initialize camera info
        self.init_camera_info()

        self.state = "processing"

        # Save largest face to file (debug)
        try:
            cv2.imwrite("face_img.jpg", face_img)
        except Exception:
            pass

        try:
            # Use DeepFace for emotion/gender/race only
            rec = DeepFace.analyze(face_img, actions=['emotion', 'gender', 'race'])[0]
            emo = rec['dominant_emotion']
            gender = rec['dominant_gender']
            race = rec['dominant_race']

            (_, y, _, _) = face_info
            height_mapping = "150cm 미만"
            for k, v in self._HEIGHT_BY_Y.items():
                if y > v * self._HEIGHT_RATIO:
                    height_mapping = k
                    continue
                else:
                    break

            # Predict age via FairFace multi-task model (age head only)
            age_label, age_est, age_conf = self._predict_age_fairface_multi(face_img)

            self.info = {
                'emotion': emo.lower(),
                'age': age_label if age_label is not None else rec.get('age'),  # prefer FairFace label
                'gender': gender.lower(),
                'race': race.lower(),
                'height': height_mapping,
                'age_estimate': int(age_est) if age_est is not None else None,
                'age_confidence': age_conf,
            }

        except Exception as e:
            print(e)
            self.info = {
                'emotion': "Not Detected",
                'age': "Not Detected",
                'gender': "Not Detected",
                'race': "Not Detected",
                'height': "Not Detected",
                'error': str(e),
            }

    def run_analysis_worker(self, face_img, face_info: tuple[int, int, int, int]):
        try:
            self.face_analysis(face_img, face_info)
        finally:
            self.state = "done"

    def init(self):
        self.info = None
        self.state = "idle"

    # ===== FairFace age utilities (multi-task checkpoint) =====
    def _build_fairface_multi_model(self) -> nn.Module:
        # Multi-task checkpoint expects output dim 18: [7 race + 2 gender + 9 age]
        model = models.resnet34(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 18)
        return model

    _img_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def _predict_age_fairface_multi(self, face_bgr: np.ndarray):
        if not self._age_model_ready:
            return None, None, None
        try:
            # face_bgr: HxWx3 BGR numpy
            face_tensor = self._img_transform(face_bgr).unsqueeze(0).to(self._DEVICE)
            logits = self._age_model(face_tensor)  # [1,18]
            logits = logits.detach().cpu().numpy().squeeze()  # [18]
            # Split heads
            age_logits = logits[9:18]  # last 9 entries for age bins
            age_probs = np.exp(age_logits) / np.sum(np.exp(age_logits))
            age_idx = int(np.argmax(age_probs))
            age_label = self._AGE_LABELS_9[age_idx]
            age_conf = float(age_probs[age_idx])
            age_est = float((age_probs * self._AGE_BIN_CENTERS_9).sum())
            return age_label, age_est, age_conf
        except Exception as exc:
            print(f"FairFace age prediction failed: {exc}")
            return None, None, None