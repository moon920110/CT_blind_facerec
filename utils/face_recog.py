import math
import threading
import os
import platform
import subprocess

import cv2
import time

from utils.speak import speak
from utils.config import config

male_video_path='babyshark.mp4'
female_video_path='pororo.mp4'

def PlayVideo(video_path: str) -> bool:
    """Open video with default OS handler (non-blocking if possible)."""
    try:
        system = platform.system()
        if system == 'Windows':
            os.startfile(video_path)  # type: ignore[attr-defined]
        elif system == 'Darwin':
            subprocess.Popen(['open', video_path])
        else:
            subprocess.Popen(['xdg-open', video_path])
        return True
    except Exception as exc:
        print(f'Failed to open video: {exc}')
        return False

class FaceRecog:
    def __init__(self):
        # Cascade
        cascade_filename = 'data/haarcascade_frontalface_alt.xml'
        self.cascade = cv2.CascadeClassifier(cascade_filename)

        # Face
        self.has_face = False
        self.largest_face = None
        self.ret = False
        self.img = None
        self.gray = None
        self._x = None
        self._y = None
        self._w = None
        self._h = None
        self.info = None

        # Camera
        self.cam = None
        self.cam_width = None
        self.cam_height = None
        self.cam_center = None
        
        # Threading
        self.detect_thread = None
        self.running = False

        # Turn on camera
        self._camera_ready()
        self.start_detection()

    def _camera_ready(self, cam_num=-1):
        # video capture from camera
        camera_index = config.get_camera_index() if cam_num == -1 else cam_num
        self.cam = cv2.VideoCapture(camera_index)
        
        # Get camera resolution from config
        camera_width = config.get('camera.width', 1920)
        camera_height = config.get('camera.height', 1080)
        
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

        self.cam_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'camera width: {self.cam_width}, height: {self.cam_height}')
        self.cam_center = (self.cam_width // 2, self.cam_height // 2)
        print(f'camera open: {self.cam.isOpened()}')

    def _find_largest_central_face(self, faces):
        def score(face):
            x, y, w, h = face
            area = w * h
            face_center = (x + w // 2, y + h // 2)
            distance = math.sqrt((self.cam_center[0] - face_center[0]) ** 2 + (self.cam_center[1] - face_center[1]) ** 2)
            return (area, -distance)

        largest_face = max(faces, key=score)
        return largest_face

    def start_detection(self):
        """Start face detection thread"""
        if self.detect_thread is None or not self.detect_thread.is_alive():
            self.running = True
            self.detect_thread = threading.Thread(target=self.face_detect, daemon=True)
            self.detect_thread.start()
    
    def stop_detection(self):
        """Stop face detection thread"""
        self.running = False
        if self.detect_thread and self.detect_thread.is_alive():
            self.detect_thread.join(timeout=1.0)

    def face_detect(self):
        print('start face recog')

        while self.running:
            # Read frame with proper guards to avoid crashes/freezes
            self.ret, frame = self.cam.read()
            if not self.ret or frame is None:
                self.has_face = False
                time.sleep(0.005)
                continue

            self.img = cv2.flip(frame, 1) 

            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

            # If gray or img is None, continue
            if self.gray is None or self.img is None:
                continue

            # Detect faces using config parameters
            detection_params = config.get_face_detection_params()
            faces = self.cascade.detectMultiScale(self.gray,
                                                        scaleFactor=detection_params.get('scale_factor', 1.1),
                                                        minNeighbors=detection_params.get('min_neighbors', 5),
                                                        minSize=tuple(detection_params.get('min_size', [20, 20])),
                                                        )
            
            # 9:16 비율 중심 ROI 계산 및 ROI 밖 얼굴 제거
            img_h, img_w = self.img.shape[0], self.img.shape[1]
            target_w_from_h = int(img_h * 9 / 16)
            if img_w >= target_w_from_h:
                x0 = (img_w - target_w_from_h) // 2
                y0 = 0
                x1 = x0 + target_w_from_h
                y1 = img_h
            else:
                target_h_from_w = int(img_w * 16 / 9)
                y0 = (img_h - target_h_from_w) // 2
                x0 = 0
                y1 = y0 + target_h_from_w
                x1 = img_w

            roi_cx, roi_cy = (x0 + x1) // 2, (y0 + y1) // 2

            faces_in_roi = []
            for (fx, fy, fw, fh) in faces:
                # 얼굴 전체가 ROI 안에 있을 때만 인정 (보다 느슨하게 하려면 중심점만 체크)
                if fx >= x0 and fy >= y0 and (fx + fw) <= x1 and (fy + fh) <= y1:
                    faces_in_roi.append((fx, fy, fw, fh))

            faces = faces_in_roi

            # If no faces are detected, continue
            if len(faces) == 0:
                self.has_face = False
                continue

            self.has_face = True

            # Find largest face
            x, y, w, h = self._find_largest_central_face(faces)
            self._y = y
            self._x = x
            self._w = w
            self._h = h
            self.largest_face = self.img[int(y):int(y + h), int(x):int(x + h)].copy()


    def reload_camera(self):
        """Reload camera with new settings from config"""
        print("Reloading camera...")
        
        # Stop current detection
        print("Stopping detection thread...")
        self.stop_detection()
        
        # Release current camera
        if self.cam is not None:
            print("Releasing current camera...")
            self.cam.release()
            time.sleep(0.1)  # Give time for camera to release
        
        # Reinitialize camera with new settings
        print("Reinitializing camera with new settings...")
        self._camera_ready()
        
        # Restart detection
        print("Restarting detection thread...")
        self.start_detection()
        
        print("Camera reloaded successfully")
    
    def run(self):
        """Run face detection (for standalone use)"""
        self.start_detection()
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_detection()


if __name__ == '__main__':
    face_recog = FaceRecog()
    face_recog.run()
    # face_recog.video_detector()
