import math
import threading
import subprocess

import cv2
import time

from utils.speak import speak

male_video_path='babyshark.mp4'
female_video_path='pororo.mp4'

def PlayVideo(video_path):
    try:
        subprocess.run(['open', '-a', 'IINA', video_path])
    except Exception as e:
        print(f'{e}')

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

        # Turn on camera
        self._camera_ready(1)

    def _camera_ready(self, cam_num=-1):
        # video capture from camera
        self.cam = cv2.VideoCapture(cam_num)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 )
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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

    def face_detect(self):
        print('start face recog')

        while True:
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

            # Detect faces
            faces = self.cascade.detectMultiScale(self.gray,
                                                        scaleFactor=1.1,
                                                        minNeighbors=5,
                                                        minSize=(20, 20),
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


    def run(self):
        detect = threading.Thread(target=self.face_detect)

        detect.start()

        detect.join()


if __name__ == '__main__':
    face_recog = FaceRecog()
    face_recog.run()
    # face_recog.video_detector()
