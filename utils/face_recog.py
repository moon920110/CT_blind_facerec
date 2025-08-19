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
        self._camera_ready(0)

    def _camera_ready(self, cam_num=-1):
        # video capture from camera
        self.cam = cv2.VideoCapture(cam_num)
        self.cam_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
