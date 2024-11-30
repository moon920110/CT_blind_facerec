import math
import threading
import subprocess

import cv2
import time

import playsound

from gtts import gTTS
from deepface import DeepFace

male_video_path='babyshark.mp4'
female_video_path='pororo.mp4'

def speak(text):
    tts = gTTS(text=text, lang='ko')
    filename = 'voice.mp3'
    tts.save(filename)
    playsound.playsound(filename, block=False)

def PlayVideo(video_path):
    try:
        subprocess.run(['open', '-a', 'IINA', video_path])
    except Exception as e:
        print(f'{e}')

    return False

class FaceRecog:
    def __init__(self):
        self.cascade_filename = 'data/haarcascade_frontalface_alt.xml'

        self.cascade = cv2.CascadeClassifier(self.cascade_filename)

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

        # self.age_net = cv2.dnn.readNetFromCaffe(
        #     'data/deploy_age.prototxt',
        #     'data/age_net.caffemodel')
		#
        # self.gender_net = cv2.dnn.readNetFromCaffe(
        #     'data/deploy_gender.prototxt',
        #     'data/gender_net.caffemodel')

        self.age_list = ['(0 ~ 2)', '(4 ~ 6)', '(8 ~ 12)', '(15 ~ 20)',
                         '(25 ~ 32)', '(38 ~ 43)', '(48 ~ 53)', '(60 ~ 100)']
        self.gender_list = ['Male', 'Female']

        self.largest_face = None
        self.img = None
        self.gray = None
        self._x = None
        self._y = None
        self._w = None
        self._h = None
        self.info = None

        self.cam = None
        self.width = None
        self.height = None
        self.image_center = None
        cv2.namedWindow('face')
        self._camera_ready(0)

    def _camera_ready(self, cam_num=-1):
        # video capture from camera
        self.cam = cv2.VideoCapture(cam_num)
        self.width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.image_center = (self.width // 2, self.height // 2)
        print(f'camera open: {self.cam.isOpened()}')

    def _find_largest_central_face(self, faces):
        def score(face):
            x, y, w, h = face
            area = w * h
            face_center = (x + w // 2, y + h // 2)
            distance = math.sqrt((self.image_center[0] - face_center[0]) ** 2 + (self.image_center[1] - face_center[1]) ** 2)
            return (area, -distance)

        largest_face = max(faces, key=score)
        return largest_face

    def face_analysis(self, largest_face):
        try:
            rec = DeepFace.analyze(largest_face, actions=['emotion', 'age', 'gender', 'race'])[0]
            emo = rec['dominant_emotion']
            age = rec['age']
            gender = rec['dominant_gender']
            race = rec['dominant_race']
            height_mappling = int(190 - self._y / 10)
            self.info = f'{emo} {age} {gender} {race} {str(height_mappling)}'
            print(rec)
            if gender == 'Man':
                PlayVideo(male_video_path)
            else:
                PlayVideo(female_video_path)

        except Exception as e:
            print(e)

    #TODO: analysis 인식된 if로 migration하고 한 번만 동작하도록 수정할 것 + 한 사람 당 한 번만 동작하도록 할 것
    def face_detect(self):
        print('start face recog')
        flag_time = time.time()
        while True:
            if self.gray is not None and self.img is not None:
                face_flag = 0
                faces = self.cascade.detectMultiScale(self.gray,
                                                           scaleFactor=1.1,
                                                           minNeighbors=5,
                                                           minSize=(20, 20),
                                                           )
                if len(faces) == 0:
                    continue
                x, y, w, h = self._find_largest_central_face(faces)
                self._y = y
                self._x = x
                self._w = w
                self._h = h
                largest_face = self.img[int(y):int(y + h), int(x):int(x + h)].copy()

                if time.time() - flag_time >= 3:
                    if self._w < 250:
                        speak("한 걸음 가까이 와주세요.")
                    elif self._w > 350:
                        speak("한 걸음 뒤로 가주세요.")
                    else:
                        if self._x < self.img.shape[1] / 3:
                            speak("한 걸음 오른쪽으로 가주세요.")
                        elif self._x + self._w > self.img.shape[1] * 2 / 3:
                            speak("한 걸음 왼쪽으로 가주세요.")
                        # elif face_flag == 1 and not self.recog:
                        else:
                            speak("인식되었습니다. 잠시만 기다려주세요.")
                            self.face_analysis(largest_face)
                    flag_time = time.time()

                    face_flag = 1
                else:
                    face_flag = 0

    def video_detector(self):
        print("Start cam")
        while True:

            ret, self.img = self.cam.read()
            self.img = cv2.resize(self.img, dsize=None, fx=1.0, fy=1.0)
            if self.img is None:
                continue
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            if self._x is not None and self._y is not None and self._w is not None and self._h is not None:
                cv2.rectangle(self.img, (self._x, self._y), (self._x + self._w, self._y + self._h), (255, 255, 255), thickness=2)
                if self.info is not None:
                    cv2.putText(self.img, self.info, (self._x, self._y - 15), 0, 0.5, (0, 255, 0), 1)

            cv2.imshow('face', self.img)

            if cv2.waitKey(1) > 0:
                break

    def run(self):
        # video = multiprocessing.Process(target=self.video_detector)
        detect = threading.Thread(target=self.face_detect)
        # analysis = threading.Thread(target=self.face_analysis)

        # video.start()
        detect.start()
        # analysis.start()

        self.video_detector()

        # video.join()
        detect.join()
        # analysis.join()


if __name__ == '__main__':
    face_recog = FaceRecog()
    face_recog.run()
    # face_recog.video_detector()
