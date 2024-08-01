import cv2
from gtts import gTTS
import playsound
import time
from ffpyplayer.player import MediaPlayer

male_video_path='babyshark.mp4'
female_video_path='pororo.mp4'

def speak(text):
    tts = gTTS(text=text, lang='ko')
    filename = 'voice.mp3'
    tts.save(filename)
    playsound.playsound(filename, block=False)

def PlayVideo(video_path):
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    while True:
        grabbed, frame=video.read()
        audio_frame, val = player.get_frame()
        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(28) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame
    video.release()
    cv2.destroyAllWindows()

class FaceRecog:
    def __init__(self):
        self.cascade_filename = 'data/haarcascade_frontalface_alt.xml'
        self.cascade = cv2.CascadeClassifier(self.cascade_filename)

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

        self.age_net = cv2.dnn.readNetFromCaffe(
            'data/deploy_age.prototxt',
            'data/age_net.caffemodel')

        self.gender_net = cv2.dnn.readNetFromCaffe(
            'data/deploy_gender.prototxt',
            'data/gender_net.caffemodel')

        self.age_list = ['(0 ~ 2)', '(4 ~ 6)', '(8 ~ 12)', '(15 ~ 20)',
                         '(25 ~ 32)', '(38 ~ 43)', '(48 ~ 53)', '(60 ~ 100)']
        self.gender_list = ['Male', 'Female']

        self.cam = None
        self._camera_ready(0)

    def _camera_ready(self, cam_num=-1):
        # video capture from camera
        self.cam = cv2.VideoCapture(cam_num)
        print(self.cam.isOpened())

    def video_detector(self):
        print("Start Face Recognition")
        flag_time = time.time()
        face_flag = 0
        insik_flag = 0
        while True:

            ret, img = self.cam.read()
            try:
                img = cv2.resize(img, dsize=None, fx=1.0, fy=1.0)
            except:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results = self.cascade.detectMultiScale(gray,
                                                    scaleFactor=1.1,
                                                    minNeighbors=5,
                                                    minSize=(20, 20),
                                                    )

            for box in results:
                x, y, w, h = box
                face = img[int(y):int(y + h), int(x):int(x + h)].copy()
                blob = cv2.dnn.blobFromImage(face, 1, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

                # gender detection
                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                gender = gender_preds.argmax()
                # Predict age
                self.age_net.setInput(blob)
                age_preds = self.age_net.forward()
                age = age_preds.argmax()

                height_mappling = int(190 - y/10)

                info = self.gender_list[gender] + ' ' + self.age_list[age] + ' ' + str(height_mappling)

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
                cv2.putText(img, info, (x, y - 15), 0, 0.5, (0, 255, 0), 1)
                print(y)

            if len(results) >= 1:
                if time.time() - flag_time >= 3:
                    flag_time = time.time()
                    max_size_face = 0
                    max_size_face_indicator = 0
                    for face in range(len(results)):
                        if results[face][3] >= max_size_face:
                            max_size_face = results[face][3]
                            max_size_face_indicator = face
                    if max_size_face >= 100:
                        if results[max_size_face_indicator][0] < 500 or results[max_size_face_indicator][0] + results[max_size_face_indicator][3] > 1200:
                            speak("가운데로 서주세요.")
                            insik_flag = 0
                        elif face_flag == 1 and insik_flag == 0:
                            speak("인식되었습니다.")
                            insik_flag = 1
                            if self.gender_list[gender] == 'Male': PlayVideo(male_video_path)
                            else: PlayVideo(female_video_path)


                    elif max_size_face >= 100 and max_size_face < 200:
                        speak("조금 더 가까이 와주세요.")
                        insik_flag = 0
                    elif max_size_face < 100:
                        pass
                    face_flag = 1
            else:
                face_flag = 0
                insik_flag = 0
            print(face_flag)

            cv2.imshow('facenet', img)

            if cv2.waitKey(1) > 0:
                break


if __name__ == '__main__':
    face_recog = FaceRecog()
    face_recog.video_detector()
