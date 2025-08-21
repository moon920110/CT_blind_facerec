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

    def __init__(self, face_recognition):
        self.info = None
        self.state = "idle"

        self.face_recognition = face_recognition

        self._CAMERA_WIDTH, self._CAMERA_HEIGHT, self._WIDTH_RATIO, self._HEIGHT_RATIO, self._CAMERA_CENTER_X, self._CAMERA_CENTER_Y = 0, 0, 0, 0, 0, 0

    def init_camera_info(self):
        self._CAMERA_WIDTH = self.face_recognition.cam_width
        self._CAMERA_HEIGHT = self.face_recognition.cam_height
        self._WIDTH_RATIO = self._CAMERA_WIDTH / 640
        self._HEIGHT_RATIO = self._CAMERA_HEIGHT / 480
        self._CAMERA_CENTER_X = self._CAMERA_WIDTH / 2
        self._CAMERA_CENTER_Y = self._CAMERA_HEIGHT / 2

    def face_analysis(self, largest_face, face_info: dict):
        self.init_camera_info()

        self.state = "processing"

        #save largest face to file
        cv2.imwrite("largest_face.jpg", largest_face)

        try:
            rec = DeepFace.analyze(largest_face, actions=['emotion', 'age', 'gender', 'race'])[0]
            emo = rec['dominant_emotion']
            age = rec['age']
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

            print(rec)
            self.info = {
                'emotion': emo.lower(),
                'age': age,
                'gender': gender.lower(),
                'race': race.lower(),
                'height': height_mapping,
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

    def run_analysis_worker(self, face_img, face_info: dict):
        try:
            self.face_analysis(face_img, face_info)
        finally:
            self.state = "done"

    def init(self):
        self.info = None
        self.state = "idle"