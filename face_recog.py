import cv2


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

                info = self.gender_list[gender] + ' ' + self.age_list[age]

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
                cv2.putText(img, info, (x, y - 15), 0, 0.5, (0, 255, 0), 1)

            cv2.imshow('facenet', img)

            if cv2.waitKey(1) > 0:
                break


if __name__ == '__main__':
    face_recog = FaceRecog()
    face_recog.video_detector()
