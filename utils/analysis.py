from deepface import DeepFace

class Analysis:
    def __init__(self):
        self.info = None
        self.state = "idle"

    def face_analysis(self, largest_face, face_y: int | None = None):
        self.state = "processing"

        try:
            rec = DeepFace.analyze(largest_face, actions=['emotion', 'age', 'gender', 'race'])[0]
            emo = rec['dominant_emotion']
            age = rec['age']
            gender = rec['dominant_gender']
            race = rec['dominant_race']
            height_mapping = int(190 - face_y / 10) if face_y is not None else None
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

    def run_analysis_worker(self, face_img, face_y: int | None = None):
        try:
            self.face_analysis(face_img, face_y)
        finally:
            self.state = "done"

    def init(self):
        self.info = None
        self.state = "idle"