from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from kivy.uix.label import Label
from kivy.uix.button import Button

from utils.speak import speak

font_path = './data/Pretendard-Regular.otf'

class ResultScreen(Screen):
    _FONT_SIZE = 60

    _EMONTION_IN_KOREAN = {
        'sad': '슬픔',
        'angry': '화남',
        'surprise': '놀람',
        'fear': '무서움',
        'happy': '행복',
        'disgust': '분노',
        'neutral': '중립',
    }

    _GENDER_IN_KOREAN = {
        'man': '남자',
        'woman': '여자',
    }

    _RACE_IN_KOREAN = {
        'indian': '인도인',
        'asian': '아시아인',
        'latino hispanic': '라틴/히스패닉인',
        'black': '흑인',
        'middle eastern': '중동인',
        'white': '백인',
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.add_widget(self.layout)

        Label(text="결과", font_size=self._FONT_SIZE, font_name=font_path)
        self._labels = {
            'emotion': Label(text="감정: ", font_size=self._FONT_SIZE, font_name=font_path),
            'age': Label(text="나이: ", font_size=self._FONT_SIZE, font_name=font_path),
            'gender': Label(text="성별: ", font_size=self._FONT_SIZE, font_name=font_path),
            'race': Label(text="인종: ", font_size=self._FONT_SIZE, font_name=font_path),
            'height': Label(text="키: ", font_size=self._FONT_SIZE, font_name=font_path),
            'msg': Label(text="", font_size=self._FONT_SIZE, font_name=font_path),
        }
        for k in ['emotion', 'age', 'gender', 'race', 'height', 'msg']:
            self.layout.add_widget(self._labels[k])

        self.layout.add_widget(Button(text="다시 시작", font_size=self._FONT_SIZE, font_name=font_path, on_press=self.restart))

    def on_pre_enter(self):
        info = (self.manager.analysis.info or {}).copy()
        self._labels['emotion'].text = f"감정: {self._EMONTION_IN_KOREAN.get(info.get('emotion', ''), '')}"
        self._labels['age'].text = f"나이: {info.get('age', '')}세"
        self._labels['gender'].text = f"성별: {self._GENDER_IN_KOREAN.get(info.get('gender', ''), '')}"
        self._labels['race'].text = f"인종: {self._RACE_IN_KOREAN.get(info.get('race', ''), '')}"
        self._labels['height'].text = f"키: {info.get('height', '')}cm"
        
        msg = "오류가 발생했습니다. 다시 시작해주세요."
        if 'error' not in info:
            msg = "A 코스로 안내드리겠습니다." if info.get('age', 0) > 18 else "B 코스로 안내드리겠습니다."

        self._labels['msg'].text = msg
        speak(msg, force=True)

    def restart(self, *args):
        self.manager.current = 'main'