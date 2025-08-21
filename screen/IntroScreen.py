from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
import threading
import requests

font_path = './data/Pretendard-Regular.otf'

class IntroScreen(Screen):
    _FONT_SIZE = 60

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)

        label = Label(text="어서 오세요 국립광주과학관입니다.", font_size=self._FONT_SIZE, font_name=font_path)
        layout.add_widget(label)

        start_button = Button(text="시작하기", size_hint=(1, 0.2), font_size=self._FONT_SIZE, font_name=font_path)
        start_button.bind(on_press=self.go_to_main)
        layout.add_widget(start_button)

        self.add_widget(layout)

    def go_to_main(self, instance):
        self.manager.current = 'main'
        threading.Thread(target=self.start_recognition_flow, daemon=True).start()

    def start_recognition_flow(self):
        # Start only face detection (not video), since video feed is already running
        self.manager.face_recognition.face_detect()

        # 어차피 실행 안 됨
        """ 
        result = self.manager.face_recognition.info
        if result:
            try:
                self.manager.face_recognition.info = None
                self.manager.face_recognition.enter_region = False
                print(f"Result: {result}")
                response = requests.post('http://your-server-address/api/result', json={'result': result})  # Replace with your server
                print(f"Server response: {response.status_code}, {response.text}")
            except Exception as e:
                print(f"Failed to send result: {e}")

        # Return to init screen on main thread
        Clock.schedule_once(lambda dt: setattr(self.manager, 'current', 'init'), 0)
        """