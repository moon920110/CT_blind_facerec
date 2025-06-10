from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import threading
import requests
import cv2
import time

# Import your face recognition module
from face_recog import FaceRecog

class InitScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        layout = BoxLayout(orientation='vertical', padding=20, spacing=10)

        label = Label(text="어서 오세요 광주 과학관입니다.", font_size=32, font_name='./data/Pretendard-Regular.otf')
        layout.add_widget(label)

        layout.add_widget(Label())  # Spacer to push button down

        start_button = Button(text="Start", size_hint=(1, 0.2), font_size=24)
        start_button.bind(on_press=self.go_to_main)
        layout.add_widget(start_button)

        self.add_widget(layout)

    def go_to_main(self, instance):
        self.manager.current = 'main'
        threading.Thread(target=self.start_recognition_flow, daemon=True).start()

    def start_recognition_flow(self):
        # Start only face detection (not video), since video feed is already running
        self.manager.face_recognition.face_detect()

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


class MainTaskScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        self.image = Image()

        self.layout.add_widget(self.image)

        self.add_widget(self.layout)

        Clock.schedule_interval(self.update_camera, 1.0 / 30)

    def update_camera(self, dt):
        ret, frame = self.manager.face_recognition.cam.read()

        if ret:
            self.manager.face_recognition.img = frame
            self.manager.face_recognition.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x = self.manager.face_recognition._x
            y = self.manager.face_recognition._y
            w = self.manager.face_recognition._w
            h = self.manager.face_recognition._h

            if x is not None and y is not None and w is not None and h is not None:
                if self.manager.face_recognition.enter_region:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

            buf = cv2.flip(frame, 0).tobytes()
            img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            self.image.texture = img_texture

class MyScreenManager(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.face_recognition = FaceRecog()
        self.add_widget(InitScreen(name='init'))
        self.add_widget(MainTaskScreen(name='main'))


class MyApp(App):
    def build(self):
        return MyScreenManager()


if __name__ == '__main__':
    MyApp().run()
