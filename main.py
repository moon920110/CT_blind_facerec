import sys

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager

# Import screens
from screen.IntroScreen import IntroScreen
from screen.MainTaskScreen import MainTaskScreen
from screen.ResultScreen import ResultScreen

# Import face recognition module
from utils.face_recog import FaceRecog
from utils.analysis import Analysis


# Prevent console output from being garbled
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

# Screen manager
class MyScreenManager(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Face recognition
        # Camera is turned on when FaceRecog is initialized
        self.face_recognition = FaceRecog()
        self.analysis = Analysis(self.face_recognition)

        # Add screens
        self.add_widget(IntroScreen(name='init'))
        self.add_widget(MainTaskScreen(name='main'))
        self.add_widget(ResultScreen(name='result'))

# App
class MyApp(App):
    def build(self):
        return MyScreenManager()


if __name__ == '__main__':
    MyApp().run()
