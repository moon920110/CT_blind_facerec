import cv2
import threading
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window

from utils.speak import speak

font_path = './data/Pretendard-Regular.otf'

def smooth_rect(prev_rect, target_rect, alpha):
	"""Linearly interpolate rectangle (x,y,w,h) toward target.

	- prev_rect: tuple or None
	- target_rect: tuple (x,y,w,h)
	- alpha: 0..1
	"""
	if target_rect is None:
		return prev_rect
	if prev_rect is None:
		return target_rect
	px, py, pw, ph = prev_rect
	tx, ty, tw, th = target_rect
	return (
		px + alpha * (tx - px),
		py + alpha * (ty - py),
		pw + alpha * (tw - pw),
		ph + alpha * (th - ph),
	)

class MainTaskScreen(Screen):
    def __init__(self, **kwargs):
        print("MainTaskScreen init")

        super().__init__(**kwargs)

        # Use FloatLayout to overlay text on top of the full-screen camera
        self.layout = FloatLayout()
        self.image = Image(allow_stretch=True, keep_ratio=False, size_hint=(1, 1), pos=(0, 0))
        self.layout.add_widget(self.image)

        # Message label
        self.text = Label(text='', font_size=60, halign='center', valign='middle', font_name=font_path,
                           size_hint=(1, None), height=120, outline_color=(0, 0, 0), outline_width=5,
                           pos_hint={'y': 0.12})
        
        # Ensure text is centered across width
        def _bind_text_size(instance, value):
            instance.text_size = (instance.width, None)
        def _on_texture_size(instance, value):
            instance.height = instance.texture_size[1] + 10
        self.text.bind(size=_bind_text_size, texture_size=_on_texture_size)
        self.layout.add_widget(self.text)

        self.add_widget(self.layout)

        # smoothing state
        self._smoothed_rect = None  # (x, y, w, h) floats
        self._SMOOTH_ALPHA = 0.2    # higher = faster response

        # Timer
        self._timer = 0

        self._ERROR_RANGE_X = 50
        self._CORRECT_SIZE_X_MIN = 150
        self._CORRECT_SIZE_X_MAX = 210

        self._is_correct_size = False
        self._is_correct_position = False

        self._start_analysis = False

        # Update camera
        Clock.schedule_interval(self.update_camera, 1.0 / 30)


    def on_pre_enter(self):
        self._timer = 0
        self._smoothed_rect = None
        self._is_correct_size = False
        self._is_correct_position = False
        self._start_analysis = False
        self.manager.analysis.init()

        self.text.text = ""


    def update_camera(self, dt):
        if self.manager.current != 'main':
            return

        self._timer += dt

        # If camera is not ready, return
        if not self.manager.face_recognition.ret:
            return

        # Get frame
        frame = self.manager.face_recognition.img

        # If analysis is done, go to result screen
        if self._start_analysis and self.manager.analysis.state == "done":
            self._start_analysis = False
            self.manager.current = 'result'

        # If face is detected, draw rectangle
        elif not self._start_analysis and self.manager.analysis.state == "idle" and self.manager.face_recognition.has_face:
            # self.draw_standard_line(frame)
            self.draw_rectangle(frame)
            self.check_correct_position()
        
        self.show_frame(frame)


    def draw_rectangle(self, frame):
        x = self.manager.face_recognition._x
        y = self.manager.face_recognition._y
        w = self.manager.face_recognition._w
        h = self.manager.face_recognition._h

        self._smoothed_rect = smooth_rect(self._smoothed_rect, (x, y, w, h), self._SMOOTH_ALPHA)
        sx, sy, sw, sh = [int(v) for v in self._smoothed_rect] 

        color = (0, 255, 0) if self._is_correct_size else (0, 0, 255)

        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), color, thickness=2)
        center = (int(sx + sw/2), int(sy + sh/2))
        # cv2.putText(frame, f"{w}x{h}", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # cv2.circle(frame, center, 5, color, -1)

        if not self._is_correct_position:
            p1 = (center[0] + 40, center[1]) if center[0] > self.manager.face_recognition.cam_width/2 else (center[0] - 40, center[1])
            p2 = (center[0] - 40, center[1]) if center[0] > self.manager.face_recognition.cam_width/2 else (center[0] + 40, center[1])
            cv2.arrowedLine(frame, p1, p2, (0, 0, 255), 2, tipLength=0.5)


    def show_frame(self, frame):
        # Crop the sides to fix aspect to 9:16 (portrait) without rotation
        src_h, src_w = frame.shape[0], frame.shape[1]
        target_ratio = 9 / 16.0  # width / height

        target_w_from_h = int(src_h * target_ratio)
        if src_w >= target_w_from_h:
            x_start = (src_w - target_w_from_h) // 2
            cropped = frame[:, x_start:x_start + target_w_from_h]
        else:
            # If input is narrower than 9:16, crop height instead
            target_h_from_w = int(src_w / target_ratio)
            y_start = (src_h - target_h_from_w) // 2
            cropped = frame[y_start:y_start + target_h_from_w, :]

        dst_w, dst_h = Window.size
        resized = cv2.resize(cropped, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

        # Update image
        buf = cv2.flip(resized, 0).tobytes()
        img_texture = Texture.create(size=(dst_w, dst_h), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        self.image.texture = img_texture
         
    def draw_standard_line(self, frame):
        camera_width = self.manager.face_recognition.cam_width or frame.shape[1]
        cx = int(camera_width / 2)
        ex = int(self._ERROR_RANGE_X)
        cv2.line(frame, (cx, 0), (cx, frame.shape[0]), (0, 0, 255), 2)
        cv2.line(frame, (cx - ex, 0), (cx - ex, frame.shape[0]), (0, 0, 255), 2)
        cv2.line(frame, (cx + ex, 0), (cx + ex, frame.shape[0]), (0, 0, 255), 2)

    def check_correct_position(self):
        
        def print_text(text, force=False):
            if self._timer > 2 or force:
                self.text.text = text
                speak(text)
                self._timer = 0

        if self._timer > 1:
            self.text.text = ""
        
        # Check size of face
        self._is_correct_size = True
        if self.manager.face_recognition._w < self._CORRECT_SIZE_X_MIN:
            self._is_correct_size = False
            print_text("한 걸음 가까이 와주세요.")
        
        elif self.manager.face_recognition._w > self._CORRECT_SIZE_X_MAX:
            self._is_correct_size = False
            print_text("한 걸음 뒤로 가주세요.")

        # Check position of face
        current_x = self.manager.face_recognition._x + self.manager.face_recognition._w / 2
        camera_width = self.manager.face_recognition.cam_width

        self._is_correct_position = True
        if current_x > camera_width/2 + self._ERROR_RANGE_X:
            self._is_correct_position = False
            print_text("한 걸음 왼쪽으로 가주세요.")
        
        elif current_x < camera_width/2 - self._ERROR_RANGE_X:
            self._is_correct_position = False
            print_text("한 걸음 오른쪽으로 가주세요.")
        
        # If face is in the correct position and has correct size, start analysis
        # 안내 멘트 및 분석을 백그라운드에서 처리하여 프레임 수집이 멈추지 않도록 함
        if self._is_correct_size and self._is_correct_position and not self._start_analysis:
            self._start_analysis = True
            print_text("인식되었습니다.\n잠시만 기다려주세요.", force=True)
            face_img = self.manager.face_recognition.largest_face.copy() if self.manager.face_recognition.largest_face is not None else None
            face_y = int(self.manager.face_recognition._y) if self.manager.face_recognition._y is not None else None
            if face_img is not None:
                threading.Thread(target=self.manager.analysis.run_analysis_worker, args=(face_img, face_y), daemon=True).start()

    
