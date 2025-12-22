from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.popup import Popup
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.clock import Clock

from utils.config import config

font_path = './data/Pretendard-Regular.otf'

class SettingsPopup(Popup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Avoid Korean in Popup native title (may break font)
        self.title = ""
        self.size_hint = (0.9, 0.9)
        self.auto_dismiss = False
        
        # Create main layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        # Header label for title using proper font
        header = Label(text="설정", font_name=font_path, size_hint_y=None, height=40)
        main_layout.add_widget(header)
        
        # Create tabbed panel
        self.tab_panel = TabbedPanel(do_default_tab=False)
        
        # Create tabs
        self.create_camera_tab()
        self.create_height_tab()
        self.create_display_tab()
        self.create_detection_tab()
        
        main_layout.add_widget(self.tab_panel)
        
        # Create buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1, spacing=10)
        
        self.save_button = Button(text="저장", font_name=font_path)
        self.save_button.bind(on_press=self.save_settings)
        
        self.reset_button = Button(text="기본값으로 초기화", font_name=font_path)
        self.reset_button.bind(on_press=self.reset_to_defaults)
        
        self.close_button = Button(text="닫기", font_name=font_path)
        self.close_button.bind(on_press=self.close_popup)
        
        button_layout.add_widget(self.reset_button)
        button_layout.add_widget(self.save_button)
        button_layout.add_widget(self.close_button)
        
        main_layout.add_widget(button_layout)
        
        # Load current settings
        self.load_current_settings()
        
        self.content = main_layout
    
    def create_camera_tab(self):
        """Create camera settings tab"""
        camera_tab = TabbedPanelItem(text="카메라", font_name=font_path)
        camera_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Camera index
        camera_index_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        camera_index_layout.add_widget(Label(text="카메라 인덱스:", font_name=font_path, size_hint_x=0.5))
        self.camera_index_input = TextInput(text="0", multiline=False, size_hint_x=0.5, font_name=font_path)
        self.camera_index_input.bind(text=self.on_camera_index_change)
        camera_index_layout.add_widget(self.camera_index_input)
        camera_layout.add_widget(camera_index_layout)
        
        # Camera resolution
        resolution_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        resolution_layout.add_widget(Label(text="카메라 너비:", font_name=font_path, size_hint_x=0.5))
        self.camera_width_input = TextInput(text="1920", multiline=False, size_hint_x=0.5, font_name=font_path)
        self.camera_width_input.bind(text=self.on_camera_width_change)
        resolution_layout.add_widget(self.camera_width_input)
        camera_layout.add_widget(resolution_layout)
        
        resolution_layout2 = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        resolution_layout2.add_widget(Label(text="카메라 높이:", font_name=font_path, size_hint_x=0.5))
        self.camera_height_input = TextInput(text="1080", multiline=False, size_hint_x=0.5, font_name=font_path)
        self.camera_height_input.bind(text=self.on_camera_height_change)
        resolution_layout2.add_widget(self.camera_height_input)
        camera_layout.add_widget(resolution_layout2)
        
        camera_tab.add_widget(camera_layout)
        self.tab_panel.add_widget(camera_tab)
    
    def create_height_tab(self):
        """Create height mapping settings tab"""
        height_tab = TabbedPanelItem(text="키 범위", font_name=font_path)
        height_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Height mapping inputs
        self.height_inputs = {}
        height_ranges = [
            ("150cm 대", "height_150"),
            ("160cm 대", "height_160"),
            ("170cm 대", "height_170"),
            ("180cm 이상", "height_180_over")
        ]
        
        for label, key in height_ranges:
            input_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1)
            input_layout.add_widget(Label(text=f"{label}:", font_name=font_path, size_hint_x=0.5))
            text_input = TextInput(multiline=False, size_hint_x=0.5, font_name=font_path)
            text_input.bind(text=lambda instance, value, k=key: self.on_height_change(k, value))
            self.height_inputs[key] = text_input
            input_layout.add_widget(text_input)
            height_layout.add_widget(input_layout)
        
        # Info label
        info_label = Label(
            text="키 범위는 Y 좌표 기준으로 설정됩니다.\n낮은 값일수록 화면 상단에 위치합니다.",
            font_name=font_path,
            size_hint_y=0.2,
            text_size=(None, None),
            halign='center'
        )
        height_layout.add_widget(info_label)
        
        height_tab.add_widget(height_layout)
        self.tab_panel.add_widget(height_tab)
    
    def create_display_tab(self):
        """Create display options tab"""
        display_tab = TabbedPanelItem(text="화면 표시", font_name=font_path)
        display_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Display options checkboxes
        self.display_checkboxes = {}
        display_options = [
            ("draw_face_rect", "얼굴 사각형 표시"),
            ("draw_standard_line", "기준선 표시"),
            ("draw_arrow", "화살표 표시"),
            ("draw_face_position", "얼굴 위치 표시"),
            ("draw_face_size", "얼굴 크기 표시"),
            ("show_height_lines", "키 기준선(디버그) 표시"),
            ("disable_auto_navigate", "분석 후 자동 화면전환 비활성화")
        ]
        
        for key, label in display_options:
            checkbox_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1)
            checkbox_layout.add_widget(Label(text=label, font_name=font_path, size_hint_x=0.8))
            checkbox = CheckBox(size_hint_x=0.2)
            checkbox.bind(active=lambda instance, value, k=key: self.on_display_option_change(k, value))
            self.display_checkboxes[key] = checkbox
            checkbox_layout.add_widget(checkbox)
            display_layout.add_widget(checkbox_layout)
        
        display_tab.add_widget(display_layout)
        self.tab_panel.add_widget(display_tab)
    
    def create_detection_tab(self):
        """Create face detection parameters tab"""
        detection_tab = TabbedPanelItem(text="검출 설정", font_name=font_path)
        detection_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Face detection parameters
        scale_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        scale_layout.add_widget(Label(text="스케일 팩터 (%):", font_name=font_path, size_hint_x=0.5))
        self.scale_factor_input = TextInput(text="110", multiline=False, size_hint_x=0.5, font_name=font_path)
        self.scale_factor_input.bind(text=self.on_scale_factor_change)
        scale_layout.add_widget(self.scale_factor_input)
        detection_layout.add_widget(scale_layout)
        
        neighbors_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        neighbors_layout.add_widget(Label(text="최소 이웃 수:", font_name=font_path, size_hint_x=0.5))
        self.min_neighbors_input = TextInput(text="5", multiline=False, size_hint_x=0.5, font_name=font_path)
        self.min_neighbors_input.bind(text=self.on_min_neighbors_change)
        neighbors_layout.add_widget(self.min_neighbors_input)
        detection_layout.add_widget(neighbors_layout)
        
        # Position control parameters
        error_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        error_layout.add_widget(Label(text="오차 범위 (px):", font_name=font_path, size_hint_x=0.5))
        self.error_range_input = TextInput(text="50", multiline=False, size_hint_x=0.5, font_name=font_path)
        self.error_range_input.bind(text=self.on_error_range_change)
        error_layout.add_widget(self.error_range_input)
        detection_layout.add_widget(error_layout)
        
        size_min_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        size_min_layout.add_widget(Label(text="최소 크기 (px):", font_name=font_path, size_hint_x=0.5))
        self.size_min_input = TextInput(text="110", multiline=False, size_hint_x=0.5, font_name=font_path)
        self.size_min_input.bind(text=self.on_size_min_change)
        size_min_layout.add_widget(self.size_min_input)
        detection_layout.add_widget(size_min_layout)
        
        size_max_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        size_max_layout.add_widget(Label(text="최대 크기 (px):", font_name=font_path, size_hint_x=0.5))
        self.size_max_input = TextInput(text="140", multiline=False, size_hint_x=0.5, font_name=font_path)
        self.size_max_input.bind(text=self.on_size_max_change)
        size_max_layout.add_widget(self.size_max_input)
        detection_layout.add_widget(size_max_layout)
        
        detection_tab.add_widget(detection_layout)
        self.tab_panel.add_widget(detection_tab)
    
    def load_current_settings(self):
        """Load current settings into the UI"""
        # Camera settings
        self._orig_camera_index = config.get_camera_index()
        self._orig_camera_width = config.get('camera.width', 1920)
        self._orig_camera_height = config.get('camera.height', 1080)

        self.camera_index_input.text = str(self._orig_camera_index)
        self.camera_width_input.text = str(self._orig_camera_width)
        self.camera_height_input.text = str(self._orig_camera_height)
        
        # Height mapping (robust to schema changes)
        height_mapping = config.get_height_mapping()
        key_to_label = {
            "height_150": "150cm 대",
            "height_160": "160cm 대",
            "height_170": "170cm 대",
            "height_180_over": "180cm 이상",
        }
        key_default = {
            "height_150": 160,
            "height_160": 170,
            "height_170": 180,
            "height_180_over": 190,
        }
        for key, widget in self.height_inputs.items():
            label = key_to_label.get(key)
            default_val = key_default.get(key, 0)
            value = height_mapping.get(label, default_val) if label else default_val
            try:
                widget.text = str(value)
            except Exception:
                pass
        
        # Display options
        display_options = config.get_display_options()
        for key, checkbox in self.display_checkboxes.items():
            checkbox.active = display_options.get(key, True)
        
        # Detection parameters
        detection_params = config.get_face_detection_params()
        self.scale_factor_input.text = str(int(detection_params.get('scale_factor', 1.1) * 100))
        self.min_neighbors_input.text = str(detection_params.get('min_neighbors', 5))
        
        # Position control parameters
        pos_params = config.get_position_control_params()
        self.error_range_input.text = str(pos_params.get('error_range_x', 50))
        self.size_min_input.text = str(pos_params.get('correct_size_x_min', 110))
        self.size_max_input.text = str(pos_params.get('correct_size_x_max', 140))
    
    # Real-time change handlers
    def on_camera_index_change(self, instance, value):
        try:
            index = int(value)
            print(f"Camera index changed to: {index} (will apply on save)")
            # Do not write to config now; apply on Save
        except ValueError:
            print(f"Invalid camera index: {value}")
            pass
    
    def on_camera_width_change(self, instance, value):
        try:
            width = int(value)
            print(f"Camera width changed to: {width} (will apply on save)")
            # Do not write to config now; apply on Save
        except ValueError:
            pass
    
    def on_camera_height_change(self, instance, value):
        try:
            height = int(value)
            print(f"Camera height changed to: {height} (will apply on save)")
            # Do not write to config now; apply on Save
        except ValueError:
            pass
    
    
    def on_height_change(self, key, value):
        try:
            height_value = int(value)
            height_mapping = config.get_height_mapping()
            key_mapping = {
                "height_150": "150cm 대",
                "height_160": "160cm 대",
                "height_170": "170cm 대",
                "height_180_over": "180cm 이상"
            }
            height_mapping[key_mapping[key]] = height_value
            config.set_height_mapping(height_mapping)
            config.save_config()
        except ValueError:
            pass
    
    def on_display_option_change(self, key, value):
        display_options = config.get_display_options()
        display_options[key] = value
        config.set_display_options(display_options)
        config.save_config()
    
    def on_scale_factor_change(self, instance, value):
        try:
            scale_factor = int(value) / 100.0
            detection_params = config.get_face_detection_params()
            detection_params['scale_factor'] = scale_factor
            config.set_face_detection_params(detection_params)
            config.save_config()
        except ValueError:
            pass
    
    def on_min_neighbors_change(self, instance, value):
        try:
            min_neighbors = int(value)
            detection_params = config.get_face_detection_params()
            detection_params['min_neighbors'] = min_neighbors
            config.set_face_detection_params(detection_params)
            config.save_config()
        except ValueError:
            pass
    
    def on_error_range_change(self, instance, value):
        try:
            error_range = int(value)
            pos_params = config.get_position_control_params()
            pos_params['error_range_x'] = error_range
            config.set_position_control_params(pos_params)
            config.save_config()
        except ValueError:
            pass
    
    def on_size_min_change(self, instance, value):
        try:
            size_min = int(value)
            pos_params = config.get_position_control_params()
            pos_params['correct_size_x_min'] = size_min
            config.set_position_control_params(pos_params)
            config.save_config()
        except ValueError:
            pass
    
    def on_size_max_change(self, instance, value):
        try:
            size_max = int(value)
            pos_params = config.get_position_control_params()
            pos_params['correct_size_x_max'] = size_max
            config.set_position_control_params(pos_params)
            config.save_config()
        except ValueError:
            pass
    
    def save_settings(self, instance):
        """Save all settings"""
        # Check if camera settings were changed by comparing with UI values
        try:
            current_index = int(self.camera_index_input.text)
            current_width = int(self.camera_width_input.text)
            current_height = int(self.camera_height_input.text)
            
            # Compare with values loaded when popup opened
            original_index = getattr(self, '_orig_camera_index', config.get_camera_index())
            original_width = getattr(self, '_orig_camera_width', config.get('camera.width', 1920))
            original_height = getattr(self, '_orig_camera_height', config.get('camera.height', 1080))
            
            # Check if camera settings changed
            camera_changed = (current_index != original_index or 
                             current_width != original_width or 
                             current_height != original_height)
            
            if camera_changed:
                print("Camera settings changed, marking for reload...")
                config.mark_camera_changed()
                # Update the actual config values
                config.set('camera.index', current_index)
                config.set('camera.width', current_width)
                config.set('camera.height', current_height)
                # Update originals for future comparisons in this session
                self._orig_camera_index = current_index
                self._orig_camera_width = current_width
                self._orig_camera_height = current_height
        except ValueError:
            print("Invalid camera values, skipping camera change check")
        
        config.save_config()
        
        # Show confirmation (avoid Korean in Popup title)
        popup = Popup(title='', 
                     content=Label(text='설정이 저장되었습니다.', font_name=font_path),
                     size_hint=(0.6, 0.3))
        popup.open()
        Clock.schedule_once(lambda dt: popup.dismiss(), 1.0)
    
    def reset_to_defaults(self, instance):
        """Reset all settings to default values"""
        config.config = config.DEFAULT_CONFIG.copy()
        self.load_current_settings()
        config.save_config()
        
        popup = Popup(title='', 
                     content=Label(text='설정이 기본값으로 초기화되었습니다.', font_name=font_path),
                     size_hint=(0.6, 0.3))
        popup.open()
        Clock.schedule_once(lambda dt: popup.dismiss(), 1.0)
    
    def close_popup(self, instance):
        """Close the popup"""
        self.dismiss()

def show_settings_popup():
    """Show the settings popup"""
    popup = SettingsPopup()
    popup.open()
    return popup

if __name__ == "__main__":
    # For testing purposes
    from kivy.app import App
    from kivy.uix.button import Button
    
    class TestApp(App):
        def build(self):
            btn = Button(text="Open Settings")
            btn.bind(on_press=lambda x: show_settings_popup())
            return btn
    
    TestApp().run()
