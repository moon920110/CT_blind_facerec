import json
import os
from typing import Dict, Any

class Config:
    """Configuration management for the face recognition app"""
    
    CONFIG_FILE = "config.json"
    DEFAULT_CONFIG = {
        "camera": {
            "index": 0,
            "width": 1920,
            "height": 1080,
            "is_changed": False
        },
        "height_mapping": {
            "150cm 대": 400,
            "160cm 대": 300,
            "170cm 대": 200,
            "180cm 이상": 100
        },
        "display_options": {
            "draw_face_rect": True,
            "draw_standard_line": False,
            "draw_arrow": True,
            "draw_face_position": True,
            "draw_face_size": True,
            "show_height_lines": False,
            "disable_auto_navigate": False
        },
        "face_detection": {
            "scale_factor": 1.1,
            "min_neighbors": 5,
            "min_size": [20, 20]
        },
        "position_control": {
            "error_range_x": 50,
            "correct_size_x_min": 110,
            "correct_size_x_max": 140,
            "correct_timer_threshold": 1.0
        }
    }
    
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return default"""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self._merge_configs(self.DEFAULT_CONFIG, config)
            except Exception as e:
                print(f"Error loading config: {e}")
                return self.DEFAULT_CONFIG.copy()
        return self.DEFAULT_CONFIG.copy()
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge user config with defaults"""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key_path: str, default=None):
        """Get value using dot notation (e.g., 'camera.index')"""
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value):
        """Set value using dot notation (e.g., 'camera.index', 1)"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
    
    def get_camera_index(self) -> int:
        return self.get('camera.index', 0)
    
    def set_camera_index(self, index: int):
        self.set('camera.index', index)
    
    def is_camera_changed(self) -> bool:
        return self.get('camera.is_changed', False)
    
    def mark_camera_reloaded(self):
        self.set('camera.is_changed', False)
    
    def set_camera_width(self, width: int):
        self.set('camera.width', width)
    
    def set_camera_height(self, height: int):
        self.set('camera.height', height)
    
    def mark_camera_changed(self):
        """Mark camera as changed (call this when save button is pressed)"""
        self.set('camera.is_changed', True)
    
    def get_height_mapping(self) -> Dict[str, int]:
        return self.get('height_mapping', self.DEFAULT_CONFIG['height_mapping'])
    
    def set_height_mapping(self, mapping: Dict[str, int]):
        self.set('height_mapping', mapping)
    
    def get_display_options(self) -> Dict[str, bool]:
        return self.get('display_options', self.DEFAULT_CONFIG['display_options'])
    
    def set_display_options(self, options: Dict[str, bool]):
        self.set('display_options', options)
    
    def get_face_detection_params(self) -> Dict[str, Any]:
        return self.get('face_detection', self.DEFAULT_CONFIG['face_detection'])
    
    def set_face_detection_params(self, params: Dict[str, Any]):
        self.set('face_detection', params)
    
    def get_position_control_params(self) -> Dict[str, Any]:
        return self.get('position_control', self.DEFAULT_CONFIG['position_control'])
    
    def set_position_control_params(self, params: Dict[str, Any]):
        self.set('position_control', params)

# Global config instance
config = Config()
