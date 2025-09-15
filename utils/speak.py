import os
import re
import threading
import time

import playsound
from gtts import gTTS
from kivy.clock import Clock
from kivy.core.audio import SoundLoader

_audio_dir = 'audio'
os.makedirs(_audio_dir, exist_ok=True)
_speaking = threading.Event()
_lock = threading.Lock()
_current_sound = None

def speak(text: str, force: bool = False):
    # 재생 중 처리
    with _lock:
        if _speaking.is_set():
            if not force:
                return
            # 강제 중단 요청: 현재 사운드가 있으면 정지
            if _current_sound is not None:
                def _stop_current(_dt):
                    try:
                        _current_sound.stop()
                    except Exception:
                        pass
                Clock.schedule_once(_stop_current, 0)
            _speaking.clear()
    _speaking.set()
    # 재생 전 0.5초 리드(여백) 추가. force일 땐 약간 더 대기
    delay = 0.5 + (0.1 if force else 0.0)
    threading.Thread(target=_synthesize_play_cleanup, args=(text, delay), daemon=True).start()

def _safe_filename_from_text(text: str) -> str:
    # Replace newlines with space, trim
    base = text.replace('\n', ' ').strip()
    # Collapse whitespace to single underscore
    base = re.sub(r'\s+', '_', base)
    # Remove characters invalid on Windows file systems <>:"/\|?*
    base = re.sub(r'[<>:"/\\|?*]', '', base)
    # Replace any other punctuation/non-word chars with underscore to avoid MCI parsing issues
    base = re.sub(r'[^\w\-가-힣]', '_', base)
    # Collapse multiple underscores
    base = re.sub(r'_+', '_', base)
    # Strip leading/trailing dots/underscores/spaces
    base = base.strip(' ._')
    # Ensure no trailing period
    base = re.sub(r'\.+$', '', base)
    # Limit length to avoid OS issues
    if len(base) > 80:
        base = base[:80]
    if not base:
        base = f'voice_{int(time.time()*1000)}'
    return f"{base}.mp3"


def _synthesize_play_cleanup(text: str, play_delay: float = 0.0):
    # Determine a stable filename based on the text
    stable_name = _safe_filename_from_text(text)
    filename = os.path.join(_audio_dir, stable_name)
    try:
        if not os.path.exists(filename):
            # Synthesize and save for reuse
            tts = gTTS(text=text, lang='ko')
            tts.save(filename)
        # Prefer Kivy SoundLoader (non-blocking, fewer Windows MCI issues)
        abs_path = os.path.abspath(filename)

        def _play_on_main_thread(_dt):
            global _current_sound
            sound = SoundLoader.load(abs_path)
            if sound is not None:
                _current_sound = sound
                def _on_stop(*_):
                    global _current_sound
                    _current_sound = None
                    _speaking.clear()
                sound.bind(on_stop=_on_stop)
                sound.play()
            else:
                # Fallback to playsound (blocking)
                try:
                    playsound.playsound(abs_path, block=True)
                finally:
                    _speaking.clear()

        Clock.schedule_once(_play_on_main_thread, play_delay)
        return
    except Exception as e:
        print(f'Audio pipeline failed: {e}')
        _speaking.clear()


def stop_speaking():
    """Stop current TTS playback immediately if any."""
    with _lock:
        target = _current_sound
        if target is not None:
            def _stop_now(_dt):
                global _current_sound
                try:
                    # Only stop if the same sound is still current (avoid stopping newly started sound)
                    if _current_sound is target:
                        _current_sound.stop()
                        _current_sound = None
                        _speaking.clear()
                except Exception:
                    pass
            Clock.schedule_once(_stop_now, 0)
        else:
            _speaking.clear()