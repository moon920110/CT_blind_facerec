import os
import threading
import time

import playsound
from gtts import gTTS

_audio_dir = 'tmp_audio'
os.makedirs(_audio_dir, exist_ok=True)
_speaking = threading.Event()

def speak(text: str, force=False):
    # 이미 재생 중이면 스킵
    if _speaking.is_set():
        return
    _speaking.set()
    threading.Thread(target=_synthesize_play_cleanup, args=(text,), daemon=True).start()

def _synthesize_play_cleanup(text: str):
    filename = os.path.join(_audio_dir, f'voice_{int(time.time()*1000)}.mp3')
    try:
        tts = gTTS(text=text, lang='ko')
        tts.save(filename)
        playsound.playsound(filename, block=True)
    except Exception as e:
        print(f'Audio pipeline failed: {e}')
    finally:
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            print(f'Failed to remove {filename}: {e}')
        _speaking.clear()