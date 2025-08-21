## CT_blind_facerec (Quick Setup)

### Requirements
- Python 3.11
- Install dependencies: `python -m pip install -r requirements.txt`

### FairFace age model (required)
- Download the FairFace multi-task checkpoint and place it here:
  - `model/res34_fair_align_multi_7_20190809.pt`
- Reference: [FairFace GitHub](https://github.com/dchen236/FairFace)

This project uses FairFace for age only. Emotion/Gender/Race are inferred via DeepFace.

### Run
- UI (Kivy): `python main.py`

### Notes
- Camera will run in portrait (9:16) layout.
- Age: FairFace bins + estimated age (from probabilities)
- Emotion/Gender/Race: DeepFace

### What you can change (config knobs)
- Model file path and age labels: `utils/analysis.py`
  - `_WEIGHTS_PATH`: change if your FairFace checkpoint is elsewhere (default: `model/res34_fair_align_multi_7_20190809.pt`)
  - `_AGE_LABELS_9`: Korean labels for age bins; edit to your desired display
  - `_HEIGHT_BY_Y`: rough height mapping by face Y; tune per camera setup
- Camera thresholds and behavior: `screen/MainTaskScreen.py`
  - `_init_camera_params()`: base thresholds scaled to camera width
    - `_ERROR_RANGE_X` (left/right tolerance)
    - `_CORRECT_SIZE_X_MIN`, `_CORRECT_SIZE_X_MAX` (acceptable face width range)
  - `_SMOOTH_ALPHA`: rectangle smoothing speed (0.1–0.3 typical)
  - Text overlay: label `font_size`, `pos_hint['y']` to move/size on screen
  - Aspect crop: `crop_frame(ratio=9/16)` if you need a different target aspect
  - Drawing toggles at top of file:
    - `_DRAW_FACE_RECT` (draw face rectangle)
    - `_DRAW_STANDARD_LINE` (draw center/guard lines)
    - `_DRAW_ARROW` (direction arrows)
    - `_DRAW_FACE_POSITION` (debug text for position)
    - `_DRAW_FACE_SIZE` (debug text for size)
- TTS behavior: `utils/speak.py`
  - `speak(text)` defaults to forced playback (no overlap). Adjust if needed

After changing code, simply rerun `python main.py`.

