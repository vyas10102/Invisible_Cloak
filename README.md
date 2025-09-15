# Invisibility Cloak — OpenCV

This project creates an “invisibility cloak” effect using real-time background substitution with OpenCV.  
The script captures a clean background, segments a chosen cloak color in HSV color space, cleans up the mask with morphology, optionally smooths the mask over time, and composites the background where the cloak appears.

---

## Features

- **Live background capture** (no pre-recorded video required)
- **Configurable cloak color**: choose any color by adjusting HSV lower/upper bounds
- **HSV tuner** (interactive sliders) for quick range tweaking
- **Auto-calibration** from the current frame (locks to cloak’s median HSV)
- **Temporal mask smoothing** (EMA) to reduce flicker
- **Robust mask cleanup** (open → close → median blur → dilate)
- **On-screen feedback**: optional mask window + “Cloak” contour boxes

---

## Demo

<video src="demo.mp4" controls width="600"></video>

---

## Requirements

- Python 3.8+
- A webcam
- Packages:
  ```bash
  # (Recommended) create & activate a venv
  python -m venv .venv
  # Windows
  .venv\Scripts\activate
  # macOS/Linux
  source .venv/bin/activate

  pip install opencv-python numpy

  # Run
  python invisibility_cloak3.py
  ```

---

## How to Use

1. **Start the script**  
   A preview window will open.

2. **Capture the background**  
   Step out of the frame. The script collects ~60 frames to build a stable background.

3. **Use the cloak**  
   Step back in with your chosen cloak color. The cloak region will now be replaced with the captured background.

4. **Tweak settings (optional):**  
   - Press **t** → Open the HSV tuner  
   - Press **c** → Auto-calibrate from the current frame  
   - Press **s** → Toggle temporal smoothing  
   - Press **m** → View the binary mask  

You can re-capture the background anytime by pressing **r**.

---

## Hotkeys

| Key         | Action                                |
|-------------|---------------------------------------|
| **q / ESC** | Quit the program                      |
| **r**       | Re-capture background                 |
| **m**       | Toggle mask window                    |
| **t**       | Toggle HSV tuner                      |
| **s**       | Toggle temporal smoothing             |
| **c**       | Auto-calibrate HSV from current frame |

---

## Default Settings

- **Frame size:** 640 × 480  
- **Brightness:** 150  
- **Background frames:** ~60  
- **Default cloak color (HSV):**  
  - Lower: `[35, 80, 40]`  
  - Upper: `[85, 255, 255]`  
- **Morphological kernels:**  
  - Open: 3 × 3 ellipse  
  - Close: 7 × 7 ellipse  
  - Dilate: 5 × 5 ellipse  

---

## How it Works

1. **Background capture** → mirror-flipped frames averaged into a clean background.
2. **Preprocessing** → live frame → flip → HSV → Gaussian blur.
3. **Masking** → `cv2.inRange(lower, upper)` selects cloak color.
4. **Morphology** → open → close → median blur → dilate to refine mask.
5. **Temporal smoothing** → optional exponential moving average on mask.
6. **Compositing** → cloak regions replaced with background; rest of frame stays live.
7. **Feedback** → contour outlines drawn around cloak.

---

## Troubleshooting

- ❌ Cloak not fully invisible? → Use `t` to widen HSV range.
- 👻 Ghosting / blur? → Re-capture background with `r`.
- ✨ Flicker? → Toggle smoothing with `s`.
- 📐 Edges rough? → Adjust morphology kernels at top of file.

---

## License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute this software, provided that the original copyright notice and this permission notice are included in all copies or substantial portions of the software.
