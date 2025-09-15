import cv2
import numpy as np
import time

# ---------------- Camera & defaults ----------------
CAM_INDEX = 0
W, H = 640, 480
BRIGHTNESS = 150

# Green band in OpenCV HSV (H in [0..179])
# Start wide; we'll allow tuning and calibration.
DEFAULTS = {
    "LOWER_GREEN": np.array([30, 60, 30], dtype=np.uint8),
    "UPPER_GREEN": np.array([95, 255, 255], dtype=np.uint8),
}
KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
KERNEL_DIL   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Temporal smoothing for mask (press 's' to toggle)
SMOOTH_ALPHA = 0.25  # 0..1, higher = more responsive, lower = more stable

def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(3, W)
    cap.set(4, H)
    cap.set(10, BRIGHTNESS)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open camera at index {index}. "
            "Close other apps using the webcam or try CAM_INDEX=1/2."
        )
    return cap

def capture_background(cap: cv2.VideoCapture, frames: int = 60) -> np.ndarray:
    """Capture a clean background. Step out of frame while this runs."""
    print("Capturing background... Please step OUT of the frame.")
    bg = None
    for i in range(frames):
        ok, frame = cap.read()
        if not ok:
            continue
        bg = cv2.flip(frame, 1)  # mirror to match live frames
        if i % 15 == 0:
            preview = bg.copy()
            cv2.putText(preview, f"Capturing BG {i+1}/{frames}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Background Preview", preview)
            cv2.waitKey(1)
        time.sleep(0.01)
    try:
        cv2.destroyWindow("Background Preview")
    except:
        pass
    if bg is None:
        raise RuntimeError("Failed to capture background. Is the camera working?")
    print("Background captured. Step IN with the green cloth.")
    return bg

# ---------------- HSV tuner (optional) ----------------
def create_tuner():
    def nothing(_): pass
    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("H_LOW",  "HSV Tuner", 30, 179, nothing)
    cv2.createTrackbar("H_HIGH", "HSV Tuner", 95, 179, nothing)
    cv2.createTrackbar("S_MIN",  "HSV Tuner", 60, 255, nothing)
    cv2.createTrackbar("V_MIN",  "HSV Tuner", 30, 255, nothing)

def read_tuner():
    h_low  = cv2.getTrackbarPos("H_LOW",  "HSV Tuner")
    h_high = cv2.getTrackbarPos("H_HIGH", "HSV Tuner")
    s_min  = cv2.getTrackbarPos("S_MIN",  "HSV Tuner")
    v_min  = cv2.getTrackbarPos("V_MIN",  "HSV Tuner")
    h_low, h_high = min(h_low, h_high), max(h_low, h_high)
    lower = np.array([h_low,  s_min, v_min], dtype=np.uint8)
    upper = np.array([h_high, 255,   255],  dtype=np.uint8)
    return lower, upper

# ---------------- Mask builder ----------------
def build_color_mask(hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    # Initial binary mask
    mask = cv2.inRange(hsv, lower, upper)

    # Stronger cleanup: open (remove dots) -> close (fill holes) -> median -> dilate (grow edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL_OPEN,  iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE, iterations=2)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.dilate(mask, KERNEL_DIL, iterations=1)

    return mask

# ---------------- Auto calibration ----------------
def calibrate_from_frame(hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Find largest green-ish area using a broad seed range, then set tighter HSV bounds
    around its median color. Return (lower, upper) or None if not found.
    """
    seed_lower = np.array([25, 40, 20], dtype=np.uint8)
    seed_upper = np.array([100, 255, 255], dtype=np.uint8)
    seed = cv2.inRange(hsv, seed_lower, seed_upper)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, KERNEL_OPEN, iterations=1)

    cnts, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 500:  # too small
        return None

    mask_roi = np.zeros(seed.shape, dtype=np.uint8)
    cv2.drawContours(mask_roi, [cnt], -1, 255, thickness=cv2.FILLED)

    # median HSV inside contour for robustness
    pts = hsv[mask_roi == 255].reshape(-1, 3)
    if pts.size == 0:
        return None
    med = np.median(pts, axis=0).astype(np.int32)  # (H, S, V)

    # margins around median
    dH, dS, dV = 15, 60, 60
    lower = np.array([np.clip(med[0]-dH, 0, 179),
                      np.clip(med[1]-dS, 0, 255),
                      np.clip(med[2]-dV, 0, 255)], dtype=np.uint8)
    upper = np.array([np.clip(med[0]+dH, 0, 179),
                      255, 255], dtype=np.uint8)
    return lower, upper

def main():
    print("OpenCV:", cv2.__version__)
    cap = open_camera(CAM_INDEX)
    time.sleep(1)  # warm-up

    background = capture_background(cap)
    show_mask = False
    tuner_on = False
    smooth_on = False
    mask_acc = None  # for temporal smoothing

    lower = DEFAULTS["LOWER_GREEN"].copy()
    upper = DEFAULTS["UPPER_GREEN"].copy()

    print("Hotkeys: q/ESC=quit | r=recapture BG | m=toggle mask | t=HSV tuner | s=toggle smoothing | c=auto-calibrate")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame read failed / stream ended.")
                break

            frame = cv2.flip(frame, 1)
            # Convert then smooth a bit (helps uneven texture/lighting)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.GaussianBlur(hsv, (7, 7), 0)

            if tuner_on:
                lower, upper = read_tuner()

            mask = build_color_mask(hsv, lower, upper)

            # Temporal smoothing (EMA)
            if smooth_on:
                if mask_acc is None:
                    mask_acc = mask.astype(np.float32)
                else:
                    mask_acc = (1.0 - SMOOTH_ALPHA) * mask_acc + SMOOTH_ALPHA * mask.astype(np.float32)
                mask = (mask_acc > 127).astype(np.uint8) * 255

            mask_inv = cv2.bitwise_not(mask)

            cloak_area = cv2.bitwise_and(background, background, mask=mask)
            live_area  = cv2.bitwise_and(frame, frame, mask=mask_inv)
            final = cv2.addWeighted(cloak_area, 1, live_area, 1, 0)

            # Optional: draw contours for visual feedback
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 800:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(final, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(final, "Cloak", (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Invisibility Cloak (Green, Robust)", final)
            if show_mask:
                cv2.imshow("Mask (white = cloak)", mask)
            else:
                try:
                    if cv2.getWindowProperty("Mask (white = cloak)", cv2.WND_PROP_VISIBLE) >= 1:
                        cv2.destroyWindow("Mask (white = cloak)")
                except:
                    pass

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):     # q or ESC
                break
            elif key == ord('r'):         # recapture background
                background = capture_background(cap)
            elif key == ord('m'):         # toggle mask window
                show_mask = not show_mask
            elif key == ord('t'):         # toggle HSV tuner
                tuner_on = not tuner_on
                if tuner_on:
                    create_tuner()
                else:
                    try:
                        cv2.destroyWindow("HSV Tuner")
                    except:
                        pass
            elif key == ord('s'):         # toggle temporal smoothing
                smooth_on = not smooth_on
                if not smooth_on:
                    mask_acc = None
                print("Smoothing:", "ON" if smooth_on else "OFF")
            elif key == ord('c'):         # auto-calibrate from current frame
                res = calibrate_from_frame(hsv)
                if res is not None:
                    lower, upper = res
                    print("Calibrated HSV:",
                          f"LOWER={lower.tolist()} UPPER={upper.tolist()}")
                else:
                    print("Calibration failed. Ensure the cloak is clearly visible.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped. Camera released.")

if __name__ == "__main__":
    main()
