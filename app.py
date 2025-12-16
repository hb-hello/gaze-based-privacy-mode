import sys
import ctypes
import tkinter as tk
from tkinter import messagebox, font
import logging
import time
import os

# Optional imports for camera functionality
try:
    import cv2
    from PIL import Image, ImageTk
except Exception:
    cv2 = None
    Image = None
    ImageTk = None

# Logging setup
LOGFILE = "app_debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Redirect stdout/stderr to logger so prints in gaze_detect are captured
class _StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, msg):
        msg = msg.rstrip()
        if msg:
            self.logger.log(self.level, msg)

    def flush(self):
        pass

sys.stdout = _StreamToLogger(logger, logging.INFO)
sys.stderr = _StreamToLogger(logger, logging.ERROR)

# Theme colors and fonts
DEFAULT_BG = "#f5f5f5"  # light grey
DEFAULT_FG = "#333333"  # dark grey
BUTTON_BG = "#ffffff"

# Optional import for gaze attention
try:
    from gaze_detect import getGazeAttention, are_there_multiple_faces
    logger.debug("Imported getGazeAttention and are_there_multiple_faces from gaze_detect")
except Exception:
    getGazeAttention = None
    are_there_multiple_faces = None
    logger.exception("Failed to import gaze detection helpers; gaze features disabled")


def lock_workstation():
    """Lock the workstation on supported Windows platforms."""
    if not sys.platform.startswith("win") or not hasattr(ctypes, "windll"):
        messagebox.showerror("Unsupported", "Locking the workstation is only supported on Windows.")
        return

    confirm = messagebox.askyesno(
        "Privacy mode",
        "This will lock your workstation and return to the lock screen. Continue?",
    )
    if not confirm:
        return

    try:
        ctypes.windll.user32.LockWorkStation()
    except Exception as exc:
        messagebox.showerror("Error", f"Failed to lock workstation:\n{exc}")


class PrivacyApp:
    def __init__(self, root):
        self.root = root
        root.title("Privacy Mode")
        # Apply theme
        root.configure(bg=DEFAULT_BG)

        # Window size and center it on screen (reduced height to fit compact layout)
        w, h = 640, 380
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws // 2) - (w // 2)
        y = (hs // 2) - (h // 2)
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.resizable(False, False)

        self.cap = None
        self.running = False
        self.frame_counter = 0
        self.gaze_interval = 10 # call gaze detection every N frames
        self.last_attention = None
        # FPS tracking
        self.fps = 0.0
        self._last_time = time.time()
        # lock-status tracking
        self.low_threshold = 40.0  # percent
        self.required_consecutive = 2
        self.consecutive_low_count = 0

        # Frame layout
        frame = tk.Frame(root, bg=DEFAULT_BG)
        frame.pack(expand=True, pady=10)

        # Header with centered app icon (in a circle) above the label
        header_font = font.Font(family="Segoe UI", size=16, weight="bold")
        header_frame = tk.Frame(frame, bg=DEFAULT_BG)
        header_frame.pack(pady=(0, 8))

        # Load app icon from assets/user.png
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        user_icon_path = os.path.join(assets_dir, "user.png")
        self.user_icon = None
        try:
            if Image is not None and ImageTk is not None:
                img = Image.open(user_icon_path).convert("RGBA")
                img = img.resize((64, 64), Image.LANCZOS)
                self.user_icon = ImageTk.PhotoImage(img)
            else:
                self.user_icon = tk.PhotoImage(file=user_icon_path)
        except Exception:
            logger.exception("Failed to load user icon from %s", user_icon_path)
            self.user_icon = None

        # Avatar canvas with subtle shadow and circular background
        avatar_size = 96
        avatar_canvas = tk.Canvas(header_frame, width=avatar_size, height=avatar_size, bg=DEFAULT_BG, highlightthickness=0)
        # Avatar canvas: centered circular background with user icon
        margin = 6
        avatar_canvas.create_oval(margin, margin, avatar_size - margin, avatar_size - margin, fill="#d9d9d9", outline="")
        # place the user icon centered inside the circle
        if self.user_icon is not None:
            avatar_canvas.create_image(avatar_size / 2, avatar_size / 2, image=self.user_icon)
        # pack centered
        avatar_canvas.pack(anchor="center")

        label = tk.Label(header_frame, text="Enter privacy mode", font=header_font, fg=DEFAULT_FG, bg=DEFAULT_BG)
        label.pack(pady=(6, 0))

        # Attention display (text only) - shown above the camera button
        self.video_width = 600
        self.video_height = 400

        self.attention_var = tk.StringVar(value="Attention: N/A")
        self.attention_label = tk.Label(frame, textvariable=self.attention_var, font=font.Font(family="Segoe UI", size=14), fg=DEFAULT_FG, bg=DEFAULT_BG)
        self.attention_label.pack(pady=(2, 6))

        # Camera start/stop button with icon
        # load eye-test icon (slightly larger for the bigger button)
        eye_icon_path = os.path.join(assets_dir, "eye-test.png")
        self.eye_icon = None
        try:
            if Image is not None and ImageTk is not None:
                img = Image.open(eye_icon_path).convert("RGBA")
                img = img.resize((26, 26), Image.LANCZOS)
                self.eye_icon = ImageTk.PhotoImage(img)
            else:
                self.eye_icon = tk.PhotoImage(file=eye_icon_path)
        except Exception:
            logger.exception("Failed to load eye icon from %s", eye_icon_path)
            self.eye_icon = None

        # Larger bold font and padding to approximate a 100x40 px button
        btn_font = font.Font(family="Segoe UI", size=14, weight="bold")
        # Use increased internal padding (padx/pady) so the button appears ~100x40 pixels
        # load camera icon (fallback to eye-test if camera asset isn't present)
        camera_icon_path = os.path.join(assets_dir, "camera.png")
        eye_icon_path = os.path.join(assets_dir, "eye-test.png")
        if not os.path.exists(camera_icon_path):
            camera_icon_path = eye_icon_path

        self.cam_icon = None
        try:
            if Image is not None and ImageTk is not None:
                img = Image.open(camera_icon_path).convert("RGBA")
                img = img.resize((36, 36), Image.LANCZOS)
                self.cam_icon = ImageTk.PhotoImage(img)
            else:
                self.cam_icon = tk.PhotoImage(file=camera_icon_path)
        except Exception:
            logger.exception("Failed to load camera icon from %s", camera_icon_path)
            self.cam_icon = None

        # Larger bold font and padding to approximate a 100x40 px button; grey background
        btn_font = font.Font(family="Segoe UI", size=14, weight="bold")
        self.camera_button = tk.Button(
            frame,
            text=" Start Camera",
            command=self.toggle_camera,
            bg="#d9d9d9",
            fg=DEFAULT_FG,
            compound="left",
            font=btn_font,
            bd=0,
            relief="raised",
            padx=36,
            pady=8,
        )
        if self.cam_icon is not None:
            self.camera_button.config(image=self.cam_icon)
        self.camera_button.pack(pady=(12, 0))

        # Bottom status label: Screen locked/unlocked (with icon)
        padlock_path = os.path.join(assets_dir, "padlock.png")
        unlocked_path = os.path.join(assets_dir, "unlocked.png")
        self.padlock_icon = None
        self.unlocked_icon = None
        try:
            # make the padlock/unlocked icons slightly larger so they read clearly
            if Image is not None and ImageTk is not None:
                img = Image.open(padlock_path).convert("RGBA").resize((28, 28), Image.LANCZOS)
                self.padlock_icon = ImageTk.PhotoImage(img)
                img2 = Image.open(unlocked_path).convert("RGBA").resize((28, 28), Image.LANCZOS)
                self.unlocked_icon = ImageTk.PhotoImage(img2)
            else:
                self.padlock_icon = tk.PhotoImage(file=padlock_path)
                self.unlocked_icon = tk.PhotoImage(file=unlocked_path)
        except Exception:
            logger.exception("Failed to load padlock/unlocked icons from assets")

        self.status_var = tk.StringVar(value="Screen unlocked")
        # increase the status text size so it matches the larger icon
        self.status_label = tk.Label(frame, textvariable=self.status_var, font=font.Font(family="Segoe UI", size=14), fg=DEFAULT_FG, bg=DEFAULT_BG, compound="left")
        # default to unlocked icon if available
        if self.unlocked_icon is not None:
            self.status_label.config(image=self.unlocked_icon)
        self.status_label.pack(pady=(8, 0))

        # Lock button (optional quick-lock)
        # self.lock_button = tk.Button(frame, text="Lock Workstation", command=lock_workstation)
        # self.lock_button.pack(pady=(6, 0))

        # Bindings
        root.bind("<Escape>", lambda e: self.close())
        root.bind("<Return>", lambda e: self.toggle_camera())

        root.protocol("WM_DELETE_WINDOW", self.close)

    def toggle_camera(self):
        if not self.running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        if cv2 is None or Image is None or ImageTk is None:
            logger.error("Missing dependency: cv2 or PIL not available")
            messagebox.showerror(
                "Missing dependency",
                "OpenCV and Pillow are required to run the camera.\nInstall with: pip install opencv-python pillow",
            )
            return

        # Open default camera
        logger.debug("Attempting to open camera (frame %d)", self.frame_counter)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            logger.exception("Camera open failed; cap=%r", self.cap)
            messagebox.showerror("Camera error", "Could not open camera. Make sure it's connected and not used by another app.")
            self.cap = None
            return

        logger.info("Camera opened successfully")

        # Try to set a reasonable resolution; backend may ignore it
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)

        self.running = True
        self.camera_button.config(text="Stop Camera")
        self.update_frame()

    def stop_camera(self):
        self.running = False
        self.camera_button.config(text=" Start Camera")
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        # Reset attention display
        self.attention_var.set("Attention: N/A")
        # Reset lock status
        self.consecutive_low_count = 0
        self.status_var.set("Screen unlocked")
        try:
            if self.unlocked_icon is not None:
                self.status_label.config(image=self.unlocked_icon)
        except Exception:
            pass

    def update_frame(self):
        if not self.running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            # Try again shortly; sometimes the first frames fail
            self.root.after(50, self.update_frame)
            return
        # Resize frame for processing (we do not display it)
        try:
            frame_proc = cv2.resize(frame, (self.video_width, self.video_height))
        except Exception:
            frame_proc = frame

        # Update face-count status for this frame (if available)
        # if are_there_multiple_faces is not None:
        #     try:
        #         multiple = are_there_multiple_faces(frame_proc)
        #         if multiple:
        #             self.faces_var.set("Multiple faces detected")
        #         else:
        #             self.faces_var.set("Single face detected")
        #     except Exception:
        #         logger.exception("are_there_multiple_faces failed on frame %d", self.frame_counter)

        # Occasionally run the gaze attention detector (expensive)
        if getGazeAttention is None:
            if self.frame_counter == 0:
                logger.warning("Gaze detection unavailable (getGazeAttention is None)")
        elif (self.frame_counter % self.gaze_interval == 0):
            try:
                logger.debug("Calling getGazeAttention frame=%d", self.frame_counter)
                result = getGazeAttention(frame_proc.copy(), self.frame_counter)
                logger.debug("getGazeAttention returned: %r", result)

                # Support both legacy single-value return and new (attention, multiple_faces)
                one_face = None
                if isinstance(result, (list, tuple)) and len(result) >= 2:
                    attention_val, one_face = result[0], result[1]
                else:
                    attention_val = result

                try:
                    self.last_attention = float(attention_val)
                    if (one_face is not None and one_face):
                        self.last_attention = max(attention_val, 50.0)
                except Exception:
                    logger.exception("Failed to parse attention value: %r", attention_val)
                    self.last_attention = None

                # Update faces label if we received face-count info
                # if multiple_faces is not None:
                #     try:
                #         if multiple_faces:
                #             self.faces_var.set("Multiple faces detected")
                #         else:
                #             self.faces_var.set("Single face detected")
                #     except Exception:
                #         logger.exception("Failed updating faces label")

            except Exception:
                logger.exception("getGazeAttention failed on frame %d", self.frame_counter)

            # Update consecutive-low counter and status when detection runs
            try:
                if self.last_attention is not None and self.last_attention < self.low_threshold:
                    self.consecutive_low_count += 1
                else:
                    self.consecutive_low_count = 0

                if self.consecutive_low_count >= self.required_consecutive:
                    self.status_var.set("Screen locked")
                    try:
                        if self.padlock_icon is not None:
                            self.status_label.config(image=self.padlock_icon)
                    except Exception:
                        pass
                else:
                    self.status_var.set("Screen unlocked")
                    try:
                        if self.unlocked_icon is not None:
                            self.status_label.config(image=self.unlocked_icon)
                    except Exception:
                        pass
            except Exception:
                logger.exception("Failed updating lock-status tracking")

        self.frame_counter += 1

        # Compute instantaneous FPS and smooth it
        now = time.time()
        delta = now - self._last_time if self._last_time else 0.0
        if delta > 0:
            inst_fps = 1.0 / delta
            # exponential moving average
            self.fps = (0.85 * self.fps) + (0.15 * inst_fps)
        self._last_time = now

        # Update attention label text (show only attention value)
        if self.last_attention is not None:
            self.attention_var.set(f"Attention: {self.last_attention:.1f}%")
        else:
            self.attention_var.set("Attention: N/A")

        # Schedule next frame
        self.root.after(15, self.update_frame)

    def close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = PrivacyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
