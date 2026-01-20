import os
import cv2
import shutil
import threading
import tempfile
import time
import tkinter as tk
from collections import deque
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

# Drag & drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False

# Link downloader (TikTok etc.)
try:
    from yt_dlp import YoutubeDL
    YTDLP_AVAILABLE = True
except Exception:
    YoutubeDL = None
    YTDLP_AVAILABLE = False


SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv")


def clean_dnd_path(p: str) -> str:
    p = p.strip()
    if p.startswith("{") and p.endswith("}"):
        p = p[1:-1]
    return p


def get_downloads_folder() -> str:
    home = os.path.expanduser("~")
    downloads = os.path.join(home, "Downloads")
    if os.path.isdir(downloads):
        return downloads

    onedrive = os.environ.get("OneDrive")
    if onedrive:
        od_downloads = os.path.join(onedrive, "Downloads")
        if os.path.isdir(od_downloads):
            return od_downloads

    return home


class FrameQualityEvaluator:
    """
    No-reference heuristics for filtering low-quality frames:
    - blur_score: variance of Laplacian (sharpness proxy)
    - brightness: mean grayscale
    - contrast: std grayscale
    - motion_hint: directional gradient energy ratio
    """

    def __init__(self):
        self.blur_threshold = 80.0
        self.brightness_min = 20.0
        self.contrast_min = 12.0
        self.motion_ratio_threshold = 3.5

    def set_thresholds(self, blur=None, bright=None, contrast=None, motion_ratio=None):
        if blur is not None:
            self.blur_threshold = float(blur)
        if bright is not None:
            self.brightness_min = float(bright)
        if contrast is not None:
            self.contrast_min = float(contrast)
        if motion_ratio is not None:
            self.motion_ratio_threshold = float(motion_ratio)

    def evaluate(self, bgr_frame):
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Downscale for speed/stability
        scale = 360 / max(h, w) if max(h, w) > 360 else 1.0
        if scale < 1.0:
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(lap.var())

        brightness = float(gray.mean())
        contrast = float(gray.std())

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        ex = float((gx * gx).mean())
        ey = float((gy * gy).mean())

        if min(ex, ey) < 1e-9:
            motion_ratio = 1.0
        else:
            motion_ratio = max(ex, ey) / max(min(ex, ey), 1e-9)

        reasons = []
        if brightness < self.brightness_min:
            reasons.append("Too Dark")
        if contrast < self.contrast_min:
            reasons.append("Low Contrast")

        if blur_score < self.blur_threshold:
            if motion_ratio >= self.motion_ratio_threshold:
                reasons.append("Motion Blur")
            else:
                reasons.append("Blurry")

        is_bad = len(reasons) > 0
        metrics = {
            "blur_score": blur_score,
            "brightness": brightness,
            "contrast": contrast,
            "motion_ratio": motion_ratio,
        }
        return is_bad, reasons, metrics


class VideoFrameExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Frame Extractor (Single View + Gallery View)")
        self.root.geometry("1300x880")

        # Video state
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.fps = 0.0
        self.duration_sec = 0.0
        self.current_frame_index = 0
        self.current_bgr = None
        self.preview_image = None

        # Link download / cleanup
        self.temp_dir = tempfile.mkdtemp(prefix="video_frame_extractor_")
        self.current_is_temp = False
        self.current_temp_path = None
        self.is_downloading = False

        # Quality filtering
        self.qeval = FrameQualityEvaluator()
        self.quality_cache = {}   # idx -> (is_bad, reasons, metrics)
        self.good_frames = None   # built by scan

        # Prevent slider recursion
        self._slider_updating = False

        # Playback state
        self.playing = False
        self._play_after_id = None

        # Screenshot anti-duplicate
        self._last_screenshot_time = 0.0
        self._last_screenshot_frame = None
        self._screenshot_debounce_sec = 0.25

        # Gallery state
        self.gallery_mode = False
        self.gallery_building = False
        self.gallery_stop_flag = False
        self.gallery_queue = deque()  # deque of (frame_idx, pil_thumb)
        self.gallery_thumb_refs = []  # keep references to PhotoImage
        self.gallery_good_indices = []  # indices shown in gallery
        self.gallery_menu = None
        self._gallery_menu_frame_idx = None

        self._build_ui()
        self._set_status("Drop a video file, open one, or paste a link. Use Toggle View for Gallery.")

        # Screenshot bind: Button-3 ONLY on preview canvas (single view)
        self._bind_button3_screenshot()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- UI ----------------

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        self.open_btn = ttk.Button(top, text="Open Video", command=self.open_video_dialog)
        self.open_btn.pack(side="left")

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(top, text="Video link:").pack(side="left", padx=(0, 6))

        self.link_entry = ttk.Entry(top, width=46)
        self.link_entry.pack(side="left")
        self.link_entry.bind("<Return>", lambda e: self.load_from_link_clicked())

        self.link_btn = ttk.Button(top, text="Load from Link", command=self.load_from_link_clicked)
        self.link_btn.pack(side="left", padx=(8, 0))

        self.insecure_ssl_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Allow insecure SSL", variable=self.insecure_ssl_var).pack(side="left", padx=(10, 0))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)

        self.toggle_view_btn = ttk.Button(top, text="Gallery View", command=self.toggle_view, state="disabled")
        self.toggle_view_btn.pack(side="left")

        self.play_btn = ttk.Button(top, text="Play", command=self.toggle_play, state="disabled")
        self.play_btn.pack(side="left", padx=(10, 0))

        self.export_btn = ttk.Button(top, text="Export Current Frame", command=self.export_current_frame, state="disabled")
        self.export_btn.pack(side="left", padx=(10, 0))

        self.screenshot_btn = ttk.Button(top, text="Save Screenshot (Downloads)", command=self.save_screenshot_to_downloads, state="disabled")
        self.screenshot_btn.pack(side="left", padx=(10, 0))

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=10)

        self.prev_btn = ttk.Button(top, text="◀ Prev", command=lambda: self.step_frame(-1), state="disabled")
        self.prev_btn.pack(side="left")

        self.next_btn = ttk.Button(top, text="Next ▶", command=lambda: self.step_frame(+1), state="disabled")
        self.next_btn.pack(side="left", padx=(10, 0))

        ttk.Label(top, text="Jump:").pack(side="left", padx=(15, 5))
        self.jump_entry = ttk.Entry(top, width=8)
        self.jump_entry.pack(side="left")
        self.jump_entry.bind("<Return>", lambda e: self.jump_to_frame())

        self.jump_btn = ttk.Button(top, text="Go", command=self.jump_to_frame, state="disabled")
        self.jump_btn.pack(side="left", padx=(5, 0))

        main = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        main.pack(fill="both", expand=True)

        # Left panel
        self.left = ttk.Frame(main)
        self.left.pack(side="left", fill="both", expand=True)

        self.drop_zone = tk.Label(
            self.left,
            text="Drag & drop video here\n(or use Open Video)\n\n<Button-3> saves screenshot to Downloads (Single View)",
            relief="groove",
            bd=2,
            padx=10,
            pady=10,
            font=("Helvetica", 14),
            justify="center"
        )
        self.drop_zone.pack(fill="x", pady=(0, 10))

        if DND_AVAILABLE and hasattr(self.root, "drop_target_register"):
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.dnd_bind("<<Drop>>", self.on_drop_file)
        else:
            self.drop_zone.configure(text="(Drag & drop unavailable)\nUse Open Video")

        # View container: holds either single-view canvas or gallery view
        self.view_container = ttk.Frame(self.left)
        self.view_container.pack(fill="both", expand=True)

        # Single view frame
        self.single_view_frame = ttk.Frame(self.view_container)
        self.single_view_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.single_view_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._refresh_preview_after_resize)

        # Gallery view frame (hidden initially)
        self.gallery_frame = ttk.Frame(self.view_container)

        # Scrollable gallery canvas
        self.gallery_canvas = tk.Canvas(self.gallery_frame, highlightthickness=0)
        self.gallery_scroll = ttk.Scrollbar(self.gallery_frame, orient="vertical", command=self.gallery_canvas.yview)
        self.gallery_canvas.configure(yscrollcommand=self.gallery_scroll.set)

        self.gallery_scroll.pack(side="right", fill="y")
        self.gallery_canvas.pack(side="left", fill="both", expand=True)

        self.gallery_inner = ttk.Frame(self.gallery_canvas)
        self.gallery_canvas_window = self.gallery_canvas.create_window((0, 0), window=self.gallery_inner, anchor="nw")

        self.gallery_inner.bind("<Configure>", self._on_gallery_inner_configure)
        self.gallery_canvas.bind("<Configure>", self._on_gallery_canvas_configure)

        # Right panel
        right = ttk.Frame(main, width=420)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        info_frame = ttk.LabelFrame(right, text="Video Info", padding=10)
        info_frame.pack(fill="x")

        self.info_text = tk.Text(info_frame, height=9, wrap="word")
        self.info_text.pack(fill="x")
        self.info_text.configure(state="disabled")

        slider_frame = ttk.LabelFrame(right, text="Frame Navigation", padding=10)
        slider_frame.pack(fill="x", pady=(10, 0))

        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=0, orient="horizontal", command=self.on_slider_changed)
        self.frame_slider.pack(fill="x")

        self.frame_readout = ttk.Label(slider_frame, text="Frame: - / -    Time: -")
        self.frame_readout.pack(anchor="w", pady=(8, 0))

        # Quality controls
        q_frame = ttk.LabelFrame(right, text="Quality Filter", padding=10)
        q_frame.pack(fill="x", pady=(10, 0))

        self.skip_bad_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(q_frame, text="Skip low-quality frames (auto)", variable=self.skip_bad_var).pack(anchor="w")

        self.quality_label = ttk.Label(q_frame, text="Quality: -")
        self.quality_label.pack(anchor="w", pady=(6, 0))

        ttk.Label(q_frame, text="Blur threshold (higher = stricter):").pack(anchor="w", pady=(8, 0))
        self.blur_slider = ttk.Scale(q_frame, from_=10, to=300, orient="horizontal", command=self._on_quality_settings_changed)
        self.blur_slider.set(self.qeval.blur_threshold)
        self.blur_slider.pack(fill="x")

        ttk.Label(q_frame, text="Min brightness:").pack(anchor="w", pady=(8, 0))
        self.bright_slider = ttk.Scale(q_frame, from_=0, to=80, orient="horizontal", command=self._on_quality_settings_changed)
        self.bright_slider.set(self.qeval.brightness_min)
        self.bright_slider.pack(fill="x")

        ttk.Label(q_frame, text="Min contrast:").pack(anchor="w", pady=(8, 0))
        self.contrast_slider = ttk.Scale(q_frame, from_=2, to=40, orient="horizontal", command=self._on_quality_settings_changed)
        self.contrast_slider.set(self.qeval.contrast_min)
        self.contrast_slider.pack(fill="x")

        self.scan_btn = ttk.Button(q_frame, text="Scan quality (build good-frame index)", command=self.scan_quality, state="disabled")
        self.scan_btn.pack(fill="x", pady=(10, 0))

        self.clear_cache_btn = ttk.Button(q_frame, text="Clear Cache", command=self.clear_cache, state="disabled")
        self.clear_cache_btn.pack(fill="x", pady=(8, 0))

        self.scan_status = ttk.Label(q_frame, text="")
        self.scan_status.pack(anchor="w", pady=(6, 0))

        # Status bar
        self.status = ttk.Label(self.root, text="", anchor="w", padding=6)
        self.status.pack(fill="x")

        # Keyboard
        self.root.bind("<Left>", lambda e: self.step_frame(-1))
        self.root.bind("<Right>", lambda e: self.step_frame(+1))
        self.root.bind("<space>", lambda e: self.toggle_play())

        if not YTDLP_AVAILABLE:
            self.link_btn.configure(state="disabled")

    def _bind_button3_screenshot(self):
        def handler(event=None):
            if self.current_bgr is None:
                return "break"
            self.save_screenshot_to_downloads()
            return "break"

        try:
            self.canvas.bind("<Button-3>", handler)
        except Exception:
            pass

    # ---------------- Gallery layout helpers ----------------

    def _on_gallery_inner_configure(self, event):
        self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))

    def _on_gallery_canvas_configure(self, event):
        # Expand inner width to match canvas width
        try:
            self.gallery_canvas.itemconfigure(self.gallery_canvas_window, width=event.width)
        except Exception:
            pass

    # ---------------- Toggle view ----------------

    def toggle_view(self):
        if not self.video_path:
            return

        # stop playback when switching views
        if self.playing:
            self._stop_playback()

        self.gallery_mode = not self.gallery_mode

        if self.gallery_mode:
            self.toggle_view_btn.config(text="Single View")
            self._show_gallery_view()
        else:
            self.toggle_view_btn.config(text="Gallery View")
            self._show_single_view()

    def _show_single_view(self):
        self.gallery_frame.pack_forget()
        self.single_view_frame.pack(fill="both", expand=True)
        self._set_status("Single View enabled.")

    def _show_gallery_view(self):
        self.single_view_frame.pack_forget()
        self.gallery_frame.pack(fill="both", expand=True)
        self._set_status("Gallery View enabled (building thumbnails if needed).")

        # Build gallery if empty
        if not self.gallery_building and len(self.gallery_good_indices) == 0:
            self.build_gallery()

    # ---------------- Gallery build ----------------

    def clear_gallery(self):
        # stop any build
        self.gallery_stop_flag = True
        self.gallery_building = False
        self.gallery_queue.clear()

        # clear UI widgets
        for child in self.gallery_inner.winfo_children():
            try:
                child.destroy()
            except Exception:
                pass

        self.gallery_thumb_refs.clear()
        self.gallery_good_indices.clear()

        # reset flags
        self.gallery_stop_flag = False

    def build_gallery(self):
        if not self.video_path or self.total_frames <= 0:
            return

        self.clear_gallery()
        self.gallery_building = True
        self.gallery_stop_flag = False

        # Start background build
        threading.Thread(target=self._gallery_worker, daemon=True).start()
        self.root.after(50, self._gallery_consume_queue)

    def _gallery_worker(self):
        """
        Scans all frames, keeps only "valid" frames (quality OK),
        and generates thumbnails for display.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.gallery_queue.append(("__error__", None))
            return

        try:
            # thumbnail sizing (width)
            thumb_w = 190
            idx = 0
            while idx < self.total_frames:
                if self.gallery_stop_flag:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    idx += 1
                    continue

                # Evaluate quality (also cache for later)
                is_bad, reasons, metrics = self.qeval.evaluate(frame)
                self.quality_cache[idx] = (is_bad, reasons, metrics)

                if is_bad:
                    continue

                # Convert to thumbnail PIL
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                w, h = pil_img.size
                scale = thumb_w / max(1, w)
                thumb_h = max(1, int(h * scale))
                pil_thumb = pil_img.resize((thumb_w, thumb_h), Image.LANCZOS)

                # Push to UI queue (PhotoImage must be created in main thread)
                self.gallery_queue.append((idx, pil_thumb))

                # Lightweight progress message
                if idx % 150 == 0:
                    self.gallery_queue.append(("__progress__", idx))

                idx += 1
        finally:
            try:
                cap.release()
            except Exception:
                pass

        self.gallery_queue.append(("__done__", None))

    def _gallery_consume_queue(self):
        """
        Pull thumbnails from the queue and add them to the gallery UI safely.
        """
        if not self.gallery_mode:
            # Still consume but do less work if not visible
            pass

        # Add a batch per tick
        batch_size = 32
        added = 0

        while self.gallery_queue and added < batch_size:
            item = self.gallery_queue.popleft()

            if item[0] == "__error__":
                self.gallery_building = False
                self._set_status("Gallery build failed: could not open video for scanning.")
                return

            if item[0] == "__progress__":
                idx = item[1]
                self._set_status(f"Building gallery... scanned frame {idx}/{self.total_frames}")
                continue

            if item[0] == "__done__":
                self.gallery_building = False
                self._set_status(f"Gallery complete. Valid frames shown: {len(self.gallery_good_indices)}")
                return

            frame_idx, pil_thumb = item

            # Create PhotoImage on main thread
            photo = ImageTk.PhotoImage(pil_thumb)
            self.gallery_thumb_refs.append(photo)
            self.gallery_good_indices.append(frame_idx)

            self._add_gallery_tile(frame_idx, photo)

            added += 1

        # Keep polling while building
        if self.gallery_building or self.gallery_queue:
            self.root.after(40, self._gallery_consume_queue)

    def _add_gallery_tile(self, frame_idx: int, photo: ImageTk.PhotoImage):
        """
        Adds one thumbnail tile into a grid layout.
        Click -> open that frame in single view.
        """
        # Grid configuration
        cols = 4  # thumbnails per row
        tile_index = len(self.gallery_good_indices) - 1
        r = tile_index // cols
        c = tile_index % cols

        tile = ttk.Frame(self.gallery_inner, padding=6)
        tile.grid(row=r, column=c, sticky="n", padx=6, pady=6)

        img_label = tk.Label(tile, image=photo, bd=2, relief="groove", cursor="hand2")
        img_label.pack()

        meta = ttk.Label(tile, text=f"Frame {frame_idx}", anchor="center")
        meta.pack(pady=(4, 0))

        def on_click(event=None, idx=frame_idx):
            # Switch back to single view and show that frame
            if self.playing:
                self._stop_playback()

            self.gallery_mode = False
            self.toggle_view_btn.config(text="Gallery View")
            self._show_single_view()
            self.show_frame(idx, prefer_good=False)

        img_label.bind("<Button-1>", on_click)
        meta.bind("<Button-1>", on_click)
        img_label.bind("<Button-3>", lambda event, idx=frame_idx: self._show_gallery_context_menu(event, idx))

    # ---------------- Cache ----------------

    def clear_cache(self):
        self.quality_cache.clear()
        self.good_frames = None
        self.scan_status.config(text="Cache cleared.")
        self._set_status("Quality cache cleared (frames will be re-evaluated).")

        # Also clear gallery, because it's based on cached validity
        self.clear_gallery()

        if self.current_bgr is not None:
            self._update_quality_label(self.current_frame_index, self.current_bgr)

    # ---------------- Cleanup ----------------

    def on_close(self):
        try:
            self._stop_playback()
        except Exception:
            pass

        try:
            self.gallery_stop_flag = True
        except Exception:
            pass

        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception:
            pass

        try:
            self._cleanup_current_temp_video()
        except Exception:
            pass

        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass

    def _cleanup_current_temp_video(self):
        if self.current_is_temp and self.current_temp_path:
            try:
                if os.path.exists(self.current_temp_path):
                    os.remove(self.current_temp_path)
            except Exception:
                pass
        self.current_is_temp = False
        self.current_temp_path = None

    # ---------------- Status / Info ----------------

    def _set_status(self, msg: str):
        try:
            self.status.config(text=msg)
        except Exception:
            pass

    def _set_info(self, msg: str):
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", msg)
        self.info_text.configure(state="disabled")

    def _format_time(self, sec: float) -> str:
        if sec <= 0:
            return "-"
        m = int(sec // 60)
        s = sec - (m * 60)
        return f"{m:02d}:{s:06.3f}"

    def _frame_to_time(self, frame_idx: int) -> float:
        if self.fps <= 0:
            return 0.0
        return frame_idx / self.fps

    def _format_video_info(self) -> str:
        name = os.path.basename(self.video_path) if self.video_path else "-"
        source = "Downloaded Link (temporary)" if self.current_is_temp else "Local file"
        return (
            f"File: {name}\n"
            f"Source: {source}\n"
            f"Path: {self.video_path}\n\n"
            f"Total Frames: {self.total_frames}\n"
            f"FPS: {self.fps:.3f}\n"
            f"Duration: {self._format_time(self.duration_sec)}\n"
        )

    # ---------------- Quality ----------------

    def _on_quality_settings_changed(self, _=None):
        blur = float(self.blur_slider.get())
        bright = float(self.bright_slider.get())
        contrast = float(self.contrast_slider.get())

        self.qeval.set_thresholds(blur=blur, bright=bright, contrast=contrast)

        self.quality_cache.clear()
        self.good_frames = None
        self.scan_status.config(text="(thresholds changed — rescan optional)")

        # Clear gallery because validity changed
        self.clear_gallery()

        if self.current_bgr is not None:
            self._update_quality_label(self.current_frame_index, self.current_bgr)

    def _evaluate_frame_quality(self, idx, frame):
        if idx in self.quality_cache:
            return self.quality_cache[idx]
        is_bad, reasons, metrics = self.qeval.evaluate(frame)
        self.quality_cache[idx] = (is_bad, reasons, metrics)
        return self.quality_cache[idx]

    def _update_quality_label(self, idx, frame):
        is_bad, reasons, metrics = self._evaluate_frame_quality(idx, frame)
        if is_bad:
            label = f"Quality: BAD ({', '.join(reasons)}) | Blur={metrics['blur_score']:.1f}"
        else:
            label = f"Quality: OK | Blur={metrics['blur_score']:.1f}"
        self.quality_label.config(text=label)

    # ---------------- Local video loading ----------------

    def on_drop_file(self, event):
        raw = event.data
        path = clean_dnd_path(raw)

        candidates = [path]
        if " " in raw and not os.path.exists(raw.strip()):
            candidates = [clean_dnd_path(x) for x in raw.split()]

        chosen = None
        for c in candidates:
            if os.path.isfile(c) and c.lower().endswith(SUPPORTED_EXTENSIONS):
                chosen = c
                break

        if not chosen:
            messagebox.showwarning("Unsupported file", f"Please drop a video file: {SUPPORTED_EXTENSIONS}")
            return

        self.load_video(chosen, is_temp=False)

    def open_video_dialog(self):
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.mov *.m4v *.avi *.mkv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.load_video(path, is_temp=False)

    def load_video(self, path: str, is_temp: bool, temp_path: str = None):
        # stop playback on new load
        self._stop_playback()

        # clear gallery and stop background workers
        self.clear_gallery()

        # remove previous temp video if any
        self._cleanup_current_temp_video()

        self.quality_cache.clear()
        self.good_frames = None
        self.scan_status.config(text="")

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open video:\n{path}")
            return

        self.cap = cap
        self.video_path = path
        self.current_is_temp = bool(is_temp)
        self.current_temp_path = temp_path if is_temp else None

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.duration_sec = (self.total_frames / self.fps) if (self.fps > 0 and self.total_frames > 0) else 0.0

        self.current_frame_index = 0
        self.current_bgr = None

        for w in (
            self.toggle_view_btn, self.play_btn, self.export_btn, self.screenshot_btn,
            self.prev_btn, self.next_btn, self.jump_btn,
            self.scan_btn, self.clear_cache_btn
        ):
            w.configure(state="normal")

        self.frame_slider.configure(from_=0, to=max(self.total_frames - 1, 0))
        self._safe_set_slider(0)

        self._set_info(self._format_video_info())
        self._set_status(f"Loaded: {os.path.basename(path)}")

        # Force single view on load
        self.gallery_mode = False
        self.toggle_view_btn.config(text="Gallery View")
        self._show_single_view()

        self.show_frame(0, prefer_good=True, direction=+1)

    # ---------------- Link loading (yt-dlp) ----------------

    def load_from_link_clicked(self):
        if not YTDLP_AVAILABLE:
            messagebox.showerror("yt-dlp missing", "Install yt-dlp first:\n\npip install yt-dlp")
            return

        if self.is_downloading:
            self._set_status("Already downloading... please wait.")
            return

        url = self.link_entry.get().strip()
        if not url:
            messagebox.showwarning("Missing link", "Paste a video link first.")
            return

        if url.startswith("www."):
            url = "https://" + url
            self.link_entry.delete(0, "end")
            self.link_entry.insert(0, url)

        self.is_downloading = True
        self.link_btn.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        self._set_status("Starting download...")

        allow_insecure = bool(self.insecure_ssl_var.get())
        threading.Thread(
            target=self._download_and_load_url,
            args=(url, allow_insecure),
            daemon=True
        ).start()

    def _download_and_load_url(self, url: str, allow_insecure_ssl: bool):
        last_status = {"msg": ""}

        def progress_hook(d):
            try:
                status = d.get("status", "")
                if status == "downloading":
                    percent = d.get("_percent_str", "").strip()
                    speed = d.get("_speed_str", "").strip()
                    eta = d.get("_eta_str", "").strip()
                    msg = f"Downloading: {percent}  Speed: {speed}  ETA: {eta}"
                elif status == "finished":
                    msg = "Download finished. Processing..."
                else:
                    msg = f"Status: {status}"

                if msg != last_status["msg"]:
                    last_status["msg"] = msg
                    self.root.after(0, lambda m=msg: self._set_status(m))
            except Exception:
                pass

        try:
            outtmpl = os.path.join(self.temp_dir, "%(extractor)s_%(id)s_%(title).80s.%(ext)s")
            ydl_opts = {
                "outtmpl": outtmpl,
                "format": "mp4/bestvideo+bestaudio/best",
                "merge_output_format": "mp4",
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "progress_hooks": [progress_hook],
            }
            if allow_insecure_ssl:
                ydl_opts["nocheckcertificate"] = True

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filepath = ydl.prepare_filename(info)

            # Sometimes merged output becomes .mp4 even if prepared filename isn't
            if not os.path.exists(filepath):
                base, _ = os.path.splitext(filepath)
                mp4_candidate = base + ".mp4"
                if os.path.exists(mp4_candidate):
                    filepath = mp4_candidate

            if not os.path.exists(filepath):
                raise RuntimeError("Download succeeded but output file was not found.")

            self.root.after(0, lambda p=filepath: self._finish_load_downloaded(p))

        except Exception as ex:
            err = str(ex)
            self.root.after(0, lambda e=err: self._link_failed(e))

    def _finish_load_downloaded(self, filepath: str):
        self.is_downloading = False
        self.link_btn.configure(state="normal")
        self.open_btn.configure(state="normal")

        self._set_status(f"Downloaded. Loading: {os.path.basename(filepath)}")
        self.load_video(filepath, is_temp=True, temp_path=filepath)

    def _link_failed(self, error_msg: str):
        self.is_downloading = False
        self.link_btn.configure(state="normal")
        self.open_btn.configure(state="normal")

        messagebox.showerror(
            "Could not load link",
            "Failed to download this link.\n\n"
            "If you see SSL certificate errors on macOS, run:\n"
            "/Applications/Python 3.xx/Install Certificates.command\n\n"
            f"Error:\n{error_msg}"
        )
        self._set_status("Link download failed.")

    # ---------------- Scan quality ----------------

    def scan_quality(self):
        if not self.video_path or self.total_frames <= 0:
            return

        self.scan_btn.configure(state="disabled")
        self.scan_status.config(text="Scanning...")
        self._set_status("Scanning quality...")

        threading.Thread(target=self._scan_quality_worker, daemon=True).start()

    def _scan_quality_worker(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.root.after(0, lambda: self._scan_done("Scan failed: could not open video."))
            return

        good = []
        bad = 0

        try:
            idx = 0
            while idx < self.total_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    idx += 1
                    continue

                is_bad, reasons, metrics = self.qeval.evaluate(frame)
                self.quality_cache[idx] = (is_bad, reasons, metrics)

                if not is_bad:
                    good.append(idx)
                else:
                    bad += 1

                if idx % 200 == 0:
                    msg = f"Scanned {idx}/{self.total_frames} (good={len(good)}, bad={bad})"
                    self.root.after(0, lambda m=msg: self.scan_status.config(text=m))
                idx += 1

        finally:
            try:
                cap.release()
            except Exception:
                pass

        self.good_frames = good
        done_msg = f"Scan done. Good frames: {len(good)} | Bad frames: {bad}"
        self.root.after(0, lambda m=done_msg: self._scan_done(m))

    def _scan_done(self, msg: str):
        self.scan_btn.configure(state="normal")
        self.scan_status.config(text=msg)
        self._set_status(msg)

    # ---------------- Slider + navigation ----------------

    def _safe_set_slider(self, value: int):
        try:
            self._slider_updating = True
            self.frame_slider.set(value)
        finally:
            self._slider_updating = False

    def on_slider_changed(self, value):
        if self._slider_updating or not self.cap:
            return

        if self.playing:
            self._stop_playback()

        if self.gallery_mode:
            # keep slider usable even in gallery mode
            pass

        idx = int(float(value))
        self.show_frame(idx, prefer_good=False)

    def step_frame(self, direction: int):
        if not self.cap:
            return
        if self.playing:
            self._stop_playback()
        self.show_frame(self.current_frame_index + direction, prefer_good=True, direction=direction)

    def jump_to_frame(self):
        if not self.cap:
            return
        txt = self.jump_entry.get().strip()
        if not txt.isdigit():
            messagebox.showwarning("Invalid input", "Enter a valid frame number (integer).")
            return
        if self.playing:
            self._stop_playback()
        self.show_frame(int(txt), prefer_good=True, direction=+1)

    def show_frame(self, index: int, prefer_good: bool = True, direction: int = +1):
        if not self.cap:
            return

        if self.total_frames > 0:
            index = max(0, min(index, self.total_frames - 1))
        else:
            index = 0

        if prefer_good and self.skip_bad_var.get():
            index = self._find_nearest_good_frame(index, direction)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._set_status(f"Failed to read frame {index}.")
            return

        self.current_frame_index = index
        self.current_bgr = frame

        self._safe_set_slider(index)
        self._update_quality_label(index, frame)

        # Only render preview if in single view
        if not self.gallery_mode:
            self._render_preview(frame)

        t = self._frame_to_time(index)
        self.frame_readout.config(
            text=f"Frame: {index} / {max(self.total_frames - 1, 0)}    Time: {self._format_time(t)}"
        )

    def _find_nearest_good_frame(self, start_idx: int, direction: int):
        if self.total_frames <= 0:
            return start_idx

        # If already scanned
        if self.good_frames is not None and len(self.good_frames) > 0:
            if direction >= 0:
                for g in self.good_frames:
                    if g >= start_idx:
                        return g
                return self.good_frames[-1]
            else:
                for g in reversed(self.good_frames):
                    if g <= start_idx:
                        return g
                return self.good_frames[0]

        # On-demand evaluation
        idx = start_idx
        checks = 180
        while 0 <= idx < self.total_frames and checks > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                idx += direction
                checks -= 1
                continue

            is_bad, _, _ = self._evaluate_frame_quality(idx, frame)
            if not is_bad:
                return idx

            idx += direction
            checks -= 1

        return max(0, min(start_idx, self.total_frames - 1))

    # ---------------- Playback ----------------

    def toggle_play(self):
        if not self.cap:
            return
        if self.gallery_mode:
            self._set_status("Play is disabled in Gallery View. Switch to Single View to play.")
            return

        if self.playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if not self.cap:
            return

        self.playing = True
        self.play_btn.config(text="Pause")

        fps = self.fps if self.fps and self.fps > 1 else 30.0
        delay_ms = max(1, int(1000 / fps))

        def tick():
            if not self.playing:
                return

            next_idx = self.current_frame_index + 1
            if self.total_frames > 0 and next_idx >= self.total_frames:
                self._stop_playback()
                return

            self.show_frame(next_idx, prefer_good=True, direction=+1)
            self._play_after_id = self.root.after(delay_ms, tick)

        tick()

    def _stop_playback(self):
        self.playing = False
        try:
            self.play_btn.config(text="Play")
        except Exception:
            pass

        if self._play_after_id is not None:
            try:
                self.root.after_cancel(self._play_after_id)
            except Exception:
                pass
            self._play_after_id = None

    # ---------------- Rendering ----------------

    def _refresh_preview_after_resize(self, event):
        if self.current_bgr is not None and not self.gallery_mode:
            self._render_preview(self.current_bgr)

    def _render_preview(self, bgr_frame):
        try:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            canvas_w = max(self.canvas.winfo_width(), 1)
            canvas_h = max(self.canvas.winfo_height(), 1)

            if canvas_w < 50 or canvas_h < 50:
                canvas_w, canvas_h = 900, 600

            img = self._fit_image(img, canvas_w, canvas_h)
            self.preview_image = ImageTk.PhotoImage(img)

            self.canvas.delete("all")
            self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.preview_image, anchor="center")

        except Exception as ex:
            self._set_status(f"Render error: {ex}")

    def _fit_image(self, pil_img: Image.Image, max_w: int, max_h: int) -> Image.Image:
        w, h = pil_img.size
        scale = min(max_w / w, max_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return pil_img.resize((new_w, new_h), Image.LANCZOS)

    # ---------------- Screenshot + Export ----------------

    def save_screenshot_to_downloads(self):
        if self.current_bgr is None or not self.video_path:
            return

        now = time.time()
        if (
            self._last_screenshot_frame == self.current_frame_index
            and (now - self._last_screenshot_time) < self._screenshot_debounce_sec
        ):
            return

        self._last_screenshot_frame = self.current_frame_index
        self._last_screenshot_time = now

        downloads_dir = get_downloads_folder()
        base = os.path.splitext(os.path.basename(self.video_path))[0]
        filename = f"{base}_frame_{self.current_frame_index:06d}.png"
        out_path = os.path.join(downloads_dir, filename)

        if os.path.exists(out_path):
            n = 1
            while True:
                alt = os.path.join(downloads_dir, f"{base}_frame_{self.current_frame_index:06d}_{n}.png")
                if not os.path.exists(alt):
                    out_path = alt
                    break
                n += 1

        ok = cv2.imwrite(out_path, self.current_bgr)
        if not ok:
            messagebox.showerror("Save failed", "Could not save the screenshot.")
            return

        self._set_status(f"Saved screenshot: {out_path}")

    def save_frame_to_downloads(self, frame_idx: int):
        if not self.video_path:
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Save failed", "Could not open video to save frame.")
            return

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                messagebox.showerror("Save failed", "Could not read the selected frame.")
                return
        finally:
            cap.release()

        downloads_dir = get_downloads_folder()
        base = os.path.splitext(os.path.basename(self.video_path))[0]
        filename = f"{base}_frame_{frame_idx:06d}.png"
        out_path = os.path.join(downloads_dir, filename)

        if os.path.exists(out_path):
            n = 1
            while True:
                alt = os.path.join(downloads_dir, f"{base}_frame_{frame_idx:06d}_{n}.png")
                if not os.path.exists(alt):
                    out_path = alt
                    break
                n += 1

        ok = cv2.imwrite(out_path, frame)
        if not ok:
            messagebox.showerror("Save failed", "Could not save the selected frame.")
            return

        self._set_status(f"Saved gallery frame: {out_path}")

    def _show_gallery_context_menu(self, event, frame_idx: int):
        if self.gallery_menu is None:
            self.gallery_menu = tk.Menu(self.root, tearoff=0)
            self.gallery_menu.add_command(
                label="Save image to Downloads",
                command=lambda: self._save_gallery_context_frame()
            )

        self._gallery_menu_frame_idx = frame_idx
        try:
            self.gallery_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.gallery_menu.grab_release()

    def _save_gallery_context_frame(self):
        if self._gallery_menu_frame_idx is None:
            return
        self.save_frame_to_downloads(self._gallery_menu_frame_idx)

    def export_current_frame(self):
        if self.current_bgr is None or not self.video_path:
            return

        base = os.path.splitext(os.path.basename(self.video_path))[0]
        default_name = f"{base}_frame_{self.current_frame_index:06d}.png"

        out_path = filedialog.asksaveasfilename(
            title="Save frame as...",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("All files", "*.*")]
        )
        if not out_path:
            return

        ok = cv2.imwrite(out_path, self.current_bgr)
        if not ok:
            messagebox.showerror("Export failed", "Could not save the image.")
            return

        self._set_status(f"Exported: {out_path}")


def main():
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    app = VideoFrameExtractorApp(root)

    if not YTDLP_AVAILABLE:
        app._set_status("Link loading requires yt-dlp (pip install yt-dlp). Local files still work.")

    root.mainloop()


if __name__ == "__main__":
    main()
