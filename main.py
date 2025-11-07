import os
import queue
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from download_models import download_model
from faster_whisper import WhisperModel
from huggingface_hub.errors import HFValidationError
from moviepy import VideoFileClip
from utils import AVAILABLE_MODELS, HF_MODEL_MAPPING, AppState, Language


class TranscriptionGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Audio/Video Transcription")
        self.root.geometry("600x600")
        self.root.resizable(True, True)

        self.stop_flag = threading.Event()
        self.transcription_thread: threading.Thread | None = None
        self.is_transcribing = False

        self.result_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()

        self.state = AppState(
            audio_file=tk.StringVar(),
            output_file=tk.StringVar(),
            language=tk.StringVar(value=Language.AUTO),
            model_size=tk.StringVar(value="base"),
        )

        self.setup_ui()

        self.check_queue()

    def setup_ui(self) -> None:
        """Create interface."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        ttk.Label(
            main_frame,
            text="🎵 Audio/Video Transcription",
            font=("Arial", 16, "bold"),
        ).grid(row=0, column=0, columnspan=3, pady=(0, 20))

        self.create_file_selector(
            parent=main_frame,
            row=1,
            label="Audio/Video file:",
            var=self.state.audio_file,
            on_browse=self.browse_audio,
        )
        self.create_file_selector(
            parent=main_frame,
            row=2,
            label="Save as:",
            var=self.state.output_file,
            on_browse=self.browse_output,
        )

        self.create_settings_frame(main_frame, row=3)

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=20)

        self.start_button = ttk.Button(
            button_frame, text="Start Transcription", command=self.start_transcription, style="Accent.TButton"
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_transcription, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)

        self.progress_frame = ttk.Frame(main_frame)
        self.progress_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(20, 0))
        self.progress_frame.columnconfigure(0, weight=1)

        self.progress_bar = ttk.Progressbar(self.progress_frame, mode="indeterminate")
        self.progress_bar.grid(row=0, column=0, sticky="ew")

        self.status_label = ttk.Label(self.progress_frame, text="Ready to work", font=("Arial", 10))
        self.status_label.grid(row=1, column=0, pady=(5, 0))

        log_frame = ttk.LabelFrame(main_frame, text="Result", padding="5")
        log_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(20, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)

        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def log_message(self, message: str) -> None:
        """Add message to log."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def update_status(self, status: str) -> None:
        """Update status."""
        self.status_label.config(text=status)

    def start_transcription(self) -> None:
        """Start transcription in separate thread."""
        if self.is_transcribing:
            return

        # Check input data
        if not self.state.audio_file.get():
            messagebox.showerror("Error", "Select audio or video file")
            return

        if not self.state.output_file.get():
            messagebox.showerror("Error", "Specify file for saving")
            return

        self.stop_flag.clear()
        self.is_transcribing = True

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_bar.start()
        self.update_status("Initializing...")

        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

        self.transcription_thread = threading.Thread(target=self.transcription_worker, daemon=True)
        self.transcription_thread.start()

    def stop_transcription(self) -> None:
        """Stop transcription."""
        self.stop_flag.set()
        self.update_status("Stopping...")

    def is_video_file(self, file_path: str) -> bool:
        """Check if file is a video file based on extension."""
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in video_extensions

    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file and return path to temporary audio file."""
        self.result_queue.put(("status", "Extracting audio from video..."))
        self.result_queue.put(("log", f"Extracting audio from video: {os.path.basename(video_path)}"))
        
        # Create temporary audio file
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        try:
            video = VideoFileClip(video_path)
            if video.audio is None:
                raise ValueError("Video file does not contain an audio track")
            
            video.audio.write_audiofile(
                temp_audio_path,
                codec='pcm_s16le',
                logger=None
            )
            video.close()
            
            self.result_queue.put(("log", "Audio extracted successfully"))
            return temp_audio_path
        except Exception as e:
            # Clean up temp file if extraction failed
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            raise Exception(f"Failed to extract audio from video: {str(e)}")

    def transcription_worker(self) -> None:
        """Worker function for transcription in separate thread."""
        temp_audio_path = None
        try:
            input_path = self.state.audio_file.get()
            output_path = self.state.output_file.get()
            language = self.state.language.get() if self.state.language.get() != "auto" else None
            model_size = self.state.model_size.get()

            # Extract audio from video if needed
            if self.is_video_file(input_path):
                if self.stop_flag.is_set():
                    return
                temp_audio_path = self.extract_audio_from_video(input_path)
                audio_path = temp_audio_path
            else:
                audio_path = input_path

            self.result_queue.put(("status", "Loading model..."))
            self.result_queue.put(("log", f"Loading model: {model_size}"))

            if self.stop_flag.is_set():
                return

            try:
                model_path = AVAILABLE_MODELS[model_size]
                model = WhisperModel(model_path, device="cpu", compute_type="int8", local_files_only=True)

            except KeyError:
                self.result_queue.put(("log", f"Model '{model_size}' not found, using 'base'"))
                model_path = AVAILABLE_MODELS["base"]
                model = WhisperModel(model_path, device="cpu", compute_type="int8", local_files_only=True)

            except HFValidationError:
                self.result_queue.put(("log", f"Model '{model_size}' not found, downloading..."))
                model_path = AVAILABLE_MODELS[model_size]
                repo_id = HF_MODEL_MAPPING[model_size]
                download_model(model_name=model_size, repo_id=repo_id, local_path=model_path)
                self.result_queue.put(("log", f"Model '{model_size}' downloaded successfully"))
                model = WhisperModel(model_path, device="cpu", compute_type="int8", local_files_only=True)

            if self.stop_flag.is_set():
                return

            self.result_queue.put(("status", "Transcribing..."))
            self.result_queue.put(("log", f"Starting transcription of file: {os.path.basename(input_path)}"))

            segments, info = model.transcribe(audio_path, beam_size=1, language=language)

            if self.stop_flag.is_set():
                return

            self.result_queue.put(
                ("log", f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
            )
            self.result_queue.put(("log", f"Approximate duration: {info.duration:.2f} seconds"))

            processed_segments = 0

            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, start=1):
                    if self.stop_flag.is_set():
                        self.result_queue.put(("log", "Transcription stopped by user"))
                        break

                    # Write segment text
                    line = f"{segment.text.strip()}\n"
                    f.write(line)
                    processed_segments += 1

                    time.sleep(0.001)

            if not self.stop_flag.is_set():
                self.result_queue.put(("log", f"Transcription completed! Processed segments: {processed_segments}"))
                self.result_queue.put(("log", f"Result saved to: {output_path}"))
                self.result_queue.put(("success", "Transcription completed successfully!"))

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            self.result_queue.put(("error", error_msg))
            self.result_queue.put(("log", f"ERROR: {error_msg}"))

        finally:
            # Clean up temporary audio file if it was created
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                    self.result_queue.put(("log", "Temporary audio file cleaned up"))
                except Exception as e:
                    self.result_queue.put(("log", f"Warning: Could not delete temporary file: {str(e)}"))
            self.result_queue.put(("finished", None))

    def check_queue(self) -> None:
        """Check results queue and update interface."""
        try:
            while True:
                message_type, data = self.result_queue.get_nowait()

                if message_type == "status" and data is not None:
                    self.update_status(data)
                elif message_type == "log" and data is not None:
                    self.log_message(data)
                elif message_type == "error" and data is not None:
                    messagebox.showerror("Error", data)
                elif message_type == "success" and data is not None:
                    messagebox.showinfo("Success", data)
                elif message_type == "finished":
                    self.transcription_finished()

        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

    def transcription_finished(self) -> None:
        """Called when transcription is finished."""
        self.is_transcribing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.update_status("Ready to work")

    def create_settings_frame(self, parent: ttk.Frame, row: int) -> None:
        """Create settings block (language and model)."""
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding="10")
        settings_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=20)
        settings_frame.columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="Language:").grid(row=0, column=0, sticky=tk.W, pady=5)
        language_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.state.language,
            values=[lang.value for lang in Language],
            state="readonly",
        )
        language_combo.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=5)

        ttk.Label(settings_frame, text="Model quality:").grid(row=1, column=0, sticky=tk.W, pady=5)
        model_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.state.model_size,
            values=[size for size in AVAILABLE_MODELS.keys()],
            state="readonly",
        )
        model_combo.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=5)

    def create_file_selector(
        self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, on_browse: Callable[[], None]
    ) -> None:
        """Create row with label, entry and 'Browse' button."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=var, width=50).grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(parent, text="Browse...", command=on_browse).grid(row=row, column=2, pady=5)

    def browse_audio(self) -> None:
        """Select audio or video file."""
        filetypes = [
            ("All supported files", "*.mp3 *.wav *.m4a *.flac *.ogg *.aac *.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.m4v"),
            ("Audio files", "*.mp3 *.wav *.m4a *.flac *.ogg *.aac"),
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.m4v"),
            ("MP3 files", "*.mp3"),
            ("WAV files", "*.wav"),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("MKV files", "*.mkv")
        ]
        filename = filedialog.askopenfilename(title="Select audio or video file", filetypes=filetypes)
        if filename:
            self.state.audio_file.set(filename)
            if not self.state.output_file.get():
                base_name = os.path.splitext(filename)[0]
                self.state.output_file.set(f"{base_name}_transcript.txt")

    def browse_output(self) -> None:
        """Select file to save result."""
        filetypes = [
            ("Text files", "*.txt"),
            ("All files", "*.*"),
        ]
        filename = filedialog.asksaveasfilename(
            title="Save result as...", filetypes=filetypes, defaultextension=".txt"
        )
        if filename:
            self.state.output_file.set(filename)


if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionGUI(root)
    root.mainloop()
