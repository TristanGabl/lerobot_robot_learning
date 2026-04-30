#!/usr/bin/env python3
"""
Wayland-compatible GUI for LeRobot Data Collection.

This script acts as a graphical wrapper around the `lerobot-record` process. 
Because Wayland restricts global keyboard listeners (pynput), this GUI uses 
`unittest.mock.patch` to intercept the internal `init_keyboard_listener` and 
replace it with a dictionary of events controlled by tkinter buttons.

Usage:
You can pass the exact same arguments you would normally pass to `lerobot-record`.
Example:
    python robot_learning_2026/scripts/record_gui.py \
        --robot.type=so100_follower \
        --robot.port=/dev/tty.usbmodem12345 \
        --dataset.repo_id="clompa/my_dataset" \
        --dataset.single_task="GUI test task"
"""

import tkinter as tk
from tkinter import scrolledtext
import threading
import sys
import logging
from unittest.mock import patch

try:
    from lerobot.scripts.lerobot_record import record
except ImportError:
    print("Error: LeRobot must be installed or in the PYTHONPATH.")
    sys.exit(1)

# The shared state that replaces the pynput keyboard listener output.
# These flags are read by the `record_loop` inside lerobot.
GUI_EVENTS = {
    "exit_early": False,
    "rerecord_episode": False,
    "stop_recording": False
}

def mock_init_keyboard_listener():
    """Mocked version of init_keyboard_listener to be controlled by the GUI."""
    # Reset events when starting over
    GUI_EVENTS["exit_early"] = False
    GUI_EVENTS["rerecord_episode"] = False
    GUI_EVENTS["stop_recording"] = False
    # Return (None, events_dict) just like the original headless fallback
    return None, GUI_EVENTS


class RedirectText(logging.Handler):
    """Logging handler to forward logs to tkinter ScrolledText widget."""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + "\n")
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)
        self.text_widget.after(0, append)

class StdoutRedirector:
    """Redirects sys.stdout and sys.stderr to a tkinter ScrolledText widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, string)
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)
        self.text_widget.after(0, append)

    def flush(self):
        pass


class LeRobotRecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LeRobot Recorder (Wayland GUI)")
        self.root.geometry("640x480")

        self.thread = None
        self.is_recording = False

        self.setup_ui()

        # Redirect standard output and errors to the UI
        self.redirector = StdoutRedirector(self.log_area)
        sys.stdout = self.redirector
        sys.stderr = self.redirector
        
        # Add root logger forwarder
        self.log_handler = RedirectText(self.log_area)
        logging.getLogger().addHandler(self.log_handler)
        # Avoid setting format globally so it inherits lerobot's logger format

    def setup_ui(self):
        # Control frame for buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10, fill=tk.X)

        self.start_btn = tk.Button(btn_frame, text="▶️ Start Recording", bg="lightgreen", command=self.start_recording)
        self.start_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.stop_btn = tk.Button(btn_frame, text="⏹ Stop", bg="tomato", state=tk.DISABLED, command=self.stop_recording)
        self.stop_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.next_btn = tk.Button(btn_frame, text="⏭ Next Episode", state=tk.DISABLED, command=self.next_episode)
        self.next_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.rerecord_btn = tk.Button(btn_frame, text="🔁 Re-record", state=tk.DISABLED, command=self.rerecord_episode)
        self.rerecord_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # Status label
        self.status_lbl = tk.Label(self.root, text="Status: Idle", font=("Arial", 12, "bold"))
        self.status_lbl.pack(pady=5)

        # Help info
        info_lbl = tk.Label(self.root, text="Pass arguments via CLI, e.g.: python record_gui.py --robot.type=... --dataset.repo_id=...", fg="gray")
        info_lbl.pack()

        # Logs area
        self.log_area = scrolledtext.ScrolledText(self.root, state=tk.DISABLED, wrap=tk.WORD, height=15, bg="black", fg="white", font=("Consolas", 10))
        self.log_area.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

    def update_state(self, recording, status_text):
        self.is_recording = recording
        self.status_lbl.config(text=f"Status: {status_text}")
        
        self.start_btn.config(state=tk.DISABLED if recording else tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL if recording else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if recording else tk.DISABLED)
        self.rerecord_btn.config(state=tk.NORMAL if recording else tk.DISABLED)

    def run_recorder_thread(self):
        try:
            # Monkeypatch the pynput keyboard listener with our GUI events dictionary
            with patch('lerobot.scripts.lerobot_record.init_keyboard_listener', mock_init_keyboard_listener):
                print("Starting LeRobot dataset recording...")
                print(f"Propagating sys.argv arguments: {' '.join(sys.argv[1:])}")
                
                # record() automatically parses sys.argv utilizing the @parser.wrap() decorator
                record()
                
            print("\n--- Recording Finished cleanly ---")
        except Exception as e:
            print(f"\n[Error] Exception during recording: {e}")
        finally:
            # Safely request UI thread to turn buttons off
            self.root.after(0, lambda: self.update_state(False, "Idle"))

    def start_recording(self):
        if not self.is_recording:
            self.update_state(True, "Recording...")
            # Reset flags to safe defaults before thread creation
            GUI_EVENTS["exit_early"] = False
            GUI_EVENTS["rerecord_episode"] = False
            GUI_EVENTS["stop_recording"] = False
            
            # Start background task to prevent freezing the UI window
            self.thread = threading.Thread(target=self.run_recorder_thread, daemon=True)
            self.thread.start()

    def stop_recording(self):
        print("\n=> [GUI] Stop recording requested.")
        GUI_EVENTS["stop_recording"] = True
        GUI_EVENTS["exit_early"] = True

    def next_episode(self):
        print("\n=> [GUI] Next episode requested.")
        GUI_EVENTS["exit_early"] = True

    def rerecord_episode(self):
        print("\n=> [GUI] Re-record episode requested.")
        GUI_EVENTS["rerecord_episode"] = True
        GUI_EVENTS["exit_early"] = True

if __name__ == "__main__":
    root = tk.Tk()
    app = LeRobotRecorderGUI(root)
    root.mainloop()