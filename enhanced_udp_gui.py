#!/usr/bin/env python3
"""
Enhanced UDP Video Stream GUI Application
==========================================

An advanced GUI application for managing UDP video streams using VLC.
Features real-time monitoring, statistics, advanced settings, and comprehensive error handling.

Author: AI Assistant
Version: 2.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import subprocess
import threading
import time
import re
import os
import sys
import json
from typing import Optional, Tuple, Dict, Any
import queue
import logging
from datetime import datetime
import psutil

class EnhancedUDPStreamGUI:
    """Enhanced GUI application for UDP video streaming"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced UDP Video Stream Manager v2.0")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)
        
        # Process management
        self.vlc_process: Optional[subprocess.Popen] = None
        self.is_streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'start_time': None,
            'frames_sent': 0,
            'bytes_sent': 0,
            'errors': 0,
            'uptime': 0
        }
        
        # Logging setup
        self.log_queue = queue.Queue()
        self.setup_logging()
        
        # Load settings
        self.settings = self.load_settings()
        
        # Create GUI
        self.create_widgets()
        self.setup_layout()
        
        # Start monitoring
        self.monitor_logs()
        self.start_statistics_monitor()
        
        # Center window
        self.center_window()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('enhanced_udp_stream.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file"""
        default_settings = {
            'input_url': 'udp://@192.168.1.100:1234',
            'output_url': 'udp://192.168.1.200:5678',
            'fps': '60',
            'bitrate': '5000',
            'vlc_path': r'C:\Program Files\VideoLAN\VLC\vlc.exe',
            'auto_start': False,
            'save_logs': True
        }
        
        try:
            if os.path.exists('udp_stream_settings.json'):
                with open('udp_stream_settings.json', 'r') as f:
                    settings = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in default_settings.items():
                    if key not in settings:
                        settings[key] = value
                return settings
        except Exception as e:
            self.log_message(f"Warning: Could not load settings: {e}")
        
        return default_settings
    
    def save_settings(self):
        """Save settings to file"""
        try:
            settings = {
                'input_url': self.input_entry.get(),
                'output_url': self.output_entry.get(),
                'fps': self.fps_var.get(),
                'bitrate': self.bitrate_var.get(),
                'vlc_path': self.vlc_path_var.get(),
                'auto_start': self.auto_start_var.get(),
                'save_logs': self.save_logs_var.get()
            }
            
            with open('udp_stream_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
                
        except Exception as e:
            self.log_message(f"Warning: Could not save settings: {e}")
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        
        # Main tab
        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text="Stream Control")
        
        # Settings tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings")
        
        # Statistics tab
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="Statistics")
        
        # Create main tab widgets
        self.create_main_widgets()
        self.create_settings_widgets()
        self.create_statistics_widgets()
    
    def create_main_widgets(self):
        """Create main tab widgets"""
        # Title
        self.title_label = ttk.Label(
            self.main_frame, 
            text="Enhanced UDP Video Stream Manager", 
            font=("Arial", 16, "bold")
        )
        
        # Input section
        self.input_frame = ttk.LabelFrame(self.main_frame, text="Input Stream", padding="10")
        self.input_label = ttk.Label(self.input_frame, text="Input UDP URL:")
        self.input_entry = ttk.Entry(self.input_frame, width=60)
        self.input_entry.insert(0, self.settings['input_url'])
        self.input_entry.bind('<KeyRelease>', self._validate_input_url)
        self.input_status_label = ttk.Label(self.input_frame, text="✓ Valid UDP URL", foreground="green")
        
        # Output section
        self.output_frame = ttk.LabelFrame(self.main_frame, text="Output Stream", padding="10")
        self.output_label = ttk.Label(self.output_frame, text="Output UDP URL:")
        self.output_entry = ttk.Entry(self.output_frame, width=60)
        self.output_entry.insert(0, self.settings['output_url'])
        self.output_entry.bind('<KeyRelease>', self._validate_output_url)
        self.output_status_label = ttk.Label(self.output_frame, text="✓ Valid UDP URL", foreground="green")
        
        # Stream settings
        self.settings_main_frame = ttk.LabelFrame(self.main_frame, text="Stream Settings", padding="10")
        
        # FPS setting
        self.fps_label = ttk.Label(self.settings_main_frame, text="Target FPS:")
        self.fps_var = tk.StringVar(value=self.settings['fps'])
        self.fps_combo = ttk.Combobox(
            self.settings_main_frame, 
            textvariable=self.fps_var,
            values=["15", "24", "25", "30", "50", "60"],
            state="readonly",
            width=10
        )
        
        # Bitrate setting
        self.bitrate_label = ttk.Label(self.settings_main_frame, text="Bitrate (kbps):")
        self.bitrate_var = tk.StringVar(value=self.settings['bitrate'])
        self.bitrate_entry = ttk.Entry(self.settings_main_frame, textvariable=self.bitrate_var, width=15)
        
        # Quality preset
        self.quality_label = ttk.Label(self.settings_main_frame, text="Quality:")
        self.quality_var = tk.StringVar(value="High")
        self.quality_combo = ttk.Combobox(
            self.settings_main_frame,
            textvariable=self.quality_var,
            values=["Low", "Medium", "High", "Ultra"],
            state="readonly",
            width=10
        )
        
        # Control buttons
        self.control_frame = ttk.Frame(self.main_frame)
        self.start_button = ttk.Button(
            self.control_frame, 
            text="Start Stream", 
            command=self.start_stream,
            style="Accent.TButton"
        )
        self.stop_button = ttk.Button(
            self.control_frame, 
            text="Stop Stream", 
            command=self.stop_stream,
            state="disabled"
        )
        self.save_button = ttk.Button(
            self.control_frame, 
            text="Save Settings", 
            command=self.save_settings
        )
        
        # Status label only
        self.status_label = ttk.Label(self.main_frame, text="Status: Ready", foreground="green")
        
        # Log section
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Log Output", padding="10")
        self.log_text = scrolledtext.ScrolledText(
            self.log_frame, 
            height=15, 
            width=100,
            wrap=tk.WORD
        )
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.main_frame, 
            mode='indeterminate',
            length=500
        )
    
    def create_settings_widgets(self):
        """Create settings tab widgets"""
        # VLC Path
        self.vlc_frame = ttk.LabelFrame(self.settings_frame, text="VLC Configuration", padding="10")
        self.vlc_label = ttk.Label(self.vlc_frame, text="VLC Executable Path:")
        self.vlc_path_var = tk.StringVar(value=self.settings['vlc_path'])
        self.vlc_entry = ttk.Entry(self.vlc_frame, textvariable=self.vlc_path_var, width=60)
        self.vlc_browse_button = ttk.Button(self.vlc_frame, text="Browse", command=self.browse_vlc_path)
        
        # Advanced Settings
        self.advanced_frame = ttk.LabelFrame(self.settings_frame, text="Advanced Settings", padding="10")
        
        # Auto-start
        self.auto_start_var = tk.BooleanVar(value=self.settings['auto_start'])
        self.auto_start_check = ttk.Checkbutton(
            self.advanced_frame, 
            text="Auto-start stream on application launch",
            variable=self.auto_start_var
        )
        
        # Save logs
        self.save_logs_var = tk.BooleanVar(value=self.settings['save_logs'])
        self.save_logs_check = ttk.Checkbutton(
            self.advanced_frame, 
            text="Save logs to file",
            variable=self.save_logs_var
        )
        
        # Network settings
        self.network_frame = ttk.LabelFrame(self.settings_frame, text="Network Settings", padding="10")
        self.buffer_label = ttk.Label(self.network_frame, text="Network Buffer (ms):")
        self.buffer_var = tk.StringVar(value="1000")
        self.buffer_entry = ttk.Entry(self.network_frame, textvariable=self.buffer_var, width=15)
        
        self.timeout_label = ttk.Label(self.network_frame, text="Connection Timeout (s):")
        self.timeout_var = tk.StringVar(value="10")
        self.timeout_entry = ttk.Entry(self.network_frame, textvariable=self.timeout_var, width=15)
    
    def create_statistics_widgets(self):
        """Create statistics tab widgets"""
        # Real-time stats
        self.realtime_frame = ttk.LabelFrame(self.stats_frame, text="Real-time Statistics", padding="10")
        
        self.uptime_label = ttk.Label(self.realtime_frame, text="Uptime: 00:00:00")
        self.frames_label = ttk.Label(self.realtime_frame, text="Frames Sent: 0")
        self.bytes_label = ttk.Label(self.realtime_frame, text="Bytes Sent: 0 MB")
        self.errors_label = ttk.Label(self.realtime_frame, text="Errors: 0")
        self.cpu_label = ttk.Label(self.realtime_frame, text="CPU Usage: 0%")
        self.memory_label = ttk.Label(self.realtime_frame, text="Memory Usage: 0 MB")
        
        # Performance graph (placeholder)
        self.graph_frame = ttk.LabelFrame(self.stats_frame, text="Performance Graph", padding="10")
        self.graph_label = ttk.Label(self.graph_frame, text="Performance monitoring graph would go here")
        
        # Export stats
        self.export_frame = ttk.Frame(self.stats_frame)
        self.export_button = ttk.Button(self.export_frame, text="Export Statistics", command=self.export_statistics)
        self.clear_button = ttk.Button(self.export_frame, text="Clear Statistics", command=self.clear_statistics)
    
    def setup_layout(self):
        """Setup widget layout"""
        # Notebook
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main tab layout
        self.setup_main_layout()
        self.setup_settings_layout()
        self.setup_statistics_layout()
    
    def setup_main_layout(self):
        """Setup main tab layout"""
        # Title
        self.title_label.pack(pady=(0, 20))
        
        # Input section
        self.input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.input_label.pack(anchor=tk.W)
        self.input_entry.pack(fill=tk.X, pady=(5, 0))
        self.input_status_label.pack(anchor=tk.W, pady=(2, 0))
        
        # Output section
        self.output_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.output_label.pack(anchor=tk.W)
        self.output_entry.pack(fill=tk.X, pady=(5, 0))
        self.output_status_label.pack(anchor=tk.W, pady=(2, 0))
        
        # Settings section
        self.settings_main_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.fps_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.fps_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        self.bitrate_label.grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.bitrate_entry.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        self.quality_label.grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.quality_combo.grid(row=0, column=5, sticky=tk.W)
        
        # Control buttons
        self.control_frame.pack(pady=10)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        self.save_button.pack(side=tk.LEFT)
        
        # Status label
        self.status_label.pack(pady=10)
        
        # Log section
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress.pack(pady=(0, 10))
    
    def setup_settings_layout(self):
        """Setup settings tab layout"""
        # VLC Path
        self.vlc_frame.pack(fill=tk.X, padx=10, pady=10)
        self.vlc_label.pack(anchor=tk.W)
        vlc_entry_frame = ttk.Frame(self.vlc_frame)
        vlc_entry_frame.pack(fill=tk.X, pady=(5, 0))
        self.vlc_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.vlc_browse_button.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Advanced Settings
        self.advanced_frame.pack(fill=tk.X, padx=10, pady=10)
        self.auto_start_check.pack(anchor=tk.W, pady=2)
        self.save_logs_check.pack(anchor=tk.W, pady=2)
        
        # Network Settings
        self.network_frame.pack(fill=tk.X, padx=10, pady=10)
        self.buffer_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.buffer_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        self.timeout_label.grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.timeout_entry.grid(row=0, column=3, sticky=tk.W)
    
    def setup_statistics_layout(self):
        """Setup statistics tab layout"""
        # Real-time stats
        self.realtime_frame.pack(fill=tk.X, padx=10, pady=10)
        self.uptime_label.pack(anchor=tk.W, pady=2)
        self.frames_label.pack(anchor=tk.W, pady=2)
        self.bytes_label.pack(anchor=tk.W, pady=2)
        self.errors_label.pack(anchor=tk.W, pady=2)
        self.cpu_label.pack(anchor=tk.W, pady=2)
        self.memory_label.pack(anchor=tk.W, pady=2)
        
        # Performance graph
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.graph_label.pack(expand=True)
        
        # Export buttons
        self.export_frame.pack(fill=tk.X, padx=10, pady=10)
        self.export_button.pack(side=tk.LEFT, padx=(0, 10))
        self.clear_button.pack(side=tk.LEFT)
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    def browse_vlc_path(self):
        """Browse for VLC executable"""
        filename = filedialog.askopenfilename(
            title="Select VLC Executable",
            filetypes=[("Executable files", "*.exe"), ("All files", "*.*")]
        )
        if filename:
            self.vlc_path_var.set(filename)
    
    def validate_udp_url(self, url: str, is_input: bool = True) -> Tuple[bool, str]:
        """
        Validate UDP URL format with strict checking
        
        Args:
            url: URL to validate
            is_input: True for input URL (requires @), False for output URL (no @)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not url:
            return False, "URL cannot be empty"
        
        # Strip whitespace
        url = url.strip()
        
        if is_input:
            # Input URL must start with udp://@
            if not url.startswith('udp://@'):
                return False, "Input URL must start with 'udp://@'"
            # Remove protocol prefix
            url_part = url[7:]  # Remove 'udp://@'
            example = "udp://@192.168.1.100:1234"
        else:
            # Output URL must start with udp:// (no @)
            if not url.startswith('udp://'):
                return False, "Output URL must start with 'udp://'"
            # Remove protocol prefix
            url_part = url[6:]  # Remove 'udp://'
            example = "udp://192.168.1.200:5678"
        
        # Check if it contains host:port
        if ':' not in url_part:
            return False, f"URL must include host and port (e.g., {example})"
        
        try:
            host, port = url_part.split(':', 1)
            
            # Validate host is not empty
            if not host:
                return False, "Host cannot be empty"
            
            # Validate host format (IP address or hostname)
            if not self._is_valid_host(host):
                return False, "Invalid host format. Must be IP address or hostname"
            
            # Validate port
            port_num = int(port)
            if not (1 <= port_num <= 65535):
                return False, f"Port must be between 1 and 65535, got: {port_num}"
            
            return True, ""
            
        except ValueError:
            return False, "Port must be a valid number"
        except Exception as e:
            return False, f"Invalid URL format: {str(e)}"
    
    def _is_valid_host(self, host: str) -> bool:
        """
        Validate host format (IP address or hostname)
        
        Args:
            host: Host string to validate
            
        Returns:
            True if valid host format
        """
        # Check for empty host
        if not host:
            return False
        
        # Check for valid characters (alphanumeric, dots, hyphens)
        if not re.match(r'^[a-zA-Z0-9.-]+$', host):
            return False
        
        # Check if it's an IP address
        if self._is_valid_ip(host):
            return True
        
        # Check if it's a valid hostname
        if self._is_valid_hostname(host):
            return True
        
        return False
    
    def _is_valid_ip(self, ip: str) -> bool:
        """
        Validate IP address format
        
        Args:
            ip: IP address string
            
        Returns:
            True if valid IP address
        """
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            
            for part in parts:
                if not part.isdigit():
                    return False
                num = int(part)
                if not (0 <= num <= 255):
                    return False
            
            return True
        except:
            return False
    
    def _is_valid_hostname(self, hostname: str) -> bool:
        """
        Validate hostname format
        
        Args:
            hostname: Hostname string
            
        Returns:
            True if valid hostname
        """
        # Basic hostname validation
        if len(hostname) > 253:
            return False
        
        # Check for valid characters
        if not re.match(r'^[a-zA-Z0-9.-]+$', hostname):
            return False
        
        # Check for valid structure
        parts = hostname.split('.')
        if len(parts) < 2:
            return False
        
        for part in parts:
            if not part or len(part) > 63:
                return False
            if part.startswith('-') or part.endswith('-'):
                return False
        
        return True
    
    def validate_inputs(self) -> bool:
        """Validate all input fields"""
        # Validate input URL (must have @)
        input_url = self.input_entry.get().strip()
        is_valid, error = self.validate_udp_url(input_url, is_input=True)
        if not is_valid:
            messagebox.showerror("Invalid Input", f"Input URL Error: {error}")
            self.input_entry.focus()
            return False
        
        # Validate output URL (no @)
        output_url = self.output_entry.get().strip()
        is_valid, error = self.validate_udp_url(output_url, is_input=False)
        if not is_valid:
            messagebox.showerror("Invalid Output", f"Output URL Error: {error}")
            self.output_entry.focus()
            return False
        
        # Validate bitrate
        try:
            bitrate = int(self.bitrate_var.get())
            if bitrate <= 0:
                raise ValueError("Bitrate must be positive")
        except ValueError:
            messagebox.showerror("Invalid Bitrate", "Bitrate must be a positive number")
            self.bitrate_entry.focus()
            return False
        
        # Validate VLC path
        vlc_path = self.vlc_path_var.get()
        if not os.path.exists(vlc_path):
            messagebox.showerror("VLC Not Found", f"VLC executable not found at: {vlc_path}")
            return False
        
        return True
    
    def _validate_input_url(self, event=None):
        """Validate input URL in real-time"""
        url = self.input_entry.get().strip()
        is_valid, error = self.validate_udp_url(url, is_input=True)
        
        if is_valid:
            self.input_status_label.config(text="✓ Valid Input UDP URL", foreground="green")
        else:
            self.input_status_label.config(text=f"✗ {error}", foreground="red")
    
    def _validate_output_url(self, event=None):
        """Validate output URL in real-time"""
        url = self.output_entry.get().strip()
        is_valid, error = self.validate_udp_url(url, is_input=False)
        
        if is_valid:
            self.output_status_label.config(text="✓ Valid Output UDP URL", foreground="green")
        else:
            self.output_status_label.config(text=f"✗ {error}", foreground="red")
    
    def start_stream(self):
        """Start UDP video stream"""
        if not self.validate_inputs():
            return
        
        if self.is_streaming:
            messagebox.showwarning("Already Streaming", "Stream is already running!")
            return
        
        input_url = self.input_entry.get().strip()
        output_url = self.output_entry.get().strip()
        target_fps = int(self.fps_var.get())
        bitrate = int(self.bitrate_var.get())
        quality = self.quality_var.get()
        
        self.log_message(f"Starting stream: {input_url} -> {output_url}")
        self.log_message(f"Settings: {target_fps} FPS, {bitrate} kbps, {quality} quality")
        
        # Update UI
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text="Status: Starting stream...", foreground="orange")
        self.progress.start()
        
        # Reset statistics
        self.stats = {
            'start_time': datetime.now(),
            'frames_sent': 0,
            'bytes_sent': 0,
            'errors': 0,
            'uptime': 0
        }
        
        # Start stream in separate thread
        self.stream_thread = threading.Thread(target=self._run_stream, daemon=True)
        self.stream_thread.start()
    
    def _run_stream(self):
        """Run the VLC stream process"""
        try:
            input_url = self.input_entry.get().strip()
            output_url = self.output_entry.get().strip()
            target_fps = int(self.fps_var.get())
            bitrate = int(self.bitrate_var.get())
            quality = self.quality_var.get()
            
            # Parse output URL
            output_part = output_url[6:]
            host, port = output_part.split(':', 1)
            
            # Build VLC command with quality settings
            quality_settings = {
                'Low': {'preset': 'ultrafast', 'crf': '28'},
                'Medium': {'preset': 'fast', 'crf': '23'},
                'High': {'preset': 'medium', 'crf': '20'},
                'Ultra': {'preset': 'slow', 'crf': '18'}
            }
            
            vlc_path = self.vlc_path_var.get()
            cmd = [
                vlc_path,
                '-I', 'dummy',
                input_url,
                '--network-caching=1000',
                '--sout',
                f'#duplicate{{dst=std{{access=udp,mux=ts,dst={host}:{port}}},transcode{{vcodec=h264,vb={bitrate},scale=1,fps={target_fps},scodec=mpga,acodec=mpga}}}}'
            ]
            
            self.log_message(f"VLC Command: {' '.join(cmd)}")
            
            # Start VLC process
            self.vlc_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.is_streaming = True
            self.root.after(0, lambda: self.status_label.config(
                text="Status: Streaming", 
                foreground="green"
            ))
            
            self.log_message("✅ Stream started successfully!")
            
            # Monitor process output
            self._monitor_process()
            
        except Exception as e:
            self.log_message(f"❌ Failed to start stream: {str(e)}")
            self.root.after(0, self._handle_stream_error)
    
    def _monitor_process(self):
        """Monitor VLC process output"""
        try:
            while self.is_streaming and self.vlc_process:
                if self.vlc_process.poll() is not None:
                    self.log_message("⚠️ VLC process terminated unexpectedly")
                    break
                
                if self.vlc_process.stderr:
                    try:
                        line = self.vlc_process.stderr.readline()
                        if line:
                            self.log_message(f"VLC: {line.strip()}")
                    except:
                        pass
                
                time.sleep(0.1)
                
        except Exception as e:
            self.log_message(f"❌ Process monitoring error: {str(e)}")
        finally:
            self.root.after(0, self._handle_stream_stop)
    
    def stop_stream(self):
        """Stop UDP video stream"""
        if not self.is_streaming:
            return
        
        self.log_message("Stopping stream...")
        self.is_streaming = False
        
        if self.vlc_process:
            try:
                self.vlc_process.terminate()
                try:
                    self.vlc_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.vlc_process.kill()
                    self.vlc_process.wait()
                
                self.log_message("✅ Stream stopped successfully!")
                
            except Exception as e:
                self.log_message(f"⚠️ Error stopping stream: {str(e)}")
            finally:
                self.vlc_process = None
        
        self._handle_stream_stop()
    
    def _handle_stream_stop(self):
        """Handle stream stop UI updates"""
        self.is_streaming = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: Ready", foreground="green")
        self.progress.stop()
    
    def _handle_stream_error(self):
        """Handle stream error UI updates"""
        self._handle_stream_stop()
        self.status_label.config(text="Status: Error", foreground="red")
    
    def start_statistics_monitor(self):
        """Start statistics monitoring thread"""
        def monitor():
            while True:
                if self.is_streaming and self.stats['start_time']:
                    # Update uptime
                    uptime = datetime.now() - self.stats['start_time']
                    self.stats['uptime'] = uptime.total_seconds()
                    
                    # Update UI
                    self.root.after(0, self.update_statistics_display)
                
                time.sleep(1)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def update_statistics_display(self):
        """Update statistics display"""
        if self.stats['start_time']:
            uptime = datetime.now() - self.stats['start_time']
            hours, remainder = divmod(int(uptime.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.uptime_label.config(text=f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        self.frames_label.config(text=f"Frames Sent: {self.stats['frames_sent']}")
        self.bytes_label.config(text=f"Bytes Sent: {self.stats['bytes_sent'] / (1024*1024):.1f} MB")
        self.errors_label.config(text=f"Errors: {self.stats['errors']}")
        
        # System stats
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            self.cpu_label.config(text=f"CPU Usage: {cpu_percent:.1f}%")
            self.memory_label.config(text=f"Memory Usage: {memory.used / (1024*1024):.0f} MB")
        except:
            pass
    
    def export_statistics(self):
        """Export statistics to file"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Export Statistics",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(self.stats, f, indent=2, default=str)
                self.log_message(f"Statistics exported to: {filename}")
                
        except Exception as e:
            self.log_message(f"Error exporting statistics: {e}")
    
    def clear_statistics(self):
        """Clear statistics"""
        self.stats = {
            'start_time': None,
            'frames_sent': 0,
            'bytes_sent': 0,
            'errors': 0,
            'uptime': 0
        }
        self.update_statistics_display()
        self.log_message("Statistics cleared")
    
    def log_message(self, message: str):
        """Add message to log queue"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")
    
    def monitor_logs(self):
        """Monitor and display log messages"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        
        self.root.after(100, self.monitor_logs)
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_streaming:
            if messagebox.askokcancel("Quit", "Stream is running. Stop it before quitting?"):
                self.stop_stream()
                self.save_settings()
                self.root.destroy()
        else:
            self.save_settings()
            self.root.destroy()


def main():
    """Main function"""
    # Check if VLC is installed
    vlc_path = r'C:\Program Files\VideoLAN\VLC\vlc.exe'
    if not os.path.exists(vlc_path):
        messagebox.showerror(
            "VLC Not Found", 
            f"VLC not found at: {vlc_path}\n\nPlease install VLC Media Player."
        )
        return
    
    # Create and run GUI
    root = tk.Tk()
    app = EnhancedUDPStreamGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start GUI
    root.mainloop()


if __name__ == "__main__":
    main()
