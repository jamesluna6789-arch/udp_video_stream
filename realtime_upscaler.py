#!/usr/bin/env python3
"""
Real-Time Video Upscaling & Enhancement System
==============================================

This script implements a real-time video upscaling pipeline using Real-ESRGAN,
an advanced AI-based super-resolution model. It enhances low-resolution videos
to high-resolution outputs (up to 4×) while maintaining visual fidelity.

Author: AI Assistant
Version: 1.0
"""

import argparse
import cv2
import torch
import numpy as np
import time
import os
import sys
from pathlib import Path
import logging
from typing import Optional, Tuple, Union
import subprocess
import threading
from queue import Queue, Empty

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('upscaler.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RealTimeUpscaler:
    """
    Real-time video upscaling system using Real-ESRGAN
    """
    
    def __init__(self, model_path: str = "RealESRGAN_x4plus.pth", 
                 device: str = "auto", target_width: int = 1920, target_height: int = 1080):
        """
        Initialize the upscaler
        
        Args:
            model_path: Path to Real-ESRGAN model weights
            device: Device to use ('auto', 'cuda', 'cpu')
            target_width: Target output width (default: 1920)
            target_height: Target output height (default: 1080)
        """
        self.model_path = model_path
        self.target_width = target_width
        self.target_height = target_height
        self.device = self._setup_device(device)
        self.model = None
        self.frame_queue = Queue(maxsize=10)
        self.output_queue = Queue(maxsize=10)
        self.running = False
        self.stats = {
            'frames_processed': 0,
            'frames_dropped': 0,
            'total_time': 0,
            'fps': 0,
            'processing_times': [],
            'target_fps': 0,
            'adaptive_quality': True
        }
        self.last_frame_time = 0
        self.frame_interval = 1.0 / 60.0  # 60 FPS interval
        
        logger.info(f"Initializing Real-Time Upscaler on {self.device}")
        self._load_model()
    
    def _calculate_scale_factor(self, input_width: int, input_height: int) -> float:
        """
        Calculate scale factor based on input resolution and target resolution
        
        Args:
            input_width: Input frame width
            input_height: Input frame height
            
        Returns:
            Scale factor as float
        """
        # Always use target resolution to calculate scale
        scale_x = self.target_width / input_width
        scale_y = self.target_height / input_height
        
        # Use the smaller scale to maintain aspect ratio
        scale_factor = min(scale_x, scale_y)
        
        logger.info(f"Auto-calculated scale: {scale_factor:.2f}x (input: {input_width}x{input_height} -> target: {self.target_width}x{self.target_height})")
        return scale_factor
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup and validate the computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_count = torch.cuda.device_count()
                
                logger.info(f"[OK] CUDA available: {gpu_name}")
                logger.info(f"GPU Count: {gpu_count}")
                logger.info(f"VRAM: {gpu_memory:.1f} GB")
                
                # Check if we have enough VRAM for Real-ESRGAN
                if gpu_memory < 4.0:
                    logger.warning(f"[WARNING] Low VRAM detected ({gpu_memory:.1f} GB). Real-ESRGAN may run slowly.")
                    logger.info("Consider using CPU or reducing batch size for better performance.")
                
                # Set memory fraction to avoid OOM errors
                try:
                    torch.cuda.set_per_process_memory_fraction(0.8)
                except Exception as e:
                    logger.warning(f"Could not set memory fraction: {e}")
                
            else:
                device = "cpu"
                logger.warning("[ERROR] CUDA not available, using CPU")
                logger.info("[INFO] For better performance, install CUDA-compatible PyTorch:")
                logger.info("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        elif device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("[ERROR] CUDA requested but not available, falling back to CPU")
                device = "cpu"
            else:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"[OK] Using GPU: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
                try:
                    # Set memory fraction and configure memory management
                    torch.cuda.set_per_process_memory_fraction(0.9)  # Use more memory
                    
                    # Set memory allocation strategy to reduce fragmentation
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
                    
                    # Clear any existing cache
                    torch.cuda.empty_cache()
                    
                    logger.info("GPU memory management configured (90% allocation, fragmentation handling enabled)")
                except Exception as e:
                    logger.warning(f"Could not configure GPU memory: {e}")
        
        return torch.device(device)
    
    def _get_gpu_memory_info(self) -> dict:
        """Get GPU memory information"""
        if self.device.type == 'cuda':
            try:
                allocated = torch.cuda.memory_allocated(0) / 1e9
                cached = torch.cuda.memory_reserved(0) / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                return {
                    'allocated': allocated,
                    'cached': cached,
                    'total': total,
                    'free': total - allocated
                }
            except:
                return {}
        return {}
    
    def _clear_gpu_cache_aggressively(self):
        """Clear GPU cache aggressively to free up memory"""
        if self.device.type == 'cuda':
            try:
                # Clear PyTorch cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear cache again after garbage collection
                torch.cuda.empty_cache()
                
                # Log memory status
                mem_info = self._get_gpu_memory_info()
                if mem_info:
                    logger.info(f"GPU cache cleared - Free: {mem_info['free']:.1f} GB, Allocated: {mem_info['allocated']:.1f} GB")
                    
            except Exception as e:
                logger.warning(f"Could not clear GPU cache: {e}")
    
    def _check_memory_pressure(self) -> bool:
        """Check if GPU is under memory pressure"""
        if self.device.type == 'cuda':
            try:
                mem_info = self._get_gpu_memory_info()
                if mem_info:
                    # Consider memory pressure if less than 1GB free or >80% allocated
                    free_gb = mem_info['free']
                    allocated_ratio = mem_info['allocated'] / mem_info['total']
                    return free_gb < 1.0 or allocated_ratio > 0.8
            except Exception as e:
                logger.warning(f"Could not check memory pressure: {e}")
        return False
    
    def _log_gpu_memory(self, context: str = ""):
        """Log current GPU memory usage"""
        if self.device.type == 'cuda':
            mem_info = self._get_gpu_memory_info()
            if mem_info:
                logger.debug(f"GPU Memory {context}: "
                           f"Allocated: {mem_info['allocated']:.1f}GB, "
                           f"Cached: {mem_info['cached']:.1f}GB, "
                           f"Free: {mem_info['free']:.1f}GB")
    
    def _load_model(self, scale_factor: float = None):
        """Load the Real-ESRGAN model"""
        try:
            # Use provided scale factor or default to 2
            if scale_factor is None:
                scale_factor = 2.0
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Please download RealESRGAN_x4plus.pth from:")
                logger.info("https://github.com/xinntao/Real-ESRGAN/releases")
                logger.info("Or run: python download_model.py")
                logger.info("Using OpenCV fallback for upscaling...")
                self.model = None
                return
            
            # Try to import Real-ESRGAN with fallback
            try:
                # Import with better error handling
                import torch
                if not torch.cuda.is_available() and self.device.type == 'cuda':
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    self.device = torch.device('cpu')
               
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
                
                # Initialize model with dynamic scale
                model_scale = int(scale_factor) if scale_factor >= 1.0 else 2
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                              num_block=23, num_grow_ch=32, scale=model_scale)
                
                # Optimize tile size based on GPU memory and scale factor
                tile_size = 0  # No tiling for best quality
                if self.device.type == 'cuda':
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    
                    # Calculate estimated memory usage for 4K input with current scale
                    estimated_memory_gb = (self.scale ** 2) * 0.5  # Rough estimate: 0.5GB per scale^2
                    
                    if gpu_memory < 6.0 or estimated_memory_gb > gpu_memory * 0.7:
                        # Use tiling for lower VRAM GPUs or when estimated usage is high
                        if self.scale >= 4:
                            tile_size = 256  # Smaller tiles for 4x+ scaling
                        else:
                            tile_size = 512  # Standard tiles for 2x scaling
                        logger.info(f"Using tiling (tile_size={tile_size}) for GPU with {gpu_memory:.1f} GB VRAM (estimated usage: {estimated_memory_gb:.1f} GB)")
                    else:
                        logger.info(f"Using full resolution processing on GPU with {gpu_memory:.1f} GB VRAM (estimated usage: {estimated_memory_gb:.1f} GB)")
                
                self.model = RealESRGANer(
                    scale=model_scale,
                    model_path=self.model_path,
                    model=model,
                    tile=tile_size,
                    tile_pad=10,
                    pre_pad=0,
                    half=True if self.device.type == 'cuda' else False,
                    gpu_id=0 if self.device.type == 'cuda' else None
                )
                
            except ImportError as e:
                logger.info(f"Real-ESRGAN not available: {e}")
                logger.info("Using OpenCV interpolation for upscaling...")
                self.model = None
                return
            
            logger.info(f"Model loaded successfully (scale: {self.scale}x)")
            self._log_gpu_memory("after model loading")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to simplified upscaling...")
            self.model = None
    
    def _is_udp_stream(self, input_path: str) -> bool:
        """Check if input is a UDP stream"""
        return input_path.startswith('udp://') or input_path.startswith('rtp://')
    
    def _is_rtmp_stream(self, input_path: str) -> bool:
        """Check if input is an RTMP stream"""
        return input_path.startswith('rtmp://') or input_path.startswith('rtmps://')
    
    def _is_http_stream(self, input_path: str) -> bool:
        """Check if input is an HTTP stream"""
        return input_path.startswith('http://') or input_path.startswith('https://')
    
    def _is_stream(self, input_path: str) -> bool:
        """Check if input is any type of stream"""
        return (self._is_udp_stream(input_path) or 
                self._is_rtmp_stream(input_path) or 
                self._is_http_stream(input_path))
    
    def _test_stream_connection(self, stream_url: str, timeout: int = 5) -> bool:
        """Test if a stream is accessible"""
        try:
            logger.info(f"Testing stream connection: {stream_url} (timeout: {timeout}s)")
            
            # Use threading to implement proper timeout
            import threading
            import time
            
            result = {'success': False, 'error': None}
            
            def test_connection():
                try:
                    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout * 1000)
                    
                    if cap.isOpened():
                        # Try to read one frame to verify stream is working
                        ret, frame = cap.read()
                        cap.release()
                        if ret and frame is not None:
                            result['success'] = True
                        else:
                            result['error'] = "No frames received"
                    else:
                        result['error'] = "Cannot open stream"
                except Exception as e:
                    result['error'] = str(e)
            
            # Start connection test in thread
            thread = threading.Thread(target=test_connection)
            thread.daemon = True
            thread.start()
            
            # Wait for timeout
            thread.join(timeout=timeout)
            
            if thread.is_alive():
                logger.warning(f"Stream connection test timed out after {timeout}s")
                return False
            
            if result['success']:
                logger.info("Stream connection test successful")
                return True
            else:
                logger.warning(f"Stream connection test failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.warning(f"Stream connection test failed: {e}")
            return False
    
    def _test_stream_connection_ffmpeg(self, stream_url: str, timeout: int = 5) -> bool:
        """Test stream connection using FFmpeg directly"""
        try:
            logger.info(f"Testing stream connection with FFmpeg: {stream_url} (timeout: {timeout}s)")
            
            # Add buffer options for UDP streams to prevent overrun
            test_url = stream_url
            if self._is_udp_stream(stream_url):
                test_url = f"{stream_url}?fifo_size=50000000&overrun_nonfatal=1"
            
            # FFmpeg command to test stream
            cmd = [
                'ffmpeg',
                '-i', test_url,
                '-t', '1',  # Test for 1 second
                '-f', 'null',  # No output
                '-',  # Output to stdout
                '-y'  # Overwrite
            ]
            
            # Run FFmpeg with timeout
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                # Wait for process with timeout
                stdout, stderr = process.communicate(timeout=timeout)
                
                if process.returncode == 0:
                    logger.info("FFmpeg stream connection test successful")
                    return True
                else:
                    logger.warning(f"FFmpeg stream test failed (code {process.returncode})")
                    if stderr:
                        logger.debug(f"FFmpeg error: {stderr[:200]}...")
                    return False
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"FFmpeg stream test timed out after {timeout}s")
                process.kill()
                process.wait()
                return False
                
        except FileNotFoundError:
            logger.warning("FFmpeg not found, falling back to OpenCV test")
            return self._test_stream_connection(stream_url, timeout)
        except Exception as e:
            logger.warning(f"FFmpeg stream test failed: {e}")
            return False
    
    def _parse_udp_url(self, udp_url: str) -> Tuple[str, int]:
        """Parse UDP URL to extract host and port"""
        # Remove protocol prefix
        if udp_url.startswith('udp://'):
            url = udp_url[6:]
        elif udp_url.startswith('rtp://'):
            url = udp_url[6:]
        else:
            url = udp_url
        
        # Parse host:port
        if ':' in url:
            host, port = url.split(':', 1)
            return host, int(port)
        else:
            return url, 1234  # Default port
    
    def _create_udp_capture(self, udp_url: str) -> cv2.VideoCapture:
        """Create OpenCV VideoCapture for UDP stream"""
        try:
            # For UDP streams, we need to use FFmpeg backend with buffer management
            # Add buffer options to prevent circular buffer overrun
            enhanced_url = f"{udp_url}?fifo_size=50000000&overrun_nonfatal=1"
            
            cap = cv2.VideoCapture(enhanced_url, cv2.CAP_FFMPEG)
            
            # Set timeout properties to avoid long waits
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second timeout
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5 second read timeout
            
            return cap
        except Exception as e:
            logger.error(f"Failed to create UDP capture: {e}")
            return None
    
    def _get_fps_filter(self, target_fps: int, method: str) -> str:
        """
        Get FFmpeg FPS filter based on method
        
        Args:
            target_fps: Target frame rate
            method: Conversion method
            
        Returns:
            FFmpeg filter string
        """
        if method == 'standard':
            return f'fps={target_fps}'
        elif method == 'interpolation':
            return f'fps={target_fps}:round=near'
        elif method == 'motion_compensated':
            return f'fps={target_fps}:round=near:interp=linear'
        else:
            return f'fps={target_fps}'
    
    def _create_udp_writer(self, input_url: str, udp_url: str, width: int, height: int, fps: int, target_fps: int = 60, 
                          fps_method: str = 'standard', fps_quality: str = 'high') -> bool:
        """
        Create FFmpeg process for UDP output with improved encoding parameters
        
        Key improvements to prevent reference frame errors:
        - Dynamic GOP size based on FPS (2 seconds worth of frames)
        - Keyframe every second for better stream recovery
        - No B-frames to avoid reference frame dependencies
        - Limited reference frames (1) for simpler decoding
        - Closed GOP for better streaming stability
        """
        try:
            host, port = self._parse_udp_url(udp_url)

            cmd = [
                r'C:\Program Files\VideoLAN\VLC\vlc.exe',
                '-I', 'dummy',
                input_url,
                '--network-caching=1000',
                '--sout',
                '#duplicate{dst=std{access=udp,mux=ts,dst='+str(host)+':'+str(port)+'},transcode{vcodec=h264,vb=5000,scale=1,fps='+str(target_fps)+',scodec=mpga,acodec=mpga}}'
            ]            
            
            logger.info(f"Starting UDP stream to {host}:{port}")
            logger.info(f"FFmpeg command: {' '.join(cmd)}")
            
            process =  subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()  # Wait for the process to complete (you can handle the output here)
            if stdout:
                logger.info(stdout.decode())
            if stderr:
                logger.error(stderr.decode())

            # process = subprocess.Popen(
            #     cmd, 
            #     stdin=subprocess.PIPE, 
            #     stdout=subprocess.PIPE, 
            #     stderr=subprocess.PIPE,
            #     bufsize=0,  # Unbuffered
            #     preexec_fn=None if os.name == 'nt' else os.setsid,  # Process group for better cleanup
            #     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            # )
            
            # Start a thread to monitor stderr for errors
            
            
        except Exception as e:
            logger.error(f"Failed to create UDP writer: {e}")
            return None
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    
    def _postprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Postprocess model output"""
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Ensure frame is in correct format for FFmpeg
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)
        
        # Ensure frame is contiguous in memory
        if not frame_bgr.flags['C_CONTIGUOUS']:
            frame_bgr = np.ascontiguousarray(frame_bgr)
            
        return frame_bgr
    
    def upscale_frame(self, frame: np.ndarray, scale_factor: float = None) -> np.ndarray:
        """
        Upscale a single frame using Real-ESRGAN or fallback method
        
        Args:
            frame: Input frame (BGR format)
            scale_factor: Scale factor to use (if None, uses calculated scale)
            
        Returns:
            Upscaled frame (BGR format)
        """
        try:
            # Calculate scale factor if not provided
            if scale_factor is None:
                input_height, input_width = frame.shape[:2]
                scale_factor = self._calculate_scale_factor(input_width, input_height)
            
            if self.model is not None:
                # Use Real-ESRGAN if available
                frame_rgb = self._preprocess_frame(frame)
                
                # Clear GPU cache before processing to avoid memory issues
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    self._log_gpu_memory("before enhancement")
                
                # Use the calculated scale factor
                model_scale = int(scale_factor) if scale_factor >= 1.0 else 2
                output, _ = self.model.enhance(frame_rgb, outscale=model_scale)
                output_bgr = self._postprocess_frame(output)
                
                # Clear GPU cache after processing
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    self._log_gpu_memory("after enhancement")
                
                return output_bgr
            else:
                # Fallback to OpenCV interpolation
                logger.debug("Using OpenCV interpolation fallback")
                height, width = frame.shape[:2]
                
                # Always use target resolution directly
                upscaled = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_CUBIC)
                
                return upscaled
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.device.type == 'cuda':
                logger.warning(f"GPU out of memory: {e}")
                logger.info("Clearing GPU cache aggressively and retrying...")
                
                # Use aggressive cache clearing
                self._clear_gpu_cache_aggressively()
                
                # Check if we still have memory pressure
                if self._check_memory_pressure():
                    logger.warning("Still under memory pressure, trying with reduced processing...")
                    
                    # Try with smaller frame or fallback to CPU
                    try:
                        # Resize frame to half size and then upscale
                        small_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
                        frame_rgb = self._preprocess_frame(small_frame)
                        output, _ = self.model.enhance(frame_rgb, outscale=self.scale*2)  # Double scale to compensate
                        output_bgr = self._postprocess_frame(output)
                        return output_bgr
                    except:
                        logger.warning("GPU processing failed, falling back to OpenCV")
                        return self._fallback_upscale(frame)
                else:
                    # Try again with original frame after cache clearing
                    try:
                        frame_rgb = self._preprocess_frame(frame)
                        output, _ = self.model.enhance(frame_rgb, outscale=self.scale)
                        output_bgr = self._postprocess_frame(output)
                        return output_bgr
                    except:
                        logger.warning("Retry failed, falling back to OpenCV")
                        return self._fallback_upscale(frame)
            else:
                logger.error(f"Error upscaling frame: {e}")
                return self._fallback_upscale(frame)
        except Exception as e:
            logger.error(f"Error upscaling frame: {e}")
            return self._fallback_upscale(frame)
    
    def _fallback_upscale(self, frame: np.ndarray, scale_factor: float = None) -> np.ndarray:
        """Fallback upscaling method using OpenCV"""
        height, width = frame.shape[:2]
        
        # Always use target resolution directly
        upscaled = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_CUBIC)
        
        # Ensure frame is in correct format
        if upscaled.dtype != np.uint8:
            upscaled = upscaled.astype(np.uint8)
        
        # Ensure frame is contiguous in memory
        if not upscaled.flags['C_CONTIGUOUS']:
            upscaled = np.ascontiguousarray(upscaled)
            
        return upscaled
    
    def _should_skip_frame(self, target_fps: int, current_fps: float, frame_processing_time: float, 
                          total_frames: int, frames_skipped: int, max_skip_ratio: float = 0.1) -> bool:
        """
        Determine if a frame should be skipped to maintain target FPS
        
        Args:
            target_fps: Target FPS
            current_fps: Current processing FPS
            frame_processing_time: Time taken to process last frame
            total_frames: Total frames processed so far
            frames_skipped: Number of frames already skipped
            max_skip_ratio: Maximum ratio of frames to skip
            
        Returns:
            True if frame should be skipped
        """
        if not self.stats['adaptive_quality']:
            return False
            
        # Check if we've already reached max skip ratio
        if total_frames > 0:
            current_skip_ratio = frames_skipped / (total_frames + frames_skipped)
            if current_skip_ratio >= max_skip_ratio:
                return False
            
        # Skip frame if we're significantly behind target FPS
        if current_fps < target_fps * 0.7:  # If we're at less than 70% of target
            return True
            
        # Skip frame if processing time is too long
        target_frame_time = 1.0 / target_fps
        if frame_processing_time > target_frame_time * 1.5:  # If processing takes 50% longer than target
            return True
            
        return False
    
    def _get_adaptive_scale(self, target_fps: int, current_fps: float, disable_for_streams: bool = False) -> float:
        """
        Get adaptive scale factor based on performance
        
        Args:
            target_fps: Target FPS
            current_fps: Current processing FPS
            disable_for_streams: If True, disable adaptive scaling for stream outputs
            
        Returns:
            Adaptive scale factor (always returns 1.0 for target resolution mode)
        """
        # In target resolution mode, we don't use adaptive scaling to maintain consistent output
        if disable_for_streams and self.stats['adaptive_quality']:
            logger.debug("Adaptive scaling disabled for stream output to prevent frame size mismatch")
        return 1.0  # No adaptive scaling in target resolution mode
    
    
    def _write_frame_to_ffmpeg(self, frame: np.ndarray, ffmpeg_process: subprocess.Popen, 
                              expected_width: int, expected_height: int, 
                              input_fps: float = 30.0, target_fps: float = 60.0) -> bool:
        """
        Safely write frame to FFmpeg process with proper timing and error handling
        
        Args:
            frame: Frame to write
            ffmpeg_process: FFmpeg subprocess
            expected_width: Expected frame width
            expected_height: Expected frame height
            input_fps: Input frame rate
            target_fps: Target output frame rate
            
        Returns:
            True if write was successful
        """
        try:
            # Check if FFmpeg process is still running
            if ffmpeg_process.poll() is not None:
                logger.error("FFmpeg process terminated unexpectedly")
                return False
            
            # Ensure frame is contiguous in memory
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # Validate frame dimensions before converting to bytes
            frame_height, frame_width = frame.shape[:2]
            if frame_width != expected_width or frame_height != expected_height:
                logger.error(f"Frame dimension mismatch: got {frame_width}x{frame_height}, expected {expected_width}x{expected_height}")
                return False
            
            # Convert frame to bytes
            frame_bytes = frame.tobytes()
            expected_size = expected_width * expected_height * 3  # 3 channels (BGR)
            
            if len(frame_bytes) != expected_size:
                logger.error(f"Frame size mismatch: {len(frame_bytes)} bytes, expected {expected_size}")
                return False
            
            # Write frame data with proper timing control
            try:
                # Send frame once - let FFmpeg handle the FPS conversion
                ffmpeg_process.stdin.write(frame_bytes)
                
                # Flush to ensure data is sent
                ffmpeg_process.stdin.flush()
                
                # Control frame rate to prevent buffer overflow
                current_time = time.time()
                if self.last_frame_time > 0:
                    elapsed = current_time - self.last_frame_time
                    if elapsed < self.frame_interval:
                        time.sleep(self.frame_interval - elapsed)
                
                self.last_frame_time = time.time()
                
                return True
                
            except BrokenPipeError:
                logger.error("Broken pipe: FFmpeg process terminated")
                return False
            except OSError as e:
                logger.error(f"OS error writing to FFmpeg: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error writing to FFmpeg: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error in frame writing: {e}")
            return False
    
    def process_video_file(self, input_path: str, output_path: str, 
                          target_fps: int = 60, real_time: bool = False, 
                          skip_stream_test: bool = False, stream_test_method: str = 'both',
                          max_skip_ratio: float = 0.1, fps_method: str = 'standard', 
                          fps_quality: str = 'high'):
        """
        Process a video file or stream with upscaling
        
        Args:
            input_path: Path to input video file or stream URL
            output_path: Path to output video file or stream URL
            target_fps: Target output FPS
            real_time: Whether to process in real-time mode
            skip_stream_test: Skip stream connection test
            stream_test_method: Stream test method ('ffmpeg', 'opencv', 'both')
            max_skip_ratio: Maximum ratio of frames to skip (0.0-1.0)
        """
        logger.info(f"Processing video: {input_path} -> {output_path}")
        
        # Determine if input is a stream
        is_input_stream = self._is_stream(input_path)
        is_output_stream = self._is_stream(output_path)
        
        if is_input_stream:
            logger.info(f"Input detected as stream: {input_path}")
            
            # Test stream connection first (unless skipped)
            if not skip_stream_test:
                test_success = False
                
                if stream_test_method in ['ffmpeg', 'both']:
                    if self._test_stream_connection_ffmpeg(input_path, timeout=5):
                        test_success = True
                    elif stream_test_method == 'ffmpeg':
                        logger.error("FFmpeg stream connection test failed. Aborting.")
                        return
                
                if not test_success and stream_test_method in ['opencv', 'both']:
                    logger.warning("FFmpeg stream test failed, trying OpenCV test...")
                    if self._test_stream_connection(input_path, timeout=5):
                        test_success = True
                    elif stream_test_method == 'opencv':
                        logger.error("OpenCV stream connection test failed. Aborting.")
                        return
                
                if not test_success:
                    logger.error("All stream connection tests failed. Aborting.")
                    return
            else:
                logger.info("Skipping stream connection test")
            
            if self._is_udp_stream(input_path):
                cap = self._create_udp_capture(input_path)
            else:
                # For other stream types (RTMP, HTTP), use OpenCV with FFmpeg backend
                cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
                # Set timeout for other stream types too
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        
        if not cap.isOpened():
            logger.error(f"Failed to open input: {input_path}")
            if is_input_stream:
                logger.info("Stream connection failed. Please check:")
                logger.info("1. Stream URL is correct and accessible")
                logger.info("2. Network connectivity")
                logger.info("3. Stream is actively broadcasting")
            return
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(original_fps)
        # Calculate output dimensions - always use target resolution
        new_width = self.target_width
        new_height = self.target_height
        scale_factor = self._calculate_scale_factor(width, height)
        logger.info(f"Target resolution mode: {width}x{height} -> {new_width}x{new_height} (scale: {scale_factor:.2f}x)")
        
        logger.info(f"Input: {width}x{height} @ {original_fps:.1f} FPS")
        logger.info(f"Output: {new_width}x{new_height} @ 60 FPS (converted from {target_fps} FPS)")
        if not is_input_stream:
            logger.info(f"Total frames: {total_frames}")
        
        # Setup output writer
        out = None
        udp_writer = None
        
        if is_output_stream:
            logger.info(f"Output detected as stream: {output_path}")
            if self._is_udp_stream(output_path):
                # Try primary UDP writer first - always output 60 FPS
                udp_writer = self._create_udp_writer(input_path, output_path, new_width, new_height, target_fps, 60, 
                                                   fps_method, fps_quality)
                
                if not udp_writer:
                    logger.error(f"Failed to create UDP output stream: {output_path}")
                    return
            else:
                # For other stream types, we'll need to implement specific handlers
                logger.error(f"Stream output type not yet supported: {output_path}")
                return
        else:
            logger.error(f"Failed to create output video: {output_path}")
            return
        
        # Process frames
        frame_count = 0
        frames_skipped = 0
        start_time = time.time()
        last_time = start_time
        self.stats['target_fps'] = target_fps
        
        # try:
        #     while True:
        #         ret, frame = cap.read()
        #         if not ret:
        #             break
                
        #         frame_start = time.time()
                
        #         # Calculate current FPS for adaptive processing
        #         current_time = time.time()
        #         elapsed = current_time - start_time
        #         current_fps = frame_count / elapsed if elapsed > 0 else 0
                
        #         # Check if we should skip this frame
        #         if frame_count > 0:  # Don't skip first frame
        #             last_processing_time = self.stats['processing_times'][-1] if self.stats['processing_times'] else 0
        #             if self._should_skip_frame(target_fps, current_fps, last_processing_time, 
        #                                      frame_count, frames_skipped, max_skip_ratio):
        #                 frames_skipped += 1
        #                 self.stats['frames_dropped'] += 1
        #                 continue
                
        #         # Always use calculated scale factor for consistent output resolution
        #         upscaled_frame = self.upscale_frame(frame, scale_factor)
                
        #         # Record processing time
        #         processing_time = time.time() - frame_start
        #         self.stats['processing_times'].append(processing_time)
                
        #         # Keep only last 30 processing times for rolling average
        #         if len(self.stats['processing_times']) > 30:
        #             self.stats['processing_times'] = self.stats['processing_times'][-30:]
                
        #         # Write frame to output
        #         if udp_writer:
        #             # Write to UDP stream using robust method with proper timing
        #             max_retries = 3
        #             success = False

        #             for retry in range(max_retries):
        #                 if self._write_frame_to_ffmpeg(upscaled_frame, udp_writer, new_width, new_height, 
        #                                              original_fps, 60.0):
        #                     success = True
        #                     break
        #                 else:
        #                     logger.warning(f"Frame write failed, retry {retry + 1}/{max_retries}")
        #                     time.sleep(0.01)  # Small delay before retry
                    
        #             if not success:
        #                 logger.error("Failed to write frame to FFmpeg process after retries")
        #                 break
        #         elif out:
        #             # Write to file
        #             out.write(upscaled_frame)
                
        #         frame_count += 1
        #         self.stats['frames_processed'] += 1
                
        #         # Calculate and log progress with performance metrics
        #         if frame_count % 10 == 0:
        #             current_time = time.time()
        #             elapsed = current_time - start_time
        #             fps = frame_count / elapsed
        #             avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0
                    
        #             progress = (frame_count / total_frames) * 100 if not is_input_stream else 0
        #             eta = (total_frames - frame_count) / fps if fps > 0 and not is_input_stream else 0
                    
        #             logger.info(f"Progress: {progress:.1f}% | "
        #                       f"FPS: {fps:.1f}/{target_fps} | "
        #                       f"Process Time: {avg_processing_time:.3f}s | "
        #                       f"Skipped: {frames_skipped} | "
        #                       f"Frame: {frame_count}")
                
        #         # Real-time mode: maintain target FPS
        #         if real_time:
        #             frame_time = time.time() - frame_start
        #             target_frame_time = 1.0 / target_fps
        #             if frame_time < target_frame_time:
        #                 time.sleep(target_frame_time - frame_time)
                
        # except KeyboardInterrupt:
        #     logger.info("Processing interrupted by user")
        
        # finally:
        #     # Cleanup
        #     cap.release()
            
        #     if out:
        #         out.release()
            
        #     if udp_writer:
        #         try:
        #             # Close stdin to signal end of input
        #             if udp_writer.stdin and not udp_writer.stdin.closed:
        #                 udp_writer.stdin.close()
                    
        #             # Wait for process to finish with timeout
        #             try:
        #                 udp_writer.wait(timeout=5)
        #             except subprocess.TimeoutExpired:
        #                 logger.warning("FFmpeg process did not terminate gracefully, killing...")
        #                 try:
        #                     udp_writer.kill()
        #                     udp_writer.wait(timeout=2)
        #                 except:
        #                     pass
                    
        #             # Check return code
        #             if udp_writer.returncode != 0:
        #                 logger.warning(f"FFmpeg process exited with code {udp_writer.returncode}")
                        
        #         except Exception as e:
        #             logger.warning(f"Error closing UDP writer: {e}")
        #             try:
        #                 if udp_writer.poll() is None:  # Process still running
        #                     udp_writer.kill()
        #                     udp_writer.wait(timeout=1)
        #             except:
        #                 pass
            
        #     # Final statistics
        #     total_time = time.time() - start_time
        #     avg_fps = frame_count / total_time if total_time > 0 else 0
        #     avg_processing_time = sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0
        #     efficiency = (avg_fps / target_fps) * 100 if target_fps > 0 else 0
            
        #     logger.info(f"Processing completed!")
        #     logger.info(f"Total time: {total_time:.1f}s")
        #     logger.info(f"Average FPS: {avg_fps:.1f}/{target_fps} ({efficiency:.1f}% efficiency)")
        #     logger.info(f"Frames processed: {frame_count}")
        #     logger.info(f"Frames skipped: {frames_skipped}")
        #     logger.info(f"Average processing time: {avg_processing_time:.3f}s per frame")
        #     logger.info(f"Target frame time: {1.0/target_fps:.3f}s per frame")
            
        #     if is_output_stream:
        #         logger.info(f"Stream sent to: {output_path}")
        #     else:
        #         logger.info(f"Output saved to: {output_path}")
    
    def process_camera_feed(self, camera_id: int = 0, output_path: Optional[str] = None,
                           target_fps: int = 15):
        """
        Process live camera feed with real-time upscaling
        
        Args:
            camera_id: Camera device ID
            output_path: Optional output file path
            target_fps: Target processing FPS
        """
        logger.info(f"Starting camera feed processing (camera {camera_id})")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        new_width = width * self.scale
        new_height = height * self.scale
        
        logger.info(f"Camera: {width}x{height}")
        logger.info(f"Upscaled: {new_width}x{new_height}")
        
        # Setup output video writer if specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, target_fps, (new_width, new_height))
        
        # Setup display window
        cv2.namedWindow('Real-Time Upscaler', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-Time Upscaler', new_width // 2, new_height // 2)
        
        self.running = True
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                frame_start = time.time()
                
                # Upscale frame
                upscaled_frame = self.upscale_frame(frame)
                
                # Save frame if output specified
                if out:
                    out.write(upscaled_frame)
                
                # Display frame
                cv2.imshow('Real-Time Upscaler', upscaled_frame)
                
                frame_count += 1
                
                # Calculate FPS
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    logger.info(f"Processing FPS: {fps:.1f}")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and output_path:
                    logger.info("Saving current frame...")
                    cv2.imwrite(f"frame_{frame_count}.jpg", upscaled_frame)
                
                # Maintain target FPS
                frame_time = time.time() - frame_start
                target_frame_time = 1.0 / target_fps
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)
        
        except KeyboardInterrupt:
            logger.info("Camera processing interrupted")
        
        finally:
            self.running = False
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            logger.info(f"Camera processing completed!")
            logger.info(f"Total time: {total_time:.1f}s")
            logger.info(f"Average FPS: {avg_fps:.1f}")
            logger.info(f"Frames processed: {frame_count}")


def convert_to_60fps(input_path: str, output_path: str, method: str = 'standard', 
                    quality: str = 'high', progress_callback=None):
    """
    Convert video to 60 FPS using FFmpeg with multiple methods
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        method: Conversion method ('standard', 'interpolation', 'motion_compensated')
        quality: Quality preset ('fast', 'high', 'best')
        progress_callback: Optional callback function for progress updates
    """
    logger.info(f"🎥 Converting to 60 FPS: {input_path} -> {output_path}")
    logger.info(f"Method: {method} | Quality: {quality}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return False
    
    # Build FFmpeg command based on method and quality
    cmd = ['ffmpeg', '-i', input_path]
    
    # Add progress reporting
    cmd.extend(['-progress', 'pipe:1'])
    
    # Video filter based on method
    if method == 'standard':
        # Standard frame duplication/interpolation (fast & reliable)
        cmd.extend(['-filter:v', 'fps=60'])
    elif method == 'interpolation':
        # Better interpolation with motion estimation
        cmd.extend(['-filter:v', 'fps=60:round=near'])
    elif method == 'motion_compensated':
        # Advanced motion-compensated frame interpolation
        cmd.extend(['-filter:v', 'fps=60:round=near:interp=linear'])
    else:
        logger.warning(f"Unknown method '{method}', using standard")
        cmd.extend(['-filter:v', 'fps=60'])
    
    # Quality settings
    if quality == 'fast':
        cmd.extend(['-preset', 'ultrafast', '-crf', '23'])
    elif quality == 'high':
        cmd.extend(['-preset', 'medium', '-crf', '20'])
    elif quality == 'best':
        cmd.extend(['-preset', 'slow', '-crf', '18'])
    else:
        cmd.extend(['-preset', 'medium', '-crf', '20'])
    
    # Audio handling
    cmd.extend(['-c:a', 'copy'])
    
    # Output options
    cmd.extend(['-y', output_path])  # Overwrite output file
    
    logger.info(f"FFmpeg command: {' '.join(cmd)}")
    
    try:
        # Run FFmpeg with progress monitoring
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor progress
        progress_data = {}
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    progress_data[key] = value
                    
                    # Report progress for key metrics
                    if key in ['out_time_ms', 'duration'] and progress_callback:
                        try:
                            current_time = int(progress_data.get('out_time_ms', 0)) / 1000000
                            total_time = int(progress_data.get('duration', 1)) / 1000000
                            if total_time > 0:
                                progress = min(100, (current_time / total_time) * 100)
                                progress_callback(progress)
                        except (ValueError, ZeroDivisionError):
                            pass
        
        # Wait for process to complete
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info("✅ 60 FPS conversion completed successfully!")
            
            # Get file size info
            if os.path.exists(output_path):
                input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
                output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                logger.info(f"📊 File sizes: {input_size:.1f}MB -> {output_size:.1f}MB")
            
            return True
        else:
            logger.error(f"❌ FFmpeg conversion failed with return code {process.returncode}")
            if stderr:
                logger.error(f"Error output: {stderr}")
            return False
            
    except FileNotFoundError:
        logger.error("❌ FFmpeg not found. Please install FFmpeg and add it to PATH")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during conversion: {e}")
        return False

def process_video_file(upscaler, args):
    upscaler.process_video_file(
        input_path=args.input,
        output_path=args.output,
        target_fps=args.fps,
        real_time=args.real_time,
        skip_stream_test=args.skip_stream_test,
        stream_test_method=args.stream_test_method,
        max_skip_ratio=args.max_skip_ratio,
        fps_method=args.fps_method,
        fps_quality=args.fps_quality
    )

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Real-Time Video Upscaling & Enhancement System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process UDP stream to UDP stream with 1920x1080 output (default)
  python realtime_upscaler.py --input udp://192.168.1.100:1234 --output udp://192.168.1.200:5678
  
  # Process with custom target resolution
  python realtime_upscaler.py --input udp://192.168.1.100:1234 --output udp://192.168.1.200:5678 --target-width 1920 --target-height 1080
  
  # Process with custom FPS and real-time mode
  python realtime_upscaler.py --input udp://192.168.1.100:1234 --output udp://192.168.1.200:5678 --fps 30 --real-time
  
  
  # Real-time camera processing
  python realtime_upscaler.py --camera 0 --output camera_output.mp4
  
  # Output automatically converted to 60 FPS (standard method)
  python realtime_upscaler.py --input video.mp4 --output upscaled.mp4
  
  # Output with high quality interpolation to 60 FPS
  python realtime_upscaler.py --input video.mp4 --output upscaled.mp4 --fps-method interpolation --fps-quality best
  
  # Output with motion compensation to 60 FPS (best quality)
  python realtime_upscaler.py --input video.mp4 --output upscaled.mp4 --fps-method motion_compensated --fps-quality best
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str, 
                           help='Input UDP stream URL (format: udp://host:port, e.g., udp://192.168.1.100:1234)')
    input_group.add_argument('--camera', '-c', type=int, default=0, help='Camera device ID')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, 
                       help='Output UDP stream URL (format: udp://host:port, e.g., udp://192.168.1.200:5678)')
    parser.add_argument('--target-width', type=int, default=1920,
                       help='Target output width (default: 1920)')
    parser.add_argument('--target-height', type=int, default=1080,
                       help='Target output height (default: 1080)')
    parser.add_argument('--fps', '-f', type=int, default=25, 
                       help='Target output FPS (default: 25)')
    parser.add_argument('--real-time', '-r', action='store_true',
                       help='Process in real-time mode (slower but maintains timing)')
    
    # Model options
    parser.add_argument('--model', '-m', type=str, default='RealESRGAN_x4plus.pth',
                       help='Path to Real-ESRGAN model file')
    parser.add_argument('--device', '-d', type=str, default='auto', 
                       choices=['auto', 'cuda', 'cpu'],
                       help='Computation device (default: auto)')
    
    # Post-processing options
    parser.add_argument('--convert-60fps', action='store_true',
                       help='Convert output to 60 FPS using FFmpeg')
    parser.add_argument('--skip-stream-test', action='store_true',
                       help='Skip stream connection test (use with caution)')
    parser.add_argument('--stream-test-method', choices=['ffmpeg', 'opencv', 'both'], 
                       default='both', help='Stream connection test method (default: both)')
    parser.add_argument('--no-adaptive', action='store_true',
                       help='Disable adaptive quality and frame skipping (process every frame)')
    parser.add_argument('--max-skip-ratio', type=float, default=0.1,
                       help='Maximum ratio of frames to skip (0.0-1.0, default: 0.1)')
    
    # 60 FPS conversion options
    parser.add_argument('--fps-method', choices=['standard', 'interpolation', 'motion_compensated'], 
                       default='standard', help='60 FPS conversion method (default: standard)')
    parser.add_argument('--fps-quality', choices=['fast', 'high', 'best'], 
                       default='high', help='60 FPS conversion quality (default: high)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input:
        # Only accept UDP stream URLs
        if not args.input.startswith('udp://'):
            logger.error(f"Input must be a UDP stream URL (format: udp://host:port). Received: {args.input}")
            logger.error("Example: udp://192.168.1.100:1234")
            sys.exit(1)
        
        # Validate UDP URL format
        try:
            # Remove protocol prefix
            url = args.input[6:]  # Remove 'udp://'
            if ':' not in url:
                logger.error(f"Invalid UDP URL format. Must include host and port: {args.input}")
                logger.error("Example: udp://192.168.1.100:1234")
                sys.exit(1)
            
            host, port = url.split(':', 1)
            port_num = int(port)
            if not (1 <= port_num <= 65535):
                logger.error(f"Invalid port number: {port}. Port must be between 1 and 65535")
                sys.exit(1)
                
        except ValueError:
            logger.error(f"Invalid UDP URL format. Port must be a number: {args.input}")
            logger.error("Example: udp://192.168.1.100:1234")
            sys.exit(1)
    
    # Validate output argument
    if args.output:
        # Only accept UDP stream URLs for output
        if not args.output.startswith('udp://'):
            logger.error(f"Output must be a UDP stream URL (format: udp://host:port). Received: {args.output}")
            logger.error("Example: udp://192.168.1.200:5678")
            sys.exit(1)
        
        # Validate UDP URL format for output
        try:
            # Remove protocol prefix
            url = args.output[6:]  # Remove 'udp://'
            if ':' not in url:
                logger.error(f"Invalid UDP URL format. Must include host and port: {args.output}")
                logger.error("Example: udp://192.168.1.200:5678")
                sys.exit(1)
            
            host, port = url.split(':', 1)
            port_num = int(port)
            if not (1 <= port_num <= 65535):
                logger.error(f"Invalid port number: {port}. Port must be between 1 and 65535")
                sys.exit(1)
                
        except ValueError:
            logger.error(f"Invalid UDP URL format. Port must be a number: {args.output}")
            logger.error("Example: udp://192.168.1.200:5678")
            sys.exit(1)
    
    if not args.output and not args.camera:
        logger.error("Output UDP stream URL required")
        sys.exit(1)
    
    # Initialize upscaler
    try:
        upscaler = RealTimeUpscaler(
            model_path=args.model,
            device=args.device,
            target_width=args.target_width,
            target_height=args.target_height
        )
        
        # Configure adaptive processing
        upscaler.stats['adaptive_quality'] = not args.no_adaptive
        if args.no_adaptive:
            logger.info("Adaptive quality processing disabled - will process every frame")
        else:
            logger.info(f"Adaptive quality processing enabled - max skip ratio: {args.max_skip_ratio}")
            
    except Exception as e:
        logger.error(f"Failed to initialize upscaler: {e}")
        sys.exit(1)
    
    # Process based on input type
    try:
        if args.input:
            # Process video file
            process_thread = threading.Thread(target=  process_video_file, args=(upscaler, args))
            process_thread.daemon = True
            process_thread.start()
            process_thread.join()
        else:
            logger.error(f"Processing failed: {e}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
