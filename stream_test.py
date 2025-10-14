#!/usr/bin/env python3
"""
Stream Testing Utility for Real-Time Video Upscaling System
==========================================================

This script provides utilities for testing UDP streams and stream processing.
"""

import cv2
import numpy as np
import time
import socket
import threading
import subprocess
import sys
import argparse
from realtime_upscaler import RealTimeUpscaler

def create_test_udp_stream(host="127.0.0.1", port=1234, duration=30):
    """
    Create a test UDP video stream using FFmpeg
    
    Args:
        host: Target host
        port: Target port
        duration: Stream duration in seconds
    """
    print(f"Creating test UDP stream to {host}:{port} for {duration} seconds...")
    
    # FFmpeg command to create test pattern and stream it
    cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'testsrc=duration={duration}:size=640x480:rate=25',
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-f', 'mpegts',
        f'udp://{host}:{port}',
        '-y'
    ]
    
    try:
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode == 0:
            print("Test stream created successfully!")
        else:
            print(f"Error creating stream: {process.stderr}")
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg to create test streams.")
    except Exception as e:
        print(f"Error creating test stream: {e}")

def receive_udp_stream(host="127.0.0.1", port=1234, duration=10):
    """
    Receive and display a UDP video stream
    
    Args:
        host: Source host
        port: Source port
        duration: Receive duration in seconds
    """
    print(f"Receiving UDP stream from {host}:{port} for {duration} seconds...")
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(1.0)  # 1 second timeout
    
    print("Waiting for stream...")
    
    # Use OpenCV to receive the stream
    stream_url = f"udp://{host}:{port}"
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("Failed to open UDP stream")
        return
    
    print("Stream opened successfully!")
    print("Press 'q' to quit")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                print("No frame received")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Display frame
            cv2.imshow('UDP Stream Receiver', frame)
            
            # Show FPS
            if frame_count % 25 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f}")
            
            # Handle key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nStream reception interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"Stream reception completed!")
        print(f"Total frames: {frame_count}")
        print(f"Average FPS: {avg_fps:.1f}")

def test_stream_processing(input_stream, output_stream, scale=2, duration=10):
    """
    Test stream processing with upscaling
    
    Args:
        input_stream: Input stream URL
        output_stream: Output stream URL
        scale: Upscaling factor
        duration: Test duration in seconds
    """
    print(f"Testing stream processing: {input_stream} -> {output_stream}")
    print(f"Scale: {scale}x, Duration: {duration}s")
    
    # Initialize upscaler
    upscaler = RealTimeUpscaler(
        model_path="nonexistent_model.pth",  # Use fallback
        device="auto",
        target_width=1280,  # Default target resolution
        target_height=720
    )
    
    # Process the stream
    start_time = time.time()
    
    try:
        # This will run until interrupted or stream ends
        upscaler.process_video_file(
            input_path=input_stream,
            output_path=output_stream,
            target_fps=25,
            real_time=True
        )
    except KeyboardInterrupt:
        print("\nStream processing interrupted")
    
    elapsed = time.time() - start_time
    print(f"Stream processing completed in {elapsed:.1f} seconds")

def test_file_to_stream(input_file, output_stream, scale=2):
    """
    Test processing a video file and outputting to a stream
    
    Args:
        input_file: Input video file path
        output_stream: Output stream URL
        scale: Upscaling factor
    """
    print(f"Testing file to stream: {input_file} -> {output_stream}")
    
    # Initialize upscaler
    upscaler = RealTimeUpscaler(
        model_path="nonexistent_model.pth",
        device="auto",
        target_width=1280,  # Default target resolution
        target_height=720
    )
    
    # Process the file
    upscaler.process_video_file(
        input_path=input_file,
        output_path=output_stream,
        target_fps=25,
        real_time=False
    )

def test_stream_to_file(input_stream, output_file, scale=2, duration=10):
    """
    Test processing a stream and saving to a file
    
    Args:
        input_stream: Input stream URL
        output_file: Output file path
        scale: Upscaling factor
        duration: Recording duration
    """
    print(f"Testing stream to file: {input_stream} -> {output_file}")
    
    # Initialize upscaler
    upscaler = RealTimeUpscaler(
        model_path="nonexistent_model.pth",
        device="auto",
        target_width=1280,  # Default target resolution
        target_height=720
    )
    
    # Process the stream
    upscaler.process_video_file(
        input_path=input_stream,
        output_path=output_file,
        target_fps=25,
        real_time=True
    )

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Stream Testing Utility for Real-Time Video Upscaling System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a test UDP stream
  python stream_test.py create-stream --host 192.168.1.100 --port 1234
  
  # Receive and display a UDP stream
  python stream_test.py receive-stream --host 192.168.1.100 --port 1234
  
  # Test stream processing (input stream -> output stream)
  python stream_test.py test-processing --input udp://192.168.1.100:1234 --output udp://192.168.1.200:5678
  
  # Test file to stream processing
  python stream_test.py file-to-stream --input video.mp4 --output udp://192.168.1.200:5678
  
  # Test stream to file processing
  python stream_test.py stream-to-file --input udp://192.168.1.100:1234 --output output.mp4
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create stream command
    create_parser = subparsers.add_parser('create-stream', help='Create a test UDP stream')
    create_parser.add_argument('--host', default='127.0.0.1', help='Target host (default: 127.0.0.1)')
    create_parser.add_argument('--port', type=int, default=1234, help='Target port (default: 1234)')
    create_parser.add_argument('--duration', type=int, default=30, help='Stream duration in seconds (default: 30)')
    
    # Receive stream command
    receive_parser = subparsers.add_parser('receive-stream', help='Receive and display a UDP stream')
    receive_parser.add_argument('--host', default='127.0.0.1', help='Source host (default: 127.0.0.1)')
    receive_parser.add_argument('--port', type=int, default=1234, help='Source port (default: 1234)')
    receive_parser.add_argument('--duration', type=int, default=10, help='Receive duration in seconds (default: 10)')
    
    # Test processing command
    test_parser = subparsers.add_parser('test-processing', help='Test stream processing')
    test_parser.add_argument('--input', required=True, help='Input stream URL')
    test_parser.add_argument('--output', required=True, help='Output stream URL')
    test_parser.add_argument('--scale', type=int, default=2, help='Upscaling factor (default: 2)')
    test_parser.add_argument('--duration', type=int, default=10, help='Test duration in seconds (default: 10)')
    
    # File to stream command
    file_to_stream_parser = subparsers.add_parser('file-to-stream', help='Process file and output to stream')
    file_to_stream_parser.add_argument('--input', required=True, help='Input video file')
    file_to_stream_parser.add_argument('--output', required=True, help='Output stream URL')
    file_to_stream_parser.add_argument('--scale', type=int, default=2, help='Upscaling factor (default: 2)')
    
    # Stream to file command
    stream_to_file_parser = subparsers.add_parser('stream-to-file', help='Process stream and save to file')
    stream_to_file_parser.add_argument('--input', required=True, help='Input stream URL')
    stream_to_file_parser.add_argument('--output', required=True, help='Output video file')
    stream_to_file_parser.add_argument('--scale', type=int, default=2, help='Upscaling factor (default: 2)')
    stream_to_file_parser.add_argument('--duration', type=int, default=10, help='Recording duration in seconds (default: 10)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'create-stream':
            create_test_udp_stream(args.host, args.port, args.duration)
        elif args.command == 'receive-stream':
            receive_udp_stream(args.host, args.port, args.duration)
        elif args.command == 'test-processing':
            test_stream_processing(args.input, args.output, args.scale, args.duration)
        elif args.command == 'file-to-stream':
            test_file_to_stream(args.input, args.output, args.scale)
        elif args.command == 'stream-to-file':
            test_stream_to_file(args.input, args.output, args.scale, args.duration)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
