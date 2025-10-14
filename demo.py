#!/usr/bin/env python3
"""
Demo Script for Real-Time Video Upscaling System
===============================================

This script demonstrates the upscaling system working with OpenCV fallback.
"""

import cv2
import numpy as np
import time
import os
from realtime_upscaler import RealTimeUpscaler

def create_test_video():
    """Create a simple test video"""
    print("Creating test video...")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_input.mp4', fourcc, 10.0, (320, 240))
    
    for i in range(50):  # 5 seconds at 10 FPS
        # Create a frame with moving pattern
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Add moving circle
        center_x = 160 + int(100 * np.sin(i * 0.2))
        center_y = 120 + int(50 * np.cos(i * 0.2))
        cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), -1)
        
        # Add text
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Test Video", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add some noise for detail
        noise = np.random.randint(0, 50, (240, 320, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
    
    out.release()
    print("Test video created: test_input.mp4")

def demo_video_processing():
    """Demonstrate video processing with upscaling"""
    print("\n=== Video Processing Demo ===")
    
    # Create test video if it doesn't exist
    if not os.path.exists('test_input.mp4'):
        create_test_video()
    
    # Initialize upscaler (will use OpenCV fallback)
    print("Initializing upscaler...")
    upscaler = RealTimeUpscaler(
        model_path="nonexistent_model.pth",  # This will trigger fallback
        device="auto",
        target_width=640,  # 2x upscaling from 320x240
        target_height=480
    )
    
    # Process the video
    print("Processing video with 2x upscaling...")
    upscaler.process_video_file(
        input_path="test_input.mp4",
        output_path="test_output_2x.mp4",
        target_fps=10,
        real_time=False
    )
    
    print("Video processing completed!")
    print("Input: test_input.mp4 (320x240)")
    print("Output: test_output_2x.mp4 (640x480)")

def demo_frame_processing():
    """Demonstrate frame-by-frame processing"""
    print("\n=== Frame Processing Demo ===")
    
    # Initialize upscaler
    upscaler = RealTimeUpscaler(
        model_path="nonexistent_model.pth",
        device="auto",
        target_width=400,  # 4x upscaling from 100x100
        target_height=400
    )
    
    # Create a test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Add some pattern
    cv2.circle(test_image, (50, 50), 30, (255, 255, 255), -1)
    cv2.putText(test_image, "TEST", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Save original
    cv2.imwrite("original_frame.jpg", test_image)
    print("Original frame saved: original_frame.jpg (100x100)")
    
    # Upscale frame
    print("Upscaling frame with 4x scale...")
    start_time = time.time()
    upscaled_frame = upscaler.upscale_frame(test_image)
    end_time = time.time()
    
    # Save upscaled
    cv2.imwrite("upscaled_frame.jpg", upscaled_frame)
    
    print(f"Upscaled frame saved: upscaled_frame.jpg ({upscaled_frame.shape[1]}x{upscaled_frame.shape[0]})")
    print(f"Processing time: {end_time - start_time:.4f} seconds")

def demo_camera_processing():
    """Demonstrate camera processing (if camera is available)"""
    print("\n=== Camera Processing Demo ===")
    print("This demo will try to access your camera...")
    print("Press 'q' to quit, 's' to save current frame")
    
    # Initialize upscaler
    upscaler = RealTimeUpscaler(
        model_path="nonexistent_model.pth",
        device="auto",
        target_width=640,  # 2x upscaling from 320x240
        target_height=480
    )
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera available. Skipping camera demo.")
        return
    
    print("Camera opened successfully!")
    print("Starting real-time upscaling...")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize input for faster processing
            small_frame = cv2.resize(frame, (320, 240))
            
            # Upscale
            upscaled_frame = upscaler.upscale_frame(small_frame)
            
            # Display
            cv2.imshow('Real-Time Upscaler Demo', upscaled_frame)
            
            frame_count += 1
            
            # Show FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Processing FPS: {fps:.1f}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"camera_frame_{frame_count}.jpg", upscaled_frame)
                print(f"Frame saved: camera_frame_{frame_count}.jpg")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"Camera demo completed!")
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {avg_fps:.1f}")

def main():
    """Main demo function"""
    print("Real-Time Video Upscaling System - Demo")
    print("=" * 40)
    print("This demo shows the system working with OpenCV fallback")
    print("(Real-ESRGAN is optional and will be used if available)")
    print()
    
    demos = {
        "1": ("Frame Processing", demo_frame_processing),
        "2": ("Video Processing", demo_video_processing),
        "3": ("Camera Processing", demo_camera_processing),
    }
    
    print("Available demos:")
    for key, (name, _) in demos.items():
        print(f"  {key}. {name}")
    print("  'all' - Run all demos")
    print("  'q' - Quit")
    
    while True:
        choice = input("\nSelect a demo (1-3, 'all', or 'q'): ").strip().lower()
        
        if choice == 'q':
            print("Goodbye!")
            break
        elif choice == 'all':
            print("\nRunning all demos...")
            for key, (name, func) in demos.items():
                print(f"\n--- {name} ---")
                try:
                    func()
                except KeyboardInterrupt:
                    print("\nDemo interrupted by user")
                    break
                except Exception as e:
                    print(f"Demo failed: {e}")
            break
        elif choice in demos:
            name, func = demos[choice]
            print(f"\nRunning: {name}")
            try:
                func()
            except KeyboardInterrupt:
                print("\nDemo interrupted by user")
            except Exception as e:
                print(f"Demo failed: {e}")
        else:
            print("Invalid choice. Please select 1-3, 'all', or 'q'")

if __name__ == "__main__":
    main()
