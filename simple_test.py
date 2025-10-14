#!/usr/bin/env python3
"""
Simple System Test Script
========================
"""

import sys
import os

def test_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("[WARN] CUDA not available, using CPU")
            
    except ImportError as e:
        print(f"[ERROR] PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"[OK] OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"[ERROR] OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"[OK] NumPy {np.__version__}")
    except ImportError as e:
        print(f"[ERROR] NumPy import failed: {e}")
        return False
    
    return True

def test_realesrgan():
    """Test Real-ESRGAN import"""
    print("\nTesting Real-ESRGAN import...")
    
    try:
        from realesrgan import RealESRGANer
        print("[OK] Real-ESRGAN imported successfully")
        return True
    except ImportError as e:
        print(f"[WARN] Real-ESRGAN import failed: {e}")
        print("System will use OpenCV fallback for upscaling")
        return False

def test_upscaler():
    """Test the upscaler class"""
    print("\nTesting RealTimeUpscaler class...")
    
    try:
        from realtime_upscaler import RealTimeUpscaler
        print("[OK] RealTimeUpscaler imported successfully")
        
        # Test initialization
        upscaler = RealTimeUpscaler(
            model_path="nonexistent_model.pth",
            device="cpu",
            target_width=640,
            target_height=480
        )
        print("[OK] RealTimeUpscaler initialized successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] RealTimeUpscaler test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Real-Time Video Upscaling System - Test")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Real-ESRGAN", test_realesrgan),
        ("Upscaler Class", test_upscaler),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! System is ready.")
    elif passed >= total - 1:
        print("System is mostly functional.")
    else:
        print("System has issues. Check installation.")

if __name__ == "__main__":
    main()
