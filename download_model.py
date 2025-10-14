#!/usr/bin/env python3
"""
Model Download Script for Real-Time Video Upscaling System
=========================================================

This script downloads the required Real-ESRGAN model weights.
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model information
MODELS = {
    "RealESRGAN_x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "size": "67.4 MB",
        "md5": "99ec365d4afad750833258a1a24f44ca"
    },
    "RealESRGAN_x4plus_anime_6B": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth", 
        "size": "17.1 MB",
        "md5": "d58ce384064ec1591c2ea7b79dbf47ba"
    }
}

def download_file(url: str, filename: str, expected_md5: str = None) -> bool:
    """
    Download a file with progress bar and MD5 verification
    
    Args:
        url: URL to download from
        filename: Local filename to save to
        expected_md5: Expected MD5 hash for verification
        
    Returns:
        True if download successful, False otherwise
    """
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) / total_size)
            sys.stdout.write(f"\rDownloading {filename}: {percent:.1f}% ({downloaded/1024/1024:.1f}MB/{total_size/1024/1024:.1f}MB)")
            sys.stdout.flush()
    
    try:
        logger.info(f"Downloading {filename} from {url}")
        urllib.request.urlretrieve(url, filename, progress_hook)
        print()  # New line after progress
        
        # Verify MD5 if provided
        if expected_md5:
            logger.info("Verifying file integrity...")
            with open(filename, 'rb') as f:
                file_md5 = hashlib.md5(f.read()).hexdigest()
            
            if file_md5.lower() != expected_md5.lower():
                logger.error(f"MD5 mismatch! Expected: {expected_md5}, Got: {file_md5}")
                os.remove(filename)
                return False
            else:
                logger.info("[OK] File integrity verified")
        
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

def main():
    """Main function"""
    print("[INFO] Real-Time Video Upscaling & Enhancement System")
    print("[INFO] Model Download Script")
    print("=" * 50)
    
    # Check if models already exist
    existing_models = []
    for model_name, model_info in MODELS.items():
        if os.path.exists(model_info["filename"]):
            existing_models.append(model_name)
            logger.info(f"[OK] {model_info['filename']} already exists")
    
    if existing_models:
        print(f"\nFound {len(existing_models)} existing model(s).")
        response = input("Do you want to re-download them? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("Skipping download.")
            return
    
    # Download models
    success_count = 0
    for model_name, model_info in MODELS.items():
        print(f"\n[DOWNLOAD] Downloading {model_name} ({model_info['size']})")
        print(f"URL: {model_info['url']}")
        
        if download_file(
            model_info["url"], 
            model_info["filename"], 
            model_info["md5"]
        ):
            success_count += 1
            logger.info(f"[OK] {model_name} downloaded successfully")
        else:
            logger.error(f"[ERROR] Failed to download {model_name}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Download Summary: {success_count}/{len(MODELS)} models downloaded")
    
    if success_count == len(MODELS):
        print("[SUCCESS] All models downloaded successfully!")
        print("\nYou can now run the upscaler:")
        print("python realtime_upscaler.py --input your_video.mp4 --output upscaled.mp4")
    else:
        print("[WARNING] Some downloads failed. Please check your internet connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
