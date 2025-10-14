#!/usr/bin/env python3
"""
Setup script for Real-Time Video Upscaling & Enhancement System
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Real-Time Video Upscaling & Enhancement System using Real-ESRGAN"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="realtime-video-upscaler",
    version="1.0.0",
    author="AI Assistant",
    author_email="ai@example.com",
    description="Real-Time Video Upscaling & Enhancement System using Real-ESRGAN",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/realtime-video-upscaler",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch-audio>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "realtime-upscaler=realtime_upscaler:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.pth", "*.onnx", "*.txt", "*.md"],
    },
    keywords="video upscaling, super-resolution, real-esrgan, ai, computer-vision, pytorch",
    project_urls={
        "Bug Reports": "https://github.com/example/realtime-video-upscaler/issues",
        "Source": "https://github.com/example/realtime-video-upscaler",
        "Documentation": "https://github.com/example/realtime-video-upscaler#readme",
    },
)
