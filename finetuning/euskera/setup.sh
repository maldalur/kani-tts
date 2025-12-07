#!/bin/bash

# Setup script for Basque (Euskera) TTS Finetuning
# This script installs all required dependencies

set -e

echo "=========================================="
echo "Kani TTS Basque Finetuning Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

required_version="3.10"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.10 or higher is required"
    exit 1
fi

echo "✓ Python version OK"
echo ""

# Check if virtual environment is recommended
echo "It's recommended to use a virtual environment."
read -p "Create and activate a virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

echo ""
echo "Installing dependencies..."
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CUDA version - adjust if needed)
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and datasets
echo ""
echo "Installing Transformers and Datasets..."
pip install transformers>=4.35.0
pip install datasets>=2.14.0
pip install accelerate>=0.24.0

# Install audio processing libraries
echo ""
echo "Installing audio processing libraries..."
pip install librosa>=0.10.0
pip install soundfile>=0.12.0

# Install NeMo toolkit
echo ""
echo "Installing NeMo toolkit..."
pip install nemo_toolkit[tts]>=1.20.0

# Install data processing libraries
echo ""
echo "Installing data processing libraries..."
pip install pandas>=2.0.0
pip install numpy>=1.24.0

# Install training utilities
echo ""
echo "Installing training utilities..."
pip install tensorboard>=2.14.0
pip install safetensors>=0.4.0
pip install tqdm>=4.65.0
pip install pyyaml>=6.0

echo ""
echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Prepare your Basque audio dataset"
echo "2. Create metadata.csv file"
echo "3. Run: python finetune_euskera.py --help"
echo ""
echo "For detailed instructions, see README.md"
echo ""
