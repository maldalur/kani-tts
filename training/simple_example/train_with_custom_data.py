#!/usr/bin/env python3
"""
Train Kani TTS with Custom Dataset

This script shows how to adapt the training pipeline for your own audio data.
"""

import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_dataset(audio_dir, metadata_file):
    """Validate that the dataset is properly formatted."""
    
    if not os.path.exists(audio_dir):
        logger.error(f"Audio directory not found: {audio_dir}")
        return False
    
    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file not found: {metadata_file}")
        return False
    
    import pandas as pd
    
    try:
        df = pd.read_csv(metadata_file)
        
        # Check required columns
        if 'audio_path' not in df.columns or 'transcript' not in df.columns:
            logger.error("Metadata file must have 'audio_path' and 'transcript' columns")
            return False
        
        # Check audio files exist
        missing_files = []
        for audio_path in df['audio_path']:
            full_path = os.path.join(os.path.dirname(metadata_file), audio_path)
            if not os.path.exists(full_path):
                missing_files.append(audio_path)
        
        if missing_files:
            logger.error(f"Found {len(missing_files)} missing audio files")
            for f in missing_files[:5]:
                logger.error(f"  - {f}")
            if len(missing_files) > 5:
                logger.error(f"  ... and {len(missing_files) - 5} more")
            return False
        
        logger.info(f"✓ Dataset validation passed: {len(df)} samples found")
        return True
        
    except Exception as e:
        logger.error(f"Error validating dataset: {e}")
        return False


def train_with_custom_data(args):
    """Train model with custom dataset."""
    
    logger.info("=" * 80)
    logger.info("Training Kani TTS with Custom Data")
    logger.info("=" * 80)
    
    # Validate dataset
    logger.info("\nValidating dataset...")
    if not validate_dataset(args.audio_dir, args.metadata):
        logger.error("Dataset validation failed. Please fix the issues and try again.")
        return 1
    
    # Import training components
    from train_model import SimpleTTSTrainer, SimpleTrainingConfig
    
    # Create configuration with custom settings
    config = SimpleTrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
    )
    
    # Note: You would need to modify SimpleTTSTrainer to load real audio
    # This is a template showing the structure
    logger.info("\n⚠️  Note: The train_model.py script needs to be modified to:")
    logger.info("   1. Load real audio files (using librosa)")
    logger.info("   2. Encode audio with NeMo NanoCodec")
    logger.info("   3. Process your metadata file")
    logger.info("\nFor full implementation, see:")
    logger.info("  - finetuning/euskera/finetune_euskera.py (complete example)")
    logger.info("  - https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline")
    
    logger.info("\nDataset Information:")
    logger.info(f"  Audio directory: {args.audio_dir}")
    logger.info(f"  Metadata file: {args.metadata}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    
    return 0


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Train Kani TTS with custom audio dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Prepare your dataset:
  data/
  ├── audio/
  │   ├── sample_001.wav
  │   ├── sample_002.wav
  │   └── ...
  └── metadata.csv

  # metadata.csv format:
  audio_path,transcript
  audio/sample_001.wav,Hello world
  audio/sample_002.wav,This is a test

  # Train the model:
  python train_with_custom_data.py \\
    --audio_dir data/audio \\
    --metadata data/metadata.csv \\
    --output_dir models/my-trained-model \\
    --epochs 5 \\
    --batch_size 4

For complete implementation, see:
  - finetuning/euskera/finetune_euskera.py
  - https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline
        """
    )
    
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files"
    )
    
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="CSV file with audio_path and transcript columns"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/custom-trained-model",
        help="Output directory for trained model"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size per device"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for testing)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.audio_dir):
        logger.error(f"Audio directory does not exist: {args.audio_dir}")
        return 1
    
    if not os.path.exists(args.metadata):
        logger.error(f"Metadata file does not exist: {args.metadata}")
        return 1
    
    # Train
    return train_with_custom_data(args)


if __name__ == "__main__":
    exit(main())
