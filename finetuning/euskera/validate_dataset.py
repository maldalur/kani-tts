#!/usr/bin/env python3
"""
Dataset validation script for Basque TTS finetuning

This script validates your dataset before training to catch common issues.

Usage:
    python validate_dataset.py --metadata data/euskera/metadata.csv
"""

import os
import argparse
import pandas as pd
import logging
from pathlib import Path
from typing import List, Tuple

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not installed. Audio validation will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validator for TTS training datasets."""
    
    # Configuration constants
    AUDIO_CHECK_SAMPLE_SIZE = 10  # Number of audio files to check for properties
    
    def __init__(self, metadata_file: str):
        """
        Initialize validator.
        
        Args:
            metadata_file: Path to metadata CSV file
        """
        self.metadata_file = metadata_file
        self.metadata_dir = os.path.dirname(os.path.abspath(metadata_file))
        self.errors = []
        self.warnings = []
        
    def validate(self) -> bool:
        """
        Run all validations.
        
        Returns:
            True if validation passes, False otherwise
        """
        logger.info("=" * 80)
        logger.info("Starting Dataset Validation")
        logger.info("=" * 80)
        
        # Check metadata file exists
        if not self._validate_metadata_file():
            return False
        
        # Load metadata
        try:
            self.df = pd.read_csv(self.metadata_file)
        except Exception as e:
            self.errors.append(f"Failed to read metadata file: {e}")
            return False
        
        # Run validations
        self._validate_metadata_format()
        self._validate_audio_files()
        self._validate_transcripts()
        
        if LIBROSA_AVAILABLE:
            self._validate_audio_properties()
        else:
            self.warnings.append(
                "librosa not available - skipping audio property validation"
            )
        
        # Print results
        self._print_results()
        
        return len(self.errors) == 0
    
    def _validate_metadata_file(self) -> bool:
        """Check if metadata file exists."""
        if not os.path.exists(self.metadata_file):
            self.errors.append(f"Metadata file not found: {self.metadata_file}")
            return False
        return True
    
    def _validate_metadata_format(self):
        """Validate metadata CSV format."""
        logger.info("Validating metadata format...")
        
        # Check required columns
        required_columns = ['audio_path', 'transcript']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            self.errors.append(
                f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Check for empty values
        if self.df['audio_path'].isna().any():
            self.errors.append("Found empty audio_path values")
        
        if self.df['transcript'].isna().any():
            self.errors.append("Found empty transcript values")
        
        logger.info(f"  Total samples: {len(self.df)}")
        
        if len(self.df) < 10:
            self.warnings.append(
                f"Very small dataset ({len(self.df)} samples). "
                "Consider collecting more data for better results."
            )
        elif len(self.df) < 50:
            self.warnings.append(
                f"Small dataset ({len(self.df)} samples). "
                "50-100+ samples recommended for better quality."
            )
    
    def _validate_audio_files(self):
        """Validate audio file paths and existence."""
        logger.info("Validating audio files...")
        
        missing_files = []
        invalid_extensions = []
        
        for idx, row in self.df.iterrows():
            audio_path = row['audio_path']
            
            # Make path absolute if relative
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(self.metadata_dir, audio_path)
            
            # Check if file exists
            if not os.path.exists(audio_path):
                missing_files.append(row['audio_path'])
                continue
            
            # Check file extension
            ext = os.path.splitext(audio_path)[1].lower()
            if ext not in ['.wav', '.mp3', '.flac', '.ogg']:
                invalid_extensions.append((row['audio_path'], ext))
        
        if missing_files:
            self.errors.append(
                f"Missing audio files ({len(missing_files)}): "
                f"{', '.join(missing_files[:5])}"
                + ("..." if len(missing_files) > 5 else "")
            )
        
        if invalid_extensions:
            self.warnings.append(
                f"Non-WAV files detected ({len(invalid_extensions)}). "
                "WAV format is recommended. Files: "
                f"{', '.join([f[0] for f in invalid_extensions[:3]])}"
            )
        
        valid_files = len(self.df) - len(missing_files)
        logger.info(f"  Valid audio files: {valid_files}/{len(self.df)}")
    
    def _validate_transcripts(self):
        """Validate transcript text."""
        logger.info("Validating transcripts...")
        
        # Check transcript lengths
        empty_transcripts = self.df[self.df['transcript'].str.len() == 0]
        if len(empty_transcripts) > 0:
            self.errors.append(
                f"Found {len(empty_transcripts)} empty transcripts"
            )
        
        # Check for very short transcripts
        short_transcripts = self.df[self.df['transcript'].str.len() < 5]
        if len(short_transcripts) > 0:
            self.warnings.append(
                f"Found {len(short_transcripts)} very short transcripts (< 5 chars)"
            )
        
        # Check for very long transcripts
        long_transcripts = self.df[self.df['transcript'].str.len() > 500]
        if len(long_transcripts) > 0:
            self.warnings.append(
                f"Found {len(long_transcripts)} very long transcripts (> 500 chars). "
                "Consider splitting into shorter segments."
            )
        
        # Statistics
        avg_length = self.df['transcript'].str.len().mean()
        logger.info(f"  Average transcript length: {avg_length:.1f} characters")
    
    def _validate_audio_properties(self):
        """Validate audio file properties using librosa."""
        logger.info("Validating audio properties...")
        
        sample_rates = []
        durations = []
        errors = []
        
        # Sample up to AUDIO_CHECK_SAMPLE_SIZE files for validation
        sample_size = min(self.AUDIO_CHECK_SAMPLE_SIZE, len(self.df))
        sample_indices = self.df.sample(n=sample_size).index
        
        for idx in sample_indices:
            audio_path = self.df.loc[idx, 'audio_path']
            
            # Make path absolute
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(self.metadata_dir, audio_path)
            
            if not os.path.exists(audio_path):
                continue
            
            try:
                # Load audio
                y, sr = librosa.load(audio_path, sr=None)
                duration = len(y) / sr
                
                sample_rates.append(sr)
                durations.append(duration)
            except Exception as e:
                errors.append(f"{audio_path}: {str(e)}")
        
        if errors:
            self.warnings.append(
                f"Failed to load {len(errors)} audio files: {errors[0]}"
            )
        
        if sample_rates:
            # Check sample rates
            unique_srs = set(sample_rates)
            if len(unique_srs) > 1:
                self.warnings.append(
                    f"Multiple sample rates detected: {unique_srs}. "
                    "Will be resampled to 22050 Hz."
                )
            
            avg_sr = sum(sample_rates) / len(sample_rates)
            logger.info(f"  Average sample rate: {avg_sr:.0f} Hz")
            
            # Check durations
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
            
            logger.info(f"  Average duration: {avg_duration:.2f}s")
            logger.info(f"  Duration range: {min_duration:.2f}s - {max_duration:.2f}s")
            
            if max_duration > 30:
                self.warnings.append(
                    f"Some audio files are very long (>{max_duration:.1f}s). "
                    "Will be truncated to 30s."
                )
            
            if min_duration < 1:
                self.warnings.append(
                    f"Some audio files are very short (<{min_duration:.1f}s). "
                    "Consider removing or replacing them."
                )
    
    def _print_results(self):
        """Print validation results."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("Validation Results")
        logger.info("=" * 80)
        
        if self.errors:
            logger.error(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        if self.warnings:
            logger.warning(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        if not self.errors and not self.warnings:
            logger.info("\n✅ ALL CHECKS PASSED!")
            logger.info("Your dataset is ready for training.")
        elif not self.errors:
            logger.info("\n✅ VALIDATION PASSED (with warnings)")
            logger.info("You can proceed with training, but review the warnings above.")
        else:
            logger.error("\n❌ VALIDATION FAILED")
            logger.error("Please fix the errors above before training.")
        
        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate dataset for Basque TTS finetuning"
    )
    
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata CSV file"
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = DatasetValidator(args.metadata)
    success = validator.validate()
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
