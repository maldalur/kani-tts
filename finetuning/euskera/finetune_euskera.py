#!/usr/bin/env python3
"""
Kani TTS Finetuning Script for Basque (Euskera)

This script provides a complete pipeline for finetuning the Kani TTS model
on Basque language audio data. It follows the dataset preparation guidelines
suggested by the repository.

Usage:
    python finetune_euskera.py --config config.yaml
"""

import os
import json
import torch
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, load_dataset
import numpy as np
from nemo.collections.tts.models import AudioCodecModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FinetuningConfig:
    """Configuration for finetuning Kani TTS model."""
    
    # Model configuration
    base_model_name: str = "nineninesix/kani-tts-400m-0.3-pt"
    codec_model_name: str = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
    
    # Dataset paths
    dataset_path: str = "data/euskera"
    audio_dir: str = "data/euskera/audio"
    transcripts_file: str = "data/euskera/metadata.csv"
    
    # Output configuration
    output_dir: str = "models/kani-tts-euskera"
    checkpoint_dir: str = "checkpoints/euskera"
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_seq_length: int = 1024
    
    # Optimization
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Data processing
    sample_rate: int = 22050
    max_audio_duration: float = 30.0  # seconds
    
    # Tokenizer settings
    tokenizer_length: int = 64400
    start_of_text: int = 1
    start_of_audio: int = 64000
    end_of_audio: int = 64399
    audio_token_range: tuple = (64001, 64399)


class AudioDatasetPreprocessor:
    """Preprocessor for audio datasets following nano-codec-dataset-pipeline guidelines."""
    
    def __init__(self, config: FinetuningConfig):
        self.config = config
        self.codec_model = None
        
    def load_codec_model(self):
        """Load the NeMo NanoCodec model for audio encoding."""
        logger.info(f"Loading codec model: {self.config.codec_model_name}")
        self.codec_model = AudioCodecModel.from_pretrained(
            self.config.codec_model_name
        )
        self.codec_model = self.codec_model.eval()
        if torch.cuda.is_available():
            self.codec_model = self.codec_model.cuda()
    
    def encode_audio(self, audio_path: str) -> List[int]:
        """
        Encode audio file to codec tokens.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of codec token IDs
        """
        import librosa
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
        
        # Ensure audio is not too long
        max_samples = int(self.config.max_audio_duration * self.config.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)
        
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.cuda()
        
        # Encode using codec
        with torch.no_grad():
            encoded = self.codec_model.encode(audio_tensor)
            tokens = encoded["tokens"].squeeze().cpu().numpy()
        
        # Convert to token IDs in the audio token range
        token_ids = tokens.flatten().tolist()
        
        # Map tokens to the audio token range [64001, 64399]
        min_token = self.config.audio_token_range[0]
        max_token = self.config.audio_token_range[1]
        token_range = max_token - min_token
        
        # Normalize and map tokens
        mapped_tokens = []
        for token in token_ids:
            # Assume tokens are in range [0, 255] or similar
            normalized = int((token % token_range)) + min_token
            mapped_tokens.append(normalized)
        
        return mapped_tokens
    
    def prepare_dataset(self, metadata_file: str) -> Dataset:
        """
        Prepare dataset from metadata file.
        
        Expected metadata format (CSV):
        audio_path,transcript
        audio/file1.wav,Kaixo mundua
        audio/file2.wav,Eskerrik asko
        
        Args:
            metadata_file: Path to metadata CSV file
            
        Returns:
            Hugging Face Dataset object
        """
        logger.info(f"Loading metadata from: {metadata_file}")
        
        # Load metadata
        import pandas as pd
        df = pd.read_csv(metadata_file)
        
        logger.info(f"Found {len(df)} samples")
        
        # Prepare samples
        samples = []
        for idx, row in df.iterrows():
            audio_path = row['audio_path']
            transcript = row['transcript']
            
            # Make path absolute if relative
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(
                    os.path.dirname(metadata_file), 
                    audio_path
                )
            
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                continue
            
            samples.append({
                'audio_path': audio_path,
                'transcript': transcript
            })
        
        logger.info(f"Prepared {len(samples)} valid samples")
        
        # Create dataset
        dataset = Dataset.from_dict({
            'audio_path': [s['audio_path'] for s in samples],
            'transcript': [s['transcript'] for s in samples]
        })
        
        return dataset


class TTSDataCollator:
    """Custom data collator for TTS training."""
    
    def __init__(self, tokenizer, config: FinetuningConfig):
        self.tokenizer = tokenizer
        self.config = config
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of features."""
        
        # Extract input_ids and labels
        input_ids = [f['input_ids'] for f in features]
        labels = [f['labels'] for f in features]
        
        # Pad sequences
        max_length = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        attention_mask = []
        
        for ids, lbls in zip(input_ids, labels):
            padding_length = max_length - len(ids)
            
            padded_input_ids.append(
                ids + [self.tokenizer.pad_token_id] * padding_length
            )
            padded_labels.append(
                lbls + [-100] * padding_length
            )
            attention_mask.append(
                [1] * len(ids) + [0] * padding_length
            )
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        }


class EuskeraFinetuner:
    """Main class for finetuning Kani TTS on Basque language."""
    
    def __init__(self, config: FinetuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.preprocessor = AudioDatasetPreprocessor(config)
        
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading base model: {self.config.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True
        )
        
        # Set special tokens if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
        )
        
        # Enable gradient checkpointing if specified
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def tokenize_function(self, examples):
        """
        Tokenize text and combine with audio tokens.
        
        For TTS, the format is typically:
        [START_OF_TEXT] <text tokens> [START_OF_AUDIO] <audio tokens> [END_OF_AUDIO]
        """
        tokenized_samples = []
        
        for transcript, audio_path in zip(examples['transcript'], examples['audio_path']):
            # Tokenize text
            text_tokens = self.tokenizer.encode(
                transcript,
                add_special_tokens=False
            )
            
            # Encode audio
            audio_tokens = self.preprocessor.encode_audio(audio_path)
            
            # Construct full sequence
            input_ids = (
                [self.config.start_of_text] +
                text_tokens +
                [self.config.start_of_audio] +
                audio_tokens +
                [self.config.end_of_audio]
            )
            
            # Truncate if too long
            if len(input_ids) > self.config.max_seq_length:
                input_ids = input_ids[:self.config.max_seq_length]
            
            # Labels are the same as input_ids (teacher forcing)
            # But we only compute loss on audio tokens
            labels = input_ids.copy()
            
            # Mask text tokens in labels (set to -100 to ignore in loss)
            text_length = len(text_tokens) + 2  # +2 for START_OF_TEXT and START_OF_AUDIO
            for i in range(text_length):
                if i < len(labels):
                    labels[i] = -100
            
            tokenized_samples.append({
                'input_ids': input_ids,
                'labels': labels
            })
        
        return tokenized_samples
    
    def prepare_datasets(self):
        """Prepare train and validation datasets."""
        logger.info("Preparing datasets...")
        
        # Load codec model for audio encoding
        self.preprocessor.load_codec_model()
        
        # Load dataset
        dataset = self.preprocessor.prepare_dataset(
            self.config.transcripts_file
        )
        
        # Split into train and validation
        dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split['train']
        eval_dataset = dataset_split['test']
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Eval samples: {len(eval_dataset)}")
        
        # Tokenize datasets
        logger.info("Tokenizing datasets...")
        
        train_tokenized = []
        for sample in train_dataset:
            tokens = self.tokenize_function({
                'transcript': [sample['transcript']],
                'audio_path': [sample['audio_path']]
            })
            train_tokenized.extend(tokens)
        
        eval_tokenized = []
        for sample in eval_dataset:
            tokens = self.tokenize_function({
                'transcript': [sample['transcript']],
                'audio_path': [sample['audio_path']]
            })
            eval_tokenized.extend(tokens)
        
        train_dataset_processed = Dataset.from_dict({
            'input_ids': [t['input_ids'] for t in train_tokenized],
            'labels': [t['labels'] for t in train_tokenized]
        })
        
        eval_dataset_processed = Dataset.from_dict({
            'input_ids': [t['input_ids'] for t in eval_tokenized],
            'labels': [t['labels'] for t in eval_tokenized]
        })
        
        return train_dataset_processed, eval_dataset_processed
    
    def train(self):
        """Execute the finetuning process."""
        logger.info("Starting finetuning process...")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_datasets()
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            logging_dir=os.path.join(self.config.output_dir, 'logs'),
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["tensorboard"],
            save_safetensors=True,
        )
        
        # Create data collator
        data_collator = TTSDataCollator(self.tokenizer, self.config)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving final model to {self.config.output_dir}")
        trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("Training completed successfully!")
        logger.info(f"Trained model saved at: {self.config.output_dir}")
        
        return self.config.output_dir


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Finetune Kani TTS model for Basque (Euskera)"
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="nineninesix/kani-tts-400m-0.3-pt",
        help="Base model to finetune from"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/euskera",
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--transcripts_file",
        type=str,
        default="data/euskera/metadata.csv",
        help="Path to transcripts metadata CSV file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/kani-tts-euskera",
        help="Output directory for trained model"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
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
    
    args = parser.parse_args()
    
    # Create configuration
    config = FinetuningConfig(
        base_model_name=args.base_model,
        dataset_path=args.dataset_path,
        transcripts_file=args.transcripts_file,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize finetuner
    finetuner = EuskeraFinetuner(config)
    
    # Run training
    trained_model_path = finetuner.train()
    
    logger.info("=" * 80)
    logger.info("FINETUNING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Trained model location: {trained_model_path}")
    logger.info("You can now use this model for Basque TTS inference")


if __name__ == "__main__":
    main()
