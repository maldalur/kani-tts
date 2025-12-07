#!/usr/bin/env python3
"""
Inference script for Basque (Euskera) Kani TTS model

This script demonstrates how to use the finetuned Basque model
for text-to-speech generation.

Usage:
    python inference_euskera.py --model models/kani-tts-euskera --text "Kaixo mundua"
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
import logging

from transformers import AutoTokenizer, AutoModelForCausalLM
from nemo.collections.tts.models import AudioCodecModel
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EuskeraTTSInference:
    """Inference class for Basque TTS model."""
    
    def __init__(self, model_path: str, codec_model_name: str = None):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the finetuned model
            codec_model_name: Name or path of the codec model
        """
        self.model_path = model_path
        self.codec_model_name = codec_model_name or "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self.codec_model = None
        
        # Token IDs
        self.start_of_text = 1
        self.start_of_audio = 64000
        self.end_of_audio = 64399
        self.audio_token_range = (64001, 64399)
        
        self._load_models()
    
    def _load_models(self):
        """Load the TTS model, tokenizer, and codec."""
        logger.info(f"Loading finetuned model from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Load TTS model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        
        logger.info(f"Loading codec model: {self.codec_model_name}")
        
        # Load codec model
        self.codec_model = AudioCodecModel.from_pretrained(
            self.codec_model_name
        )
        self.codec_model.eval()
        if self.device == "cuda":
            self.codec_model = self.codec_model.cuda()
        
        logger.info("Models loaded successfully")
    
    def generate(
        self,
        text: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 1200,
        repetition_penalty: float = 1.1
    ) -> np.ndarray:
        """
        Generate speech from text.
        
        Args:
            text: Input text in Basque
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum number of tokens to generate
            repetition_penalty: Penalty for token repetition
            
        Returns:
            Audio waveform as numpy array
        """
        logger.info(f"Generating speech for: {text}")
        
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.model.device)
        
        # Prepend START_OF_TEXT token
        input_ids = inputs['input_ids']
        start_token = torch.tensor([[self.start_of_text]], device=input_ids.device)
        audio_start_token = torch.tensor([[self.start_of_audio]], device=input_ids.device)
        
        input_ids = torch.cat([start_token, input_ids, audio_start_token], dim=1)
        
        # Generate audio tokens
        logger.info("Generating audio tokens...")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract audio tokens (everything after START_OF_AUDIO, before END_OF_AUDIO)
        generated_ids = outputs[0].cpu().numpy()
        
        # Find START_OF_AUDIO position
        audio_start_pos = np.where(generated_ids == self.start_of_audio)[0]
        if len(audio_start_pos) == 0:
            logger.error("No audio tokens generated")
            return None
        
        audio_start_pos = audio_start_pos[0] + 1
        
        # Find END_OF_AUDIO position
        end_of_audio_pos = np.where(generated_ids == self.end_of_audio)[0]
        if len(end_of_audio_pos) > 0:
            audio_end_pos = end_of_audio_pos[0]
        else:
            audio_end_pos = len(generated_ids)
        
        # Extract audio tokens
        audio_tokens = generated_ids[audio_start_pos:audio_end_pos]
        
        # Filter tokens to valid audio range
        audio_tokens = audio_tokens[
            (audio_tokens >= self.audio_token_range[0]) & 
            (audio_tokens <= self.audio_token_range[1])
        ]
        
        if len(audio_tokens) == 0:
            logger.error("No valid audio tokens in generated sequence")
            return None
        
        logger.info(f"Generated {len(audio_tokens)} audio tokens")
        
        # Map tokens back to codec token space
        # Assuming codec expects tokens in range [0, 398]
        codec_tokens = audio_tokens - self.audio_token_range[0]
        
        # Decode audio using codec
        logger.info("Decoding audio with codec...")
        audio_waveform = self._decode_audio(codec_tokens)
        
        return audio_waveform
    
    def _decode_audio(self, tokens: np.ndarray) -> np.ndarray:
        """
        Decode audio tokens to waveform using codec.
        
        Args:
            tokens: Audio tokens to decode
            
        Returns:
            Audio waveform as numpy array
        """
        # Reshape tokens for codec (assuming codec expects specific shape)
        # This may need adjustment based on actual codec requirements
        tokens_tensor = torch.LongTensor(tokens).unsqueeze(0)
        
        if self.device == "cuda":
            tokens_tensor = tokens_tensor.cuda()
        
        with torch.no_grad():
            # Decode using codec
            audio_dict = {"tokens": tokens_tensor.unsqueeze(-1)}
            decoded_audio = self.codec_model.decode(audio_dict)
            
            # Extract waveform
            if isinstance(decoded_audio, dict):
                audio_waveform = decoded_audio["audio"].squeeze().cpu().numpy()
            else:
                audio_waveform = decoded_audio.squeeze().cpu().numpy()
        
        return audio_waveform
    
    def save_audio(self, audio: np.ndarray, output_path: str, sample_rate: int = 22050):
        """
        Save audio to file.
        
        Args:
            audio: Audio waveform
            output_path: Path to save the audio file
            sample_rate: Audio sample rate
        """
        sf.write(output_path, audio, sample_rate)
        logger.info(f"Audio saved to: {output_path}")


def main():
    """Main entry point for inference script."""
    parser = argparse.ArgumentParser(
        description="Generate Basque speech using finetuned Kani TTS model"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the finetuned model directory"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize (in Basque)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output_euskera.wav",
        help="Output audio file path"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter (default: 0.95)"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1200,
        help="Maximum tokens to generate (default: 1200)"
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    logger.info("Initializing Euskera TTS inference engine...")
    inference = EuskeraTTSInference(args.model)
    
    # Generate speech
    audio = inference.generate(
        text=args.text,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    if audio is not None:
        # Save audio
        inference.save_audio(audio, args.output)
        logger.info("=" * 80)
        logger.info("SUCCESS! Audio generation complete.")
        logger.info(f"Output file: {args.output}")
        logger.info("=" * 80)
    else:
        logger.error("Failed to generate audio")


if __name__ == "__main__":
    main()
