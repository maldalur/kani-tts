#!/usr/bin/env python3
"""
Test script to verify the trained model can be loaded and used.
"""

import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_trained_model():
    """Test loading and using the trained model."""
    
    model_path = "/home/runner/work/kani-tts/kani-tts/training/simple_example/models/kani-tts-trained"
    
    logger.info("=" * 80)
    logger.info("Testing Trained TTS Model")
    logger.info("=" * 80)
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at: {model_path}")
        return False
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        logger.info("✓ Tokenizer loaded successfully")
        logger.info(f"  Vocab size: {len(tokenizer)}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        logger.info("✓ Model loaded successfully")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
        # Test inference
        logger.info("\nTesting inference...")
        test_text = "Hello world"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        logger.info(f"Input text: '{test_text}'")
        logger.info(f"Input token IDs: {inputs['input_ids'].tolist()}")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=50,
                num_return_sequences=1,
                do_sample=False,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated tokens: {outputs[0].tolist()}")
        logger.info(f"Generated text: '{generated_text}'")
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Model test completed successfully!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_trained_model()
    exit(0 if success else 1)
