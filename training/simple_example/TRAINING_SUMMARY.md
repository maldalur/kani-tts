# Training Summary

## Overview

This document summarizes the trained TTS model saved in this directory.

## Model Information

- **Model Name:** kani-tts-trained
- **Model Type:** GPT-2 based TTS model
- **Parameters:** 3.38 Million
- **Architecture:**
  - Layers: 4
  - Embedding dimension: 256
  - Attention heads: 4
  - Inner dimension: 1024
  - Vocabulary size: 362

## Training Details

- **Base Model:** Created from scratch (not finetuned)
- **Dataset:** Synthetic dataset (10 text-audio pairs)
- **Training Epochs:** 1
- **Batch Size:** 1
- **Gradient Accumulation Steps:** 2
- **Learning Rate:** 5e-5
- **Training Time:** ~0.25 seconds
- **Final Training Loss:** 5.551

## Training Progress

```
Epoch 0.2: loss=5.7586, grad_norm=12.40, lr=5e-05
Epoch 0.4: loss=5.6083, grad_norm=9.83, lr=4e-05
Epoch 0.6: loss=5.3949, grad_norm=9.82, lr=3e-05
Epoch 0.8: loss=5.3365, grad_norm=9.28, lr=2e-05
Epoch 1.0: loss=5.3464, grad_norm=9.72, lr=1e-05
```

## Model Files

Located in: `models/kani-tts-trained/`

```
├── model.safetensors          # Model weights (13 MB)
├── config.json                # Model configuration
├── tokenizer_config.json      # Tokenizer configuration
├── vocab.json                 # Vocabulary (362 tokens)
├── merges.txt                 # BPE merges
├── training_metadata.json     # Training information
└── checkpoint-5/              # Training checkpoint
```

## Dataset Used

**Type:** Synthetic demonstration dataset

**Samples:**
1. "Hello world, this is a test."
2. "The quick brown fox jumps over the lazy dog."
3. "Machine learning is fascinating."
4. "Text to speech synthesis."
5. "Welcome to the future of AI."
6. "Natural language processing."
7. "Deep learning models are powerful."
8. "Speech recognition technology."
9. "Audio generation systems."
10. "Transformer architectures work well."

## Testing Results

✓ Model loads successfully
✓ Tokenizer works correctly
✓ Inference generates output
✓ Model size: 3.38M parameters

**Test Input:** "Hello world"
**Test Output:** Generated sequence of tokens

## Production Usage Notes

⚠️ **Important:** This is a demonstration model trained on synthetic data.

For production TTS:
1. Use a real audio dataset (see [DATASET_INFO.md](DATASET_INFO.md))
2. Train for more epochs (5-10+)
3. Use a larger dataset (2-10+ hours of audio)
4. Fine-tune hyperparameters for your use case
5. Consider using a pretrained base model

## Recommended Next Steps

1. **Collect Real Data:**
   - Use [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) for English
   - Use [Common Voice](https://commonvoice.mozilla.org/) for multilingual
   - Use [Datamio](https://app.datamio.dev/) for custom data

2. **Prepare Dataset:**
   - Follow [nano-codec-dataset-pipeline](https://github.com/nineninesix-ai/nano-codec-dataset-pipeline)
   - Format audio to 22050 Hz WAV
   - Create accurate transcriptions

3. **Train Production Model:**
   - Use the scripts in this directory as a template
   - Train for at least 5-10 epochs
   - Monitor validation loss
   - Use GPU for faster training

4. **Evaluate:**
   - Test on held-out validation set
   - Listen to generated audio quality
   - Measure objective metrics (if applicable)
   - Compare with baseline models

## Additional Resources

- [Main README](../../README.md)
- [Training Guide](README.md)
- [Dataset Information](DATASET_INFO.md)
- [Basque Finetuning Example](../../finetuning/euskera/)
- [KaniTTS Finetune Pipeline](https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline)
- [Discord Community](https://discord.gg/NzP3rjB4SB)

## License

Apache 2.0 (same as the main Kani TTS repository)

---

**Model Trained:** 2025-12-07
**Training Platform:** GitHub Copilot Environment
**Purpose:** Demonstration and educational use
