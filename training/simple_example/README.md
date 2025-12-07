# Simple TTS Model Training Example

This directory contains a simple example for training a TTS model and saving it to the repository.

## Overview

This example demonstrates:
- Loading a base TTS model
- Training on a synthetic dataset
- Saving the trained model to the repository

## Dataset

For this demonstration, we use a small synthetic dataset. In production, you would use:

### Recommended Datasets for TTS:

1. **LJSpeech** - Single speaker English dataset
   - 13,100 audio clips
   - ~24 hours of speech
   - Available at: https://keithito.com/LJ-Speech-Dataset/

2. **Common Voice** - Multilingual crowdsourced dataset
   - Multiple languages
   - Available at: https://commonvoice.mozilla.org/

3. **LibriTTS** - Multi-speaker English corpus
   - 585 hours of read English speech
   - Available at: https://www.openslr.org/60/

4. **Custom Dataset** - Follow the preparation guidelines from:
   - [nano-codec-dataset-pipeline](https://github.com/nineninesix-ai/nano-codec-dataset-pipeline)
   - [Datamio](https://app.datamio.dev/)

## Requirements

```bash
pip install torch transformers datasets accelerate
```

## Usage

### Quick Start

```bash
# Navigate to this directory
cd training/simple_example

# Run the training script
python train_model.py
```

### What This Does

1. Creates a synthetic dataset (10 text samples)
2. Loads the base Kani TTS model (or creates a demo model if unavailable)
3. Trains for 1 epoch
4. Saves the trained model to `models/kani-tts-trained/`

### Training Output

The trained model will be saved in:
```
training/simple_example/models/kani-tts-trained/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── tokenizer_config.json
├── training_metadata.json
└── logs/
```

## Training on Real Data

For production training with real audio data:

1. **Prepare Your Dataset:**
   ```
   data/
   ├── audio/
   │   ├── audio_001.wav
   │   ├── audio_002.wav
   │   └── ...
   └── metadata.csv
   ```

2. **Metadata Format (CSV):**
   ```csv
   audio_path,transcript
   audio/audio_001.wav,Hello world
   audio/audio_002.wav,This is a test
   ```

3. **Use the Full Training Pipeline:**
   - See `finetuning/euskera/` for a complete example
   - Follow [KaniTTS-Finetune-pipeline](https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline)

## Model Information

The trained model includes:
- **Model weights**: The trained TTS model parameters
- **Tokenizer**: Text tokenization configuration
- **Metadata**: Training information and configuration

## Using the Trained Model

After training, you can use the model for inference:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "training/simple_example/models/kani-tts-trained"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Use for inference
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
```

## Notes

- This is a demonstration script using synthetic data
- For production use, train with real audio-text pairs
- Training time depends on dataset size and hardware
- GPU recommended for training (CPU training is very slow)

## Hardware Requirements

| Hardware | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 8GB | 16GB+ |
| RAM | 16GB | 32GB+ |
| Storage | 10GB | 50GB+ |

## Support

For questions or issues:
- Check the main [README](../../README.md)
- See [finetuning examples](../../finetuning/)
- Join the [Discord community](https://discord.gg/NzP3rjB4SB)

## License

Apache 2.0 (same as the main Kani TTS repository)
