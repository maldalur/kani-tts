# Kani TTS Training Examples

This directory contains training examples and scripts for Kani TTS models.

## Overview

These examples demonstrate how to:
- Train a TTS model from scratch or finetune an existing model
- Save trained models to the repository
- Use appropriate datasets for TTS training
- Test and validate trained models

## Available Examples

### Simple Example

**Location:** [`simple_example/`](simple_example/)

A complete, minimal example for training a TTS model:
- Creates a small training dataset
- Trains a model
- Saves the trained model to the repository
- Includes testing script

**Quick Start:**
```bash
cd simple_example
pip install -r requirements.txt
python train_model.py
```

**Features:**
- ✓ Self-contained training script
- ✓ No external dataset required (uses synthetic data for demo)
- ✓ Saves model in repository-compatible format
- ✓ Includes model testing script
- ✓ Complete documentation

**Output:** Trained model saved in `simple_example/models/kani-tts-trained/`

### See Also

For more advanced training examples:
- **Basque (Euskera) Finetuning:** See [`../finetuning/euskera/`](../finetuning/euskera/)
- **General Finetuning:** See [KaniTTS-Finetune-pipeline](https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline)

## Training a Model

### Step 1: Choose Your Dataset

For production training, you'll need a real audio dataset. See [Dataset Information](simple_example/DATASET_INFO.md) for:
- Recommended public datasets (LJSpeech, Common Voice, LibriTTS, etc.)
- Dataset preparation guidelines
- Audio and text requirements
- Tools for dataset collection and processing

### Step 2: Prepare Your Dataset

Option A: Use a public dataset
```bash
# Example: Download LJSpeech
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
```

Option B: Use custom tools
- [Datamio](https://app.datamio.dev/) - Audio dataset collection tool
- [nano-codec-dataset-pipeline](https://github.com/nineninesix-ai/nano-codec-dataset-pipeline) - Audio processing pipeline

### Step 3: Train the Model

Using the simple example:
```bash
cd simple_example
python train_model.py
```

For real data, modify the configuration in the script or use command-line arguments.

### Step 4: Verify the Model

Test your trained model:
```bash
cd simple_example
python test_model.py
```

## Model Output

Trained models are saved with the following structure:
```
models/kani-tts-trained/
├── config.json              # Model configuration
├── model.safetensors        # Model weights (SafeTensors format)
├── tokenizer_config.json    # Tokenizer configuration
├── vocab.json               # Vocabulary
├── merges.txt               # BPE merges
├── training_metadata.json   # Training information
└── logs/                    # Training logs
```

## Using Your Trained Model

After training, you can use the model for inference:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model
model_path = "training/simple_example/models/kani-tts-trained"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Generate speech tokens
text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA GPU with 8GB VRAM | 16GB+ VRAM |
| RAM | 16GB | 32GB+ |
| Storage | 10GB free | 50GB+ free |
| CPU | 4 cores | 8+ cores |

## Training Tips

### Memory Optimization
- Use gradient checkpointing
- Enable mixed precision (fp16)
- Reduce batch size if OOM
- Increase gradient accumulation steps

### Quality Improvement
- Use high-quality audio recordings
- Ensure accurate transcriptions
- Train for 5-10 epochs minimum
- Use diverse training data

### Speed Optimization
- Use multiple GPUs with `accelerate`
- Enable compile mode (PyTorch 2.0+)
- Optimize data loading

## Datasets

For detailed information about suitable datasets, see:
- [Dataset Information Guide](simple_example/DATASET_INFO.md)

Popular choices:
1. **LJSpeech** - English, single speaker, 24 hours
2. **Common Voice** - Multilingual, crowdsourced
3. **LibriTTS** - English, multi-speaker, 585 hours
4. **VCTK** - English with various accents, 110 speakers

## Additional Resources

- **Main Repository:** [Kani TTS](https://github.com/maldalur/kani-tts)
- **Finetuning Pipeline:** [KaniTTS-Finetune-pipeline](https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline)
- **Basque Example:** [Euskera Finetuning](../finetuning/euskera/)
- **Discord Community:** [Join us](https://discord.gg/NzP3rjB4SB)

## Contributing

We welcome contributions! If you create a training example for a new language or use case:

1. Follow the structure of existing examples
2. Include comprehensive documentation
3. Test your pipeline before submitting
4. Share example results

## Support

For questions or issues:
- Check the [main README](../README.md)
- Open an issue on [GitHub](https://github.com/maldalur/kani-tts/issues)
- Join our [Discord community](https://discord.gg/NzP3rjB4SB)

## License

Apache 2.0 (same as the main Kani TTS repository)
