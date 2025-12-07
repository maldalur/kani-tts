# Kani TTS Finetuning for Basque (Euskera)

This directory contains a complete pipeline for finetuning the Kani TTS model on Basque (Euskera) language audio data.

## Overview

The finetuning pipeline follows the dataset preparation guidelines suggested by the Kani TTS repository:
- Uses the NeMo NanoCodec for audio encoding (as suggested in [nano-codec-dataset-pipeline](https://github.com/nineninesix-ai/nano-codec-dataset-pipeline))
- Implements the training methodology from [KaniTTS-Finetune-pipeline](https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline)
- Returns a fully trained model ready for Basque TTS inference

## Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch>=2.0.0`
- `transformers>=4.35.0`
- `datasets>=2.14.0`
- `librosa>=0.10.0`
- `pandas>=2.0.0`
- `nemo_toolkit[tts]>=1.20.0`
- `numpy>=1.24.0`

### Hardware Requirements

- **Recommended:** NVIDIA GPU with 16GB+ VRAM
- **Minimum:** NVIDIA GPU with 12GB VRAM
- CUDA 11.8 or later
- 32GB+ system RAM

## Dataset Preparation

### 1. Dataset Structure

Your Basque audio dataset should be organized as follows:

```
data/euskera/
├── audio/
│   ├── sample_001.wav
│   ├── sample_002.wav
│   └── ...
└── metadata.csv
```

### 2. Metadata Format

The `metadata.csv` file should contain two columns:

```csv
audio_path,transcript
audio/sample_001.wav,Kaixo mundua
audio/sample_002.wav,Eskerrik asko
audio/sample_003.wav,Zer moduz zaude?
```

**Requirements for audio files:**
- **Format:** WAV files
- **Sample Rate:** 22050 Hz (will be resampled automatically)
- **Duration:** Up to 30 seconds per sample
- **Quality:** Clean recordings without background noise
- **Language:** Basque (Euskera) transcriptions

### 3. Creating Your Dataset

You can use [Datamio](https://app.datamio.dev/) to collect and organize your audio dataset, as suggested in the main repository README.

For audio processing, follow the guidelines from the [nano-codec-dataset-pipeline](https://github.com/nineninesix-ai/nano-codec-dataset-pipeline):
1. Collect raw audio recordings
2. Preprocess and normalize audio
3. Extract features
4. Format for training
5. Validate quality

## Usage

### Basic Training

To start finetuning with default settings:

```bash
python finetune_euskera.py \
    --dataset_path data/euskera \
    --transcripts_file data/euskera/metadata.csv \
    --output_dir models/kani-tts-euskera
```

### Advanced Configuration

```bash
python finetune_euskera.py \
    --base_model nineninesix/kani-tts-400m-0.3-pt \
    --dataset_path data/euskera \
    --transcripts_file data/euskera/metadata.csv \
    --output_dir models/kani-tts-euskera \
    --num_epochs 5 \
    --batch_size 4 \
    --learning_rate 5e-5
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--base_model` | Base model to finetune | `nineninesix/kani-tts-400m-0.3-pt` |
| `--dataset_path` | Path to dataset directory | `data/euskera` |
| `--transcripts_file` | Path to metadata CSV | `data/euskera/metadata.csv` |
| `--output_dir` | Output directory for model | `models/kani-tts-euskera` |
| `--num_epochs` | Number of training epochs | `3` |
| `--batch_size` | Training batch size per device | `4` |
| `--learning_rate` | Learning rate | `5e-5` |

## Training Configuration

### Default Hyperparameters

The finetuning script uses these default settings (can be modified in the code):

```python
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 5e-5
warmup_steps = 500
weight_decay = 0.01
max_seq_length = 1024
fp16 = True  # Mixed precision training
gradient_checkpointing = True  # Memory optimization
```

### Training Strategy

1. **Text Encoding:** Basque text is tokenized using the base model's tokenizer
2. **Audio Encoding:** Audio files are encoded using NeMo NanoCodec to audio tokens
3. **Sequence Format:** `[START_OF_TEXT] <text_tokens> [START_OF_AUDIO] <audio_tokens> [END_OF_AUDIO]`
4. **Loss Computation:** Only computed on audio tokens (text tokens are masked)
5. **Optimization:** AdamW optimizer with linear warmup and decay

## Output

After training completes, you'll find:

```
models/kani-tts-euskera/
├── config.json              # Model configuration
├── model.safetensors       # Trained model weights
├── tokenizer_config.json   # Tokenizer configuration
├── tokenizer.json          # Tokenizer vocabulary
└── logs/                   # TensorBoard training logs
```

## Using the Trained Model

### Inference Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nemo.collections.tts.models import AudioCodecModel

# Load the finetuned model
model_path = "models/kani-tts-euskera"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load codec for audio decoding
codec_model = AudioCodecModel.from_pretrained(
    "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
)

# Generate speech
text = "Kaixo, zer moduz zaude?"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        temperature=0.6,
        top_p=0.95,
        do_sample=True
    )

# Decode audio tokens (extract tokens after START_OF_AUDIO)
# ... (audio decoding logic)
```

For a complete inference example, refer to the `examples/basic` directory in the main repository.

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir models/kani-tts-euskera/logs
```

Then open your browser to `http://localhost:6006`

## Tips for Best Results

1. **Dataset Quality:**
   - Use high-quality, clean audio recordings
   - Ensure accurate transcriptions
   - Aim for at least 2-5 hours of audio data
   - Include diverse speakers and speaking styles

2. **Training:**
   - Start with the pretrained checkpoint (`kani-tts-400m-0.3-pt`)
   - Use a smaller learning rate (5e-5 to 1e-5)
   - Train for 3-10 epochs depending on dataset size
   - Monitor validation loss to avoid overfitting

3. **Hardware:**
   - Use mixed precision (fp16) to save memory
   - Enable gradient checkpointing if VRAM is limited
   - Adjust batch size based on available VRAM

4. **Evaluation:**
   - Test the model on held-out validation samples
   - Listen to generated audio quality
   - Check for pronunciation accuracy
   - Verify naturalness and prosody

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:
- Reduce `per_device_train_batch_size` (try 2 or 1)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable `gradient_checkpointing = True`
- Reduce `max_seq_length` to 512 or 768

### Slow Training

- Ensure you're using a GPU (check with `nvidia-smi`)
- Enable mixed precision training (`fp16=True`)
- Increase batch size if you have spare VRAM
- Use multiple GPUs with `accelerate` or `deepspeed`

### Poor Audio Quality

- Increase training duration (more epochs)
- Collect more high-quality training data
- Ensure audio preprocessing is correct
- Try different hyperparameters (learning rate, temperature)

## References

- [Kani TTS Main Repository](https://github.com/maldalur/kani-tts)
- [KaniTTS Finetuning Pipeline](https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline)
- [NanoCodec Dataset Pipeline](https://github.com/nineninesix-ai/nano-codec-dataset-pipeline)
- [NeMo Toolkit Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [Datamio Dataset Tool](https://app.datamio.dev/)

## License

Apache 2.0 (same as the main Kani TTS repository)

## Contributing

Contributions to improve the finetuning pipeline are welcome! Please submit issues or pull requests to the main repository.
