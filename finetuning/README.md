# Kani TTS Finetuning

This directory contains finetuning pipelines for Kani TTS in different languages.

## Available Languages

### 🇪🇺 Basque (Euskera)

Complete finetuning pipeline for Basque language TTS.

**Location:** [`euskera/`](euskera/)

**Features:**
- Full training script with audio codec integration
- Dataset preparation and validation tools
- Inference script for testing trained models
- Google Colab notebook for cloud training
- Comprehensive documentation

**Quick Start:**
```bash
cd euskera
./setup.sh
python finetune_euskera.py --help
```

**Documentation:**
- [Full Guide](euskera/README.md)
- [Quick Start](euskera/QUICKSTART.md)
- [Colab Notebook](euskera/euskera_finetuning_colab.ipynb)

## Dataset Preparation

All finetuning pipelines follow the dataset preparation guidelines from:
- [nano-codec-dataset-pipeline](https://github.com/nineninesix-ai/nano-codec-dataset-pipeline) - Audio processing
- [Datamio](https://app.datamio.dev/) - Dataset collection tool

### General Dataset Requirements

For any language:
1. **Audio Files:** WAV format, 22050 Hz sample rate
2. **Transcripts:** Accurate text transcriptions
3. **Metadata:** CSV file mapping audio files to transcripts
4. **Quality:** Clean recordings without background noise
5. **Duration:** 2-5 hours of audio recommended

### Metadata Format

```csv
audio_path,transcript
audio/sample_001.wav,Text transcription here
audio/sample_002.wav,Another transcription
```

## Adding New Languages

To add finetuning support for a new language:

1. Create a new directory: `finetuning/<language_code>/`
2. Copy and adapt the scripts from `euskera/` directory
3. Update dataset handling for language-specific requirements
4. Create language-specific documentation
5. Add example metadata with text in your target language

**Example structure:**
```
finetuning/
├── README.md (this file)
└── <language_code>/
    ├── README.md
    ├── QUICKSTART.md
    ├── requirements.txt
    ├── setup.sh
    ├── finetune_<language>.py
    ├── inference_<language>.py
    ├── validate_dataset.py
    ├── config.yaml
    └── example_metadata.csv
```

## General Finetuning Resources

For other languages or advanced finetuning:
- [KaniTTS-Finetune-pipeline](https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline) - Official external pipeline
- [Main Repository](https://github.com/maldalur/kani-tts) - Kani TTS main repo
- [Discord Community](https://discord.gg/NzP3rjB4SB) - Get help and share

## Training Tips

### Hardware Requirements

| GPU VRAM | Batch Size | Training Time (1000 samples) |
|----------|------------|------------------------------|
| 12GB     | 1-2        | ~8 hours                     |
| 16GB     | 2-4        | ~5 hours                     |
| 24GB     | 4-8        | ~3 hours                     |

### Optimization Tips

1. **Memory Optimization:**
   - Enable gradient checkpointing
   - Use mixed precision (fp16)
   - Reduce batch size if OOM
   - Increase gradient accumulation steps

2. **Quality Improvement:**
   - Use high-quality audio recordings
   - Ensure accurate transcriptions
   - Train for 5-10 epochs
   - Collect diverse training data

3. **Speed Optimization:**
   - Use multiple GPUs with `accelerate`
   - Enable compile mode (PyTorch 2.0+)
   - Use efficient data loading

## Contributing

We welcome contributions for new languages! Please:

1. Follow the existing structure and patterns
2. Include comprehensive documentation
3. Test your pipeline before submitting
4. Share example results with the community

## Support

- **Issues:** [GitHub Issues](https://github.com/maldalur/kani-tts/issues)
- **Discussions:** [GitHub Discussions](https://github.com/maldalur/kani-tts/discussions)
- **Discord:** [Join our community](https://discord.gg/NzP3rjB4SB)

## License

Apache 2.0 (same as the main Kani TTS repository)
