# Dataset Information for TTS Training

This document provides information about suitable datasets for training TTS models.

## Recommended Public Datasets

### 1. LJSpeech Dataset

**Description:** Single-speaker English TTS dataset

- **Size:** ~2.6 GB
- **Duration:** ~24 hours of speech
- **Speaker:** Single female speaker (Linda Johnson)
- **Sample Rate:** 22050 Hz
- **Format:** WAV files with text transcriptions
- **Language:** English
- **License:** Public Domain
- **URL:** https://keithito.com/LJ-Speech-Dataset/

**Usage:**
```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
```

### 2. Common Voice

**Description:** Multilingual crowdsourced speech dataset by Mozilla

- **Languages:** 100+ languages
- **Duration:** Varies by language (100-2000+ hours)
- **Sample Rate:** 48000 Hz
- **Format:** MP3 files with metadata
- **License:** CC0 (Public Domain)
- **URL:** https://commonvoice.mozilla.org/

**Features:**
- Diverse speakers (age, gender, accent)
- Multiple languages
- Validated transcriptions
- Community-driven

### 3. LibriTTS

**Description:** Multi-speaker English corpus derived from LibriVox audiobooks

- **Size:** ~245 GB
- **Duration:** 585 hours
- **Speakers:** 2,456 speakers
- **Sample Rate:** 24000 Hz
- **Format:** FLAC files
- **Language:** English
- **License:** CC BY 4.0
- **URL:** https://www.openslr.org/60/

**Subsets:**
- `train-clean-100`: 100 hours, clean speech
- `train-clean-360`: 360 hours, clean speech
- `train-other-500`: 500 hours, other speech
- `dev-clean`: Development set (clean)
- `test-clean`: Test set (clean)

### 4. VCTK Corpus

**Description:** Multi-speaker English corpus with various accents

- **Size:** ~10 GB
- **Duration:** ~44 hours
- **Speakers:** 110 speakers (109 English, 1 American)
- **Sample Rate:** 48000 Hz
- **Format:** WAV files
- **Language:** English (multiple accents)
- **License:** CC BY 4.0
- **URL:** https://datashare.ed.ac.uk/handle/10283/3443

### 5. M-AILABS Speech Dataset

**Description:** Multi-language multi-speaker dataset from audiobooks

- **Languages:** English, German, Spanish, French, Italian, Polish, Russian, Ukrainian
- **Duration:** ~1000 hours total
- **Sample Rate:** 16000 Hz
- **Format:** WAV files
- **License:** Varies (mostly permissive)
- **URL:** https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/

## Dataset Preparation Guidelines

### Audio Requirements

1. **Format:** WAV files (recommended) or FLAC
2. **Sample Rate:** 22050 Hz or 24000 Hz
3. **Bit Depth:** 16-bit PCM
4. **Channels:** Mono (single channel)
5. **Quality:** Clean recordings without background noise
6. **Duration:** 
   - Minimum per clip: 1-2 seconds
   - Maximum per clip: 10-15 seconds
   - Total dataset: 2+ hours recommended, 10+ hours ideal

### Text Requirements

1. **Transcriptions:** Accurate, word-for-word transcriptions
2. **Encoding:** UTF-8
3. **Normalization:** 
   - Expand abbreviations
   - Spell out numbers
   - Remove special characters (or handle appropriately)
4. **Punctuation:** Include proper punctuation

### Metadata Format

Create a CSV file with the following structure:

```csv
audio_path,transcript
audio/sample_001.wav,This is the first sample text
audio/sample_002.wav,This is the second sample text
audio/sample_003.wav,This is the third sample text
```

## Using Custom Datasets

### Option 1: Datamio

[Datamio](https://app.datamio.dev/) is a tool for collecting and managing audio datasets:

1. Record or upload audio files
2. Transcribe the audio
3. Export in the required format
4. Use with Kani TTS training pipeline

### Option 2: nano-codec-dataset-pipeline

Use the [nano-codec-dataset-pipeline](https://github.com/nineninesix-ai/nano-codec-dataset-pipeline) for audio processing:

1. Collect raw audio files
2. Run preprocessing (normalization, resampling)
3. Extract features using NeMo NanoCodec
4. Format for training

**Steps:**
```bash
git clone https://github.com/nineninesix-ai/nano-codec-dataset-pipeline
cd nano-codec-dataset-pipeline
pip install -r requirements.txt

# Process your dataset
python process_dataset.py \
  --input_dir /path/to/audio \
  --output_dir /path/to/processed \
  --sample_rate 22050
```

## Dataset Statistics

| Dataset | Language | Hours | Speakers | Sample Rate | License |
|---------|----------|-------|----------|-------------|---------|
| LJSpeech | English | 24 | 1 | 22.05 kHz | Public Domain |
| Common Voice | 100+ | Varies | Many | 48 kHz | CC0 |
| LibriTTS | English | 585 | 2,456 | 24 kHz | CC BY 4.0 |
| VCTK | English | 44 | 110 | 48 kHz | CC BY 4.0 |
| M-AILABS | 8 languages | 1000+ | Many | 16 kHz | Varies |

## Quality Checklist

Before training, ensure your dataset meets these criteria:

- [ ] Audio files are in WAV or FLAC format
- [ ] Sample rate is consistent (22050 Hz or 24000 Hz)
- [ ] Audio is mono (single channel)
- [ ] No background noise or artifacts
- [ ] Transcriptions are accurate
- [ ] Text is properly normalized
- [ ] Metadata file is correctly formatted
- [ ] Dataset is split into train/validation sets
- [ ] Total duration is at least 2 hours (preferably 10+)

## License Considerations

When using public datasets:

1. **Check the license** before using any dataset
2. **Respect attribution requirements**
3. **Don't use datasets for commercial purposes** if prohibited
4. **Share derived datasets** appropriately if required

## Additional Resources

- [Kani TTS Main Repository](https://github.com/maldalur/kani-tts)
- [KaniTTS Finetune Pipeline](https://github.com/nineninesix-ai/KaniTTS-Finetune-pipeline)
- [Basque Finetuning Example](../../finetuning/euskera/)
- [Discord Community](https://discord.gg/NzP3rjB4SB)

## Support

For questions about dataset preparation:
- Open an issue on [GitHub](https://github.com/maldalur/kani-tts/issues)
- Join the [Discord community](https://discord.gg/NzP3rjB4SB)
- Check the [documentation](../../README.md)
