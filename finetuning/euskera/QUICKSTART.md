# Quick Start Guide: Basque TTS Finetuning

This guide will help you get started with finetuning Kani TTS for Basque (Euskera) in just a few steps.

## 1. Install Dependencies

```bash
cd finetuning/euskera
pip install -r requirements.txt
```

## 2. Prepare Your Dataset

### Option A: Use Sample Data (for testing)

A sample metadata file is provided at `example_metadata.csv`. Create the audio directory:

```bash
mkdir -p data/euskera/audio
```

Then add your `.wav` files to `data/euskera/audio/` matching the filenames in `example_metadata.csv`.

### Option B: Create Your Own Dataset

1. Create the directory structure:
```bash
mkdir -p data/euskera/audio
```

2. Add your audio files to `data/euskera/audio/`

3. Create `data/euskera/metadata.csv`:
```csv
audio_path,transcript
audio/file1.wav,Kaixo mundua
audio/file2.wav,Eskerrik asko
```

**Dataset Requirements:**
- At least 50-100 audio samples (more is better)
- Clean audio recordings
- Accurate Basque transcriptions
- WAV format, 22050 Hz sample rate

## 3. Start Finetuning

### Basic Training (Default Settings)

```bash
python finetune_euskera.py \
    --dataset_path data/euskera \
    --transcripts_file data/euskera/metadata.csv \
    --output_dir models/kani-tts-euskera
```

### Custom Configuration

```bash
python finetune_euskera.py \
    --dataset_path data/euskera \
    --transcripts_file data/euskera/metadata.csv \
    --output_dir models/kani-tts-euskera \
    --num_epochs 5 \
    --batch_size 2 \
    --learning_rate 3e-5
```

## 4. Monitor Training

While training is running, open a new terminal and launch TensorBoard:

```bash
tensorboard --logdir models/kani-tts-euskera/logs
```

Then open http://localhost:6006 in your browser to see:
- Training loss
- Validation loss
- Learning rate schedule
- Other metrics

## 5. Test the Trained Model

After training completes, test your model:

```bash
python inference_euskera.py \
    --model models/kani-tts-euskera \
    --text "Kaixo, zer moduz zaude?" \
    --output test_output.wav
```

Play the generated audio:
```bash
# On Linux
aplay test_output.wav

# On Mac
afplay test_output.wav

# Or use any audio player
vlc test_output.wav
```

## Expected Training Time

| Dataset Size | GPU | Training Time (3 epochs) |
|--------------|-----|--------------------------|
| 100 samples  | RTX 4080 | ~30 minutes |
| 500 samples  | RTX 4080 | ~2 hours |
| 1000 samples | RTX 4080 | ~4 hours |
| 5000 samples | RTX 4080 | ~20 hours |

*Times are approximate and may vary based on audio duration and hardware*

## Troubleshooting

### "CUDA out of memory"
Reduce batch size:
```bash
python finetune_euskera.py --batch_size 1 ...
```

### "Audio file not found"
Check that:
- Audio files exist in the specified directory
- Paths in metadata.csv are correct
- File extensions match (.wav)

### "Module not found"
Reinstall dependencies:
```bash
pip install -r requirements.txt --upgrade
```

### Poor audio quality
- Train for more epochs (5-10)
- Collect more training data
- Check input audio quality
- Try different hyperparameters

## Next Steps

1. **Improve Quality:**
   - Add more training data
   - Train for more epochs
   - Fine-tune hyperparameters
   - Use higher quality audio recordings

2. **Experiment:**
   - Try different base models
   - Adjust temperature and top_p for inference
   - Test with different text inputs
   - Create a web interface

3. **Share:**
   - Upload your model to Hugging Face
   - Share with the Basque language community
   - Contribute back to the project

## Support

- **Main Repository:** https://github.com/maldalur/kani-tts
- **Discord:** https://discord.gg/NzP3rjB4SB
- **Issues:** https://github.com/maldalur/kani-tts/issues

## Example Commands Summary

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train
python finetune_euskera.py \
    --dataset_path data/euskera \
    --transcripts_file data/euskera/metadata.csv \
    --output_dir models/kani-tts-euskera

# 3. Inference
python inference_euskera.py \
    --model models/kani-tts-euskera \
    --text "Kaixo mundua" \
    --output output.wav
```

Good luck with your finetuning! 🎤🇪🇺
