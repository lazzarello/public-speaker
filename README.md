# Audio Analysis and Text-to-Speech Tool

This project provides a Python-based tool for audio analysis and text-to-speech conversion using state-of-the-art AI models.

## Features

- **Audio Analysis**: Analyze audio files using Qwen Audio Chat model
- **Text-to-Speech**: Convert text to speech using multiple TTS models

## Audio Analysis

The tool uses [Qwen Audio Chat](https://huggingface.co/Qwen/Qwen-Audio-Chat), a multimodal model that can understand and respond to questions about audio content. It supports:

- Analyzing any audio file (local or URL)
- Asking custom questions about the audio content
- Saving analysis results to text files

## Text-to-Speech

The tool provides a flexible TTS system with multiple fallback options:

- Primary: [Coqui XTTS-v2](https://github.com/coqui-ai/TTS)
- Alternatives:
  - Microsoft SpeechT5
  - Facebook MMS-TTS
  - Other transformer-based models

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

The following Python packages are required:

```
transformers
torch
soundfile
TTS (optional, for XTTS support)
sentencepiece (for SpeechT5)
numpy
```

## Usage

```python
# Audio analysis
analyze_audio("path/to/audio.wav", "What instruments are playing in this audio?")

# Text-to-speech
speak("Hello, this is synthesized speech")
speak("Custom voice example", voice="path/to/reference.wav")
speak("Using XTTS model", model_name="coqui/xtts-v2")
```

## Output

All outputs are saved to the `output` directory:
- Audio analysis: `output/analysis_filename.txt`
- Generated speech: `output/text_preview.wav`
