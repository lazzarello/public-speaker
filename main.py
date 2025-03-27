from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
import torch
import os
import soundfile as sf
import sys

def analyze_audio(audio_path: str, question: str = "What does this audio contain?"):
    """
    Analyze audio using Qwen Audio Chat model and return the response.

    Args:
        audio_path: Path to the audio file to analyze
        question: Question to ask about the audio (default: "What does this audio contain?")

    Returns:
        Model's response as text
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="auto", trust_remote_code=True)
    model.eval()

    # Format the query using the list format
    query = tokenizer.from_list_format([
        {'audio': audio_path},  # Can be a local path or URL
        {'text': question}
    ])

    # Generate response using the chat method
    response, history = model.chat(tokenizer, query=query, history=None)

    # Print the response
    print(f"Model response: {response}")

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Save the response to a file
    output_path = f"output/analysis_{os.path.basename(audio_path)}.txt"
    with open(output_path, 'w') as f:
        f.write(response)

    print(f"Response saved to {output_path}")
    return response

def speak(text: str, voice: str = "default", model_name: str = "microsoft/speecht5_tts"):
    """
    Convert text to speech using a TTS model and save as WAV file.
    
    Args:
        text: The text to convert to speech
        voice: Voice style to use (default: "default")
        model_name: Model to use for TTS (default: "microsoft/speecht5_tts")
    
    Returns:
        Path to the generated WAV file
    """
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    output_path = f"output/{text[:20].replace(' ', '_')}.wav"
    
    # Handle coqui/xtts-v2 model
    if "coqui/xtts" in model_name:
        try:
            # Try to import TTS library
            from TTS.api import TTS
            
            # Initialize TTS with the XTTS model
            tts = TTS(model_name)
            
            # Generate speech
            if voice != "default":
                # Use specified voice reference file with English language
                tts.tts_to_file(text=text, file_path=output_path, speaker_wav=voice, language="en")
            else:
                # For XTTS models, we need to provide a speaker reference
                # Try to use a built-in sample if available
                try:
                    # Check if the model has built-in speaker samples
                    if hasattr(tts, "synthesizer") and hasattr(tts.synthesizer, "tts_model") and \
                       hasattr(tts.synthesizer.tts_model, "speaker_manager") and \
                       hasattr(tts.synthesizer.tts_model.speaker_manager, "sample_wav"):
                        # Use the built-in sample
                        sample_voice = tts.synthesizer.tts_model.speaker_manager.sample_wav
                        tts.tts_to_file(text=text, file_path=output_path, speaker_wav=sample_voice, language="en")
                    else:
                        # Try to use a speaker from the model's speaker list if available
                        speakers = tts.speakers
                        if speakers and len(speakers) > 0:
                            # Use the first available speaker
                            tts.tts_to_file(text=text, file_path=output_path, speaker=speakers[0], language="en")
                        else:
                            # Download a sample reference file as last resort
                            import urllib.request
                            import tempfile
                            import urllib.error
                            
                            print("Downloading a sample reference voice...")
                            # List of potential sample URLs to try
                            sample_urls = [
                                "https://github.com/coqui-ai/TTS/raw/main/tests/inputs/ljspeech/LJ001-0001.wav",
                                "https://github.com/coqui-ai/TTS/raw/dev/tests/inputs/ljspeech/LJ001-0001.wav",
                                "https://raw.githubusercontent.com/coqui-ai/TTS/main/tests/inputs/ljspeech/LJ001-0001.wav",
                                "https://github.com/coqui-ai/TTS/raw/refs/heads/main/tests/inputs/example_1.wav"
                            ]
                            
                            sample_voice = None
                            for url in sample_urls:
                                try:
                                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                                        urllib.request.urlretrieve(url, temp_file.name)
                                        sample_voice = temp_file.name
                                        print(f"Successfully downloaded voice sample from {url}")
                                        break
                                except urllib.error.HTTPError as e:
                                    print(f"Failed to download from {url}: {e}")
                            
                            if sample_voice is None:
                                # If all downloads fail, create a simple sine wave as fallback
                                print("Creating a fallback sine wave audio file...")
                                import numpy as np
                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                                    sample_rate = 22050
                                    duration = 3  # seconds
                                    t = np.linspace(0, duration, int(sample_rate * duration))
                                    # Generate a simple sine wave
                                    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
                                    sf.write(temp_file.name, audio, sample_rate)
                                    sample_voice = temp_file.name
                            
                            tts.tts_to_file(text=text, file_path=output_path, speaker_wav=sample_voice, language="en")
                except Exception as inner_e:
                    print(f"Error with default voice selection: {inner_e}")
                    # Download a sample reference file as fallback
                    import urllib.request
                    import tempfile
                    import urllib.error
                    
                    print("Downloading a sample reference voice...")
                    # List of potential sample URLs to try
                    sample_urls = [
                        "https://github.com/coqui-ai/TTS/raw/main/tests/inputs/ljspeech/LJ001-0001.wav",
                        "https://github.com/coqui-ai/TTS/raw/dev/tests/inputs/ljspeech/LJ001-0001.wav",
                        "https://raw.githubusercontent.com/coqui-ai/TTS/main/tests/inputs/ljspeech/LJ001-0001.wav",
                        "https://github.com/coqui-ai/TTS/raw/refs/heads/main/tests/inputs/example_1.wav"
                    ]
                    
                    sample_voice = None
                    for url in sample_urls:
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                                urllib.request.urlretrieve(url, temp_file.name)
                                sample_voice = temp_file.name
                                print(f"Successfully downloaded voice sample from {url}")
                                break
                        except urllib.error.HTTPError as e:
                            print(f"Failed to download from {url}: {e}")
                    
                    if sample_voice is None:
                        # If all downloads fail, create a simple sine wave as fallback
                        print("Creating a fallback sine wave audio file...")
                        import numpy as np
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                            sample_rate = 22050
                            duration = 3  # seconds
                            t = np.linspace(0, duration, int(sample_rate * duration))
                            # Generate a simple sine wave
                            audio = 0.5 * np.sin(2 * np.pi * 440 * t)
                            sf.write(temp_file.name, audio, sample_rate)
                            sample_voice = temp_file.name
                    
                    tts.tts_to_file(text=text, file_path=output_path, speaker_wav=sample_voice, language="en")
            
            print(f"Speech generated and saved to {output_path}")
            return output_path
            
        except ImportError:
            print("Error: TTS library not found")
            print("Please install TTS: pip install TTS")
            print("Falling back to alternative TTS model...")
        except Exception as e:
            print(f"Error with coqui/xtts model: {e}")
            print("Falling back to alternative TTS model...")
    
    # Different handling based on model type
    if "speecht5" in model_name:
        try:
            # Check for sentencepiece
            import sentencepiece
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            import numpy as np
            
            # Load model, processor and vocoder
            processor = SpeechT5Processor.from_pretrained(model_name)
            model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Get speaker embeddings (for SpeechT5)
            if voice == "default":
                # Use a default speaker embedding
                speaker_embeddings = torch.zeros((1, 512)).to("cuda" if torch.cuda.is_available() else "cpu")
            else:
                # Load specific speaker embedding if provided
                speaker_embeddings = torch.load(voice).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Process input
            inputs = processor(text=text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Generate speech
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            
            # Save audio
            sf.write(output_path, speech.cpu().numpy(), 16000)
            
            print(f"Speech generated and saved to {output_path}")
            return output_path
            
        except ImportError as e:
            print(f"Error: {e}")
            print("Please install sentencepiece: pip install sentencepiece")
            print("Falling back to alternative TTS model...")
    
    # Try using VITS model as an alternative
    try:
        # Use a reliable model with proper imports
        from transformers import VitsModel, AutoTokenizer
        
        tts_model = "facebook/mms-tts-eng"  # Specific English model
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tts_model)
        model = VitsModel.from_pretrained(tts_model).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Process input
        inputs = tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate speech
        with torch.no_grad():
            output = model(**inputs).waveform
        
        # Save audio
        sf.write(output_path, output.cpu().numpy()[0], model.config.sampling_rate)
        
        print(f"Speech generated and saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error with facebook/mms-tts-eng model: {e}")
        print("Trying one more alternative model...")
        
        try:
            # Try a simpler model as last resort
            from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
            
            # Use a reliable model
            processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to("cuda" if torch.cuda.is_available() else "cpu")
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Use a default speaker embedding
            speaker_embeddings = torch.zeros((1, 512)).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Process input
            inputs = processor(text=text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Generate speech
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            
            # Save audio
            sf.write(output_path, speech.cpu().numpy(), 16000)
            
            print(f"Speech generated and saved to {output_path}")
            return output_path
            
        except Exception as e2:
            print(f"All TTS models failed. Last error: {e2}")
            print("Please install additional dependencies: pip install transformers[sentencepiece] torch soundfile TTS")
            return None

def main():
    # Example usage with default TTS model
    speak("Hello, world! This is a test of the text-to-speech system.")
    
    # Try with alternative model if the first one fails
    speak("This is a test with an alternative model.", model_name="espnet/kan-bayashi_ljspeech_vits")

    # Try with coqui/xtts model
    speak("Hello, this is a test using XTTS.", model_name="coqui/xtts-v2")
if __name__ == "__main__":
    main()
