import gradio as gr
import librosa
import torch
import torchaudio
import logging
import numpy as np
from model.model import LSTMCTC
from alphabet import Alphabet

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Initialize mel spectrogram converter - exactly as in training
mel_spectrogram_converter = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    n_mels=40
)

def load_model():
    """Load the custom LSTM-CTC model"""
    alphabet = Alphabet()
    model = LSTMCTC(
        input_size=40,
        hidden_size=320,
        num_layers=4,
        num_classes=len(alphabet.alphabet),
        dropout_rate=0.2
    ).to(device)
    model.load_state_dict(torch.load("gd.pth", weights_only=True))
    model.eval()
    return model, alphabet


def preprocess_audio(audio_array, sr):
    """Preprocess audio data for model input - matching training preprocessing"""
    try:
        logging.info(f"Input audio shape: {audio_array.shape}, dtype: {audio_array.dtype}")
        
        # Normalize input shape
        t1 = audio_array
        logging.info(f"Normalized audio shape: {t1.shape}")
        
        # Resample if necessary
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            t1 = resampler(t1)
            logging.info(f"Audio shape after resampling: {t1.shape}")
        
        # Apply preemphasis
        t1 = torchaudio.functional.preemphasis(t1)
        logging.info(f"Shape after preemphasis: {t1.shape}")
        
        # Convert to mel spectrogram
        mel_spectrogram = mel_spectrogram_converter(t1)
        logging.info(f"Shape after mel spectrogram: {mel_spectrogram.shape}")
        
        # Apply log
        mel_spectrogram = torch.log(mel_spectrogram + 1e-9)
        
        # Normalize
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()
        
        # Transpose to (time, mels) - exactly as in training code
        mel_spectrogram = mel_spectrogram.transpose(0, 1)
        logging.info(f"Shape after transpose: {mel_spectrogram.shape}")
        
        # Add batch dimension for model
        mel_spectrogram = mel_spectrogram.unsqueeze(0)
        logging.info(f"Final shape: {mel_spectrogram.shape}")
        
        return mel_spectrogram
        
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}, Audio shape: {audio_array.shape if isinstance(audio_array, (torch.Tensor, np.ndarray)) else 'unknown'}")
        raise

def transcribe_audio(audio_data):
    """Process audio and return transcription"""
    try:
        if audio_data is None:
            return "No audio detected. Please record or upload audio."
            
        sr, audio_array = audio_data
        logging.info(f"Received audio - Sample rate: {sr}, Shape: {np.array(audio_array).shape}")
        audio_array = audio_array.astype('float32').T
        logging.info(f"Modified audio1 - Sample rate: {sr}, Shape: {audio_array.shape}")
        audio_array = librosa.to_mono(audio_array)
        audio_array = torch.tensor(audio_array, dtype=torch.float32)
        logging.info(f"Modified audio2 - Sample rate: {sr}, Shape: {audio_array.shape}")
        
        if len(audio_array) == 0:
            return "Audio is empty. Please record or upload valid audio."
        
        # Load model
        model, alphabet = load_model()
        
        # Preprocess audio
        mel_spectrogram = preprocess_audio(audio_array, sr)
        mel_spectrogram = mel_spectrogram.to(device)
        
        # Generate transcription
        with torch.no_grad():
            output = model(mel_spectrogram)
            log_probs = torch.nn.functional.log_softmax(output, dim=-1)
            transcription = alphabet.decode(log_probs, remove_blanks=True)[0]
        
        return transcription
    
    except Exception as e:
        logging.error(f"Error processing audio: {str(e)}")
        return f"Error processing audio: {str(e)}"

def create_gradio_interface():
    """Create and launch the Gradio interface"""
    iface = gr.Interface(
        fn=transcribe_audio,
        inputs=gr.Audio(
            sources=["microphone", "upload"],
            label="Tallenna tai lataa ääntä.",
            type="numpy",
            streaming=False
        ),
        outputs=gr.Textbox(label="Transcription"),
        examples=['examples/audio1.mp3','examples/audio2.mp3'],
        title="Puheentunnistus demo",
        description="Tallenna tai lataa ääntä puhtaaksikirjoitettavaksi.",
        cache_examples=True
    )
    return iface

if __name__ == "__main__":
    # Create and launch the interface
    iface = create_gradio_interface()
    iface.launch(share=True)