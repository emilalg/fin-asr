import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from pywer import wer, cer
import time
import pandas as pd
import torchaudio
import numpy as np

from model.model import LSTMCTC
from alphabet import Alphabet
from data.huggingface import HuggingFaceDataset
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load custom model
def load_custom_model(model_path, device):
    alphabet = Alphabet()
    model = LSTMCTC(
        input_size=40,
        hidden_size=320,
        num_layers=4,
        num_classes=len(alphabet.alphabet),
        dropout_rate=0.2
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, alphabet

# Load Hugging Face model
def load_hf_model(model_name, device):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor

# Preprocessing function
mel_spectrogram_converter = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    n_mels=40
)

def preprocess_audio(audio, sr):
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    
    audio = torchaudio.functional.preemphasis(audio)
    mel_spectrogram = mel_spectrogram_converter(audio)
    mel_spectrogram = torch.log(mel_spectrogram + 1e-9)
    mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()
    mel_spectrogram = mel_spectrogram.transpose(0, 1)
    return mel_spectrogram

# Evaluation function for custom model
def evaluate_custom_model(model, alphabet, audio, sr):
    mel_spectrogram = preprocess_audio(audio, sr)
    mel_spectrogram = mel_spectrogram.unsqueeze(0).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(mel_spectrogram)
        log_probs = F.log_softmax(output, dim=-1)
        decoded_text = alphabet.decode(log_probs, remove_blanks=True)[0]
    end_time = time.time()
    
    return decoded_text, end_time - start_time

# Evaluation function for Hugging Face model
def evaluate_hf_model(model, processor, audio, sr):
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    
    input_features = processor(audio.numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(device)
    
    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(input_features=input_features, language="fi", task="transcribe")
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    end_time = time.time()
    
    return transcription, end_time - start_time

# Main evaluation loop
def main():
    # Load models
    custom_model_path = "gd.pth"  # Adjust this path
    custom_model, alphabet = load_custom_model(custom_model_path, device)
    hf_model, hf_processor = load_hf_model("Finnish-NLP/whisper-large-finnish-v3", device)

    # Load dataset
    dataset = HuggingFaceDataset(token='your_token_here', predl=True, data_dir='data/hf')  # Adjust parameters as needed
    test_dataset = dataset.test_dataset

    results = []
    num_samples = len(test_dataset)

    for i in tqdm(range(num_samples), desc="Evaluating"):
        sample = test_dataset[i]
        ground_truth = sample['sentence']
        audio = torch.tensor(sample['audio']['array']).float()
        sr = sample['audio']['sampling_rate']

        # Evaluate custom model
        custom_output, custom_time = evaluate_custom_model(custom_model, alphabet, audio, sr)
        custom_wer = wer([ground_truth], [custom_output])
        custom_cer = cer([ground_truth], [custom_output])

        # Evaluate Hugging Face model
        hf_output, hf_time = evaluate_hf_model(hf_model, hf_processor, audio, sr)
        hf_wer = wer([ground_truth], [hf_output])
        hf_cer = cer([ground_truth], [hf_output])

        results.append({
            'sample_id': i,
            'custom_time': custom_time,
            'custom_wer': custom_wer,
            'custom_cer': custom_cer,
            'hf_time': hf_time,
            'hf_wer': hf_wer,
            'hf_cer': hf_cer
        })

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Calculate and add average metrics
    df['avg_custom_time'] = df['custom_time'].mean()
    df['avg_custom_wer'] = df['custom_wer'].mean()
    df['avg_custom_cer'] = df['custom_cer'].mean()
    df['avg_hf_time'] = df['hf_time'].mean()
    df['avg_hf_wer'] = df['hf_wer'].mean()
    df['avg_hf_cer'] = df['hf_cer'].mean()

    # Save results to CSV
    csv_filename = 'model_comparison_results_best.csv'
    df.to_csv(csv_filename, index=False)
    logging.info(f"Results saved to {csv_filename}")

    # Print summary
    print("\nEvaluation Summary:")
    print(f"Number of samples: {num_samples}")
    print(f"Custom Model - Avg Time: {df['avg_custom_time'].iloc[0]:.4f}s, Avg WER: {df['avg_custom_wer'].iloc[0]:.4f}, Avg CER: {df['avg_custom_cer'].iloc[0]:.4f}")
    print(f"HuggingFace Model - Avg Time: {df['avg_hf_time'].iloc[0]:.4f}s, Avg WER: {df['avg_hf_wer'].iloc[0]:.4f}, Avg CER: {df['avg_hf_cer'].iloc[0]:.4f}")

if __name__ == "__main__":
    main()