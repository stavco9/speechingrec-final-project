#!/usr/bin/env python3
"""
Part 4: Background Noise Augmentation Script
Adds background noise to CommonVoice test set audio files

Configuration for digits modulo 6 = 0:
- Signal type: Noise (רעש)
- Strength: Strong (חזק)
- SNR range: 0-6 dB
"""

import os
import random
import numpy as np
import soundfile as sf
from scipy.signal import decimate


# Configuration
TEST_TSV = "cv-corpus-24.0-2025-12-05/he/test.tsv"
CLIPS_DIR = "cv-corpus-24.0-2025-12-05/he/clips"
NOISE_DIR = "noise/free-sound"
OUTPUT_DIR = "noisy_clips"
LOG_FILE = "part4_augmentation_log.tsv"

# Settings for result 0 (digits modulo 6)
SIGNAL_TYPE = "noise"  # רעש
SNR_MIN = 0  # dB
SNR_MAX = 6  # dB
MIN_NOISE_DURATION = 30  # seconds


def find_long_noise_files(noise_dir, min_duration=30):
    """Find noise files longer than min_duration seconds"""
    noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]
    long_files = []

    for noise_file in noise_files:
        filepath = os.path.join(noise_dir, noise_file)
        try:
            info = sf.info(filepath)
            duration = info.duration
            if duration > min_duration:
                long_files.append(noise_file)
        except Exception as e:
            print(f"Warning: Could not read {noise_file}: {e}")

    return long_files


def calculate_rms(signal):
    """Calculate Root Mean Square (RMS) of signal"""
    return np.sqrt(np.mean(signal ** 2))


def add_noise_at_snr(speech, noise, target_snr_db):
    """
    Add noise to speech signal at specified SNR

    SNR (dB) = 20 * log10(RMS_signal / RMS_noise)

    Returns: mixed signal
    """
    # Calculate RMS of speech and noise
    rms_speech = calculate_rms(speech)
    rms_noise = calculate_rms(noise)

    # Calculate required noise scaling factor
    # SNR_db = 20 * log10(rms_speech / rms_noise_scaled)
    # 10^(SNR_db/20) = rms_speech / rms_noise_scaled
    # rms_noise_scaled = rms_speech / 10^(SNR_db/20)
    # scaling_factor = rms_noise_scaled / rms_noise

    target_rms_noise = rms_speech / (10 ** (target_snr_db / 20))
    scaling_factor = target_rms_noise / rms_noise if rms_noise > 0 else 0

    # Scale and add noise
    scaled_noise = noise * scaling_factor
    mixed = speech + scaled_noise

    return mixed


def process_audio_file(audio_path, noise_files, noise_dir, output_dir):
    """
    Process a single audio file: add background noise

    Returns: (background_file, start_point_sec, snr_db, output_path)
    """
    # Read speech audio (CommonVoice: 32kHz mono MP3)
    speech, sr = sf.read(audio_path)

    # Downsample from 32kHz to 16kHz to match Musan noise files
    if sr == 32000:
        speech = decimate(speech, 2)
        sr = 16000

    # Select random noise file
    noise_file = random.choice(noise_files)
    noise_path = os.path.join(noise_dir, noise_file)

    # Read noise file (Musan: 16kHz mono WAV, >30 seconds)
    noise, noise_sr = sf.read(noise_path)

    # Select random starting point in noise
    speech_samples = len(speech)
    max_start_sample = len(noise) - speech_samples
    start_sample = random.randint(0, max_start_sample)
    start_point_sec = start_sample / noise_sr

    # Extract noise segment of exact same length as speech
    noise_segment = noise[start_sample:start_sample + speech_samples]

    # Random SNR in range
    snr_db = random.uniform(SNR_MIN, SNR_MAX)

    # Add noise at target SNR
    mixed = add_noise_at_snr(speech, noise_segment, snr_db)

    # Create output filename
    basename = os.path.basename(audio_path)
    filename_without_ext = os.path.splitext(basename)[0]
    output_path = os.path.join(output_dir, filename_without_ext + ".wav")

    # Save mixed audio
    sf.write(output_path, mixed, sr)

    return noise_file, start_point_sec, snr_db, output_path


def main():
    """Main function to process all test files"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    noise_files = find_long_noise_files(NOISE_DIR, MIN_NOISE_DURATION)

    if len(noise_files) == 0:
        print("Error: No suitable noise files found!")
        return

    with open(TEST_TSV, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header = lines[0].strip().split('\t')
    path_idx = header.index('path')

    test_files = []
    for line in lines[1:]:
        fields = line.strip().split('\t')
        if len(fields) > path_idx:
            filename = fields[path_idx]
            test_files.append(filename)

    log_entries = []
    log_entries.append(['Filename', 'Background file', 'Start point (in seconds)', 'SNR'])

    for filename in test_files:
        audio_path = os.path.join(CLIPS_DIR, filename)

        if not os.path.exists(audio_path):
            print(f"Warning: File not found: {audio_path}")
            continue

        try:
            noise_file, start_sec, snr_db, output_path = process_audio_file(
                audio_path, noise_files, NOISE_DIR, OUTPUT_DIR
            )

            clean_filename = os.path.splitext(filename)[0]

            log_entries.append([
                clean_filename,
                noise_file,
                f"{start_sec:.2f}",
                f"{snr_db:.2f}"
            ])

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        for entry in log_entries:
            f.write('\t'.join(entry) + '\n')


if __name__ == "__main__":
    main()
