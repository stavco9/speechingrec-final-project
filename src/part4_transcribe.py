#!/usr/bin/env python3
"""
Part 4: Transcribe Noisy Audio Files
Transcribes all noisy audio files using Whisper model
"""

import os
import faster_whisper

# Configuration
TEST_TSV = "../cv-corpus-24.0-2025-12-05/he/test.tsv"
NOISY_CLIPS_DIR = "../musan/noisy_clips"
OUTPUT_FILE = "results/part4_noisy_transcriptions.tsv"

# Load Whisper model
model = faster_whisper.WhisperModel('ivrit-ai/whisper-large-v3-turbo-ct2')

# Read test.tsv to get filenames and reference texts
transcriptions = []
with open(TEST_TSV, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Parse header
header = lines[0].strip().split('\t')
path_idx = header.index('path')
sentence_idx = header.index('sentence')

# Extract filenames and references
for line in lines[1:]:
    fields = line.strip().split('\t')
    if len(fields) > max(path_idx, sentence_idx):
        filename = fields[path_idx].split('.')[0]  # Remove extension
        reference_text = fields[sentence_idx]
        transcriptions.append({
            'filename': filename,
            'reference_text': reference_text
        })

# Transcribe each noisy file
for i, transcription in enumerate(transcriptions, 1):
    filename = transcription['filename']
    audio_path = os.path.join(NOISY_CLIPS_DIR, f"{filename}.wav")

    try:
        # Transcribe
        segments, _ = model.transcribe(audio_path, language='he')
        texts = [s.text for s in segments]
        transcribed_text = ' '.join(texts)
        transcription['transcribed_text'] = transcribed_text

    except Exception as e:
        print(f"Error transcribing {filename}: {e}")
        transcription['transcribed_text'] = ""

# Write output TSV
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    # Write header
    f.write("Filename\tReference Text\tTranscribed Text\n")

    # Write data
    for transcription in transcriptions:
        f.write(f"{transcription['filename']}\t{transcription['reference_text']}\t{transcription['transcribed_text']}\n")