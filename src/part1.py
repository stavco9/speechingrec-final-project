import os
import faster_whisper
import pandas as pd

# Read the Ground Truth file
referenced_file = os.path.join("..", "cv-corpus-24.0-2025-12-05", "he", "test.tsv")

# Load the Whisper model
model = faster_whisper.WhisperModel('ivrit-ai/whisper-large-v3-turbo-ct2')

# Set the base directory for the clips
base_clips_dir = os.path.join("..", "cv-corpus-24.0-2025-12-05", "he", "clips")

# List to store the transcriptions
transciptions = []

# Read the Ground Truth file
df_in = pd.read_csv(referenced_file, sep='\t')

# Iterate over the rows of the Ground Truth file
for _, row in df_in.iterrows():        
    # Add the filename and the reference text to the list of transcriptions
    transciptions.append({
        'filename': row['path'].split('.')[0],
        'reference_text': row['sentence']
    })

print(transciptions[:10])

cnt = 0

print(f"Total of {len(transciptions)} clips to transcribe")

# Iterate over the transcriptions and transcribe the clips
for transription in transciptions:
    if ((cnt+1) % 10 == 0):
        print(f"Clip {cnt+1} out of {len(transciptions)}")
    
    # Transcribe the clip
    segs, _ = model.transcribe(
        os.path.join(base_clips_dir, f"{transription['filename']}.mp3"),
        language='he',
        temperature=0.0,
        beam_size=10)

    # Get the text from the transcriptions
    texts = [s.text for s in segs]

    # Join the text from the transcriptions into a single string
    transcribed_text = ' '.join(texts)

    # Add the transcribed text to the transcription
    transciptions[cnt]['transcribed_text'] = transcribed_text

    cnt+=1

# Create a DataFrame from the transcriptions
df_out = pd.DataFrame(transciptions)

# Save the DataFrame to a TSV file
df_out.to_csv(os.path.join('results', 'part1_transcriptions.tsv'), sep='\t', index=False)
