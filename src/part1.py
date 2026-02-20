import os
import faster_whisper
import pandas as pd

referenced_file = os.path.join("..", "cv-corpus-24.0-2025-12-05", "he", "test.tsv")
model = faster_whisper.WhisperModel('ivrit-ai/whisper-large-v3-turbo-ct2')
base_clips_dir = os.path.join("..", "cv-corpus-24.0-2025-12-05", "he", "clips")

##
## Read the referenced file and extract the filename and the reference text
##

transciptions = []

df_in = pd.read_csv(referenced_file, sep='\t')

for _, row in df_in.iterrows():        
    transciptions.append({
        'filename': row['path'].split('.')[0],
        'reference_text': row['sentence']
    })

print(transciptions[:10])

##
## Transcribe the clips and save the transcriptions to a file
##

cnt = 0

print(f"Total of {len(transciptions)} clips to transcribe")

for transription in transciptions:
    if ((cnt+1) % 10 == 0):
        print(f"Clip {cnt+1} out of {len(transciptions)}")
    
    segs, _ = model.transcribe(
        os.path.join(base_clips_dir, f"{transription['filename']}.mp3"),
        language='he',
        temperature=0.0,
        beam_size=10)

    texts = [s.text for s in segs]

    transcribed_text = ' '.join(texts)
    transciptions[cnt]['transcribed_text'] = transcribed_text

    cnt+=1

df_out = pd.DataFrame(transciptions)

df_out.to_csv('transcriptions_new.tsv', sep='\t', index=False)
