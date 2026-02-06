import csv
import os
import faster_whisper
import pandas as pd

referenced_file = os.path.join("cv-corpus-24.0-2025-12-05", "he", "test.tsv")

transciptions = []

cnt=0

with open(referenced_file, encoding="utf-8") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in list(rd)[1:]:
        #if cnt == 10:
        #    break
        
        transciptions.append({
            'filename': row[1].split('.')[0],
            'reference_text': row[3]
        })

        cnt+=1

print(transciptions)


model = faster_whisper.WhisperModel('ivrit-ai/whisper-large-v3-turbo-ct2')

base_clips_dir = os.path.join("cv-corpus-24.0-2025-12-05", "he", "clips")

cnt = 0

print(f"Total of {len(transciptions)} clips to transcribe")

for transription in transciptions:
    #if cnt == 10:
    #    break

    if ((cnt+1) % 10 == 0):
        print(f"Clip {cnt+1} out of {len(transciptions)}")
    
    segs, _ = model.transcribe(os.path.join(base_clips_dir, f"{transription['filename']}.mp3"), language='he')

    texts = [s.text for s in segs]

    transcribed_text = ' '.join(texts)
    transciptions[cnt]['transcribed_text'] = transcribed_text

    cnt+=1

df = pd.DataFrame(transciptions)

df.to_csv('transcriptions.tsv', index=False)
