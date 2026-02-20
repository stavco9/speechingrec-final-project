
import os
import faster_whisper
import yake
import pandas as pd

referenced_file = os.path.join("..", "cv-corpus-24.0-2025-12-05", "he", "test.tsv")
model = faster_whisper.WhisperModel('ivrit-ai/whisper-large-v3-turbo-ct2')
base_clips_dir = os.path.join("..", "cv-corpus-24.0-2025-12-05", "he", "clips")


#kw_extractor = yake.KeywordExtractor(lan="he", n=1, top=5, stopwords=None)

#keywords = kw_extractor.extract_keywords(sentence)
#prompt_string = " ".join([kw[0] for kw in keywords])
#print(f"Prompt string: {prompt_string}")

files = [
    "common_voice_he_38094124.mp3",
    "common_voice_he_38085646.mp3",
    "common_voice_he_38196123.mp3",
    "common_voice_he_38118861.mp3"
]

for file in files:
    segs, _ = model.transcribe(
        os.path.join(base_clips_dir, file),
        language='he',
        temperature=0.0,
        beam_size=10,
        condition_on_previous_text=False,
        initial_prompt="השתמש בעברית ספרותית ורהוטה עם מילים ותחביר הגיוניים")

    texts = [s.text for s in segs]

    transcribed_text = ' '.join(texts)

    print(transcribed_text)
