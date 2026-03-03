import os
import faster_whisper
import pandas as pd


class Part1:
    def __init__(self, referenced_file: str, base_clips_dir: str, output_file: str):
        # Set the referenced file and the base clips directory
        self.referenced_file = referenced_file
        self.base_clips_dir = base_clips_dir
        self.output_file = output_file
        
        # Load the Whisper model
        self.model = faster_whisper.WhisperModel('ivrit-ai/whisper-large-v3-turbo-ct2')

        # List to store the transcriptions
        self.transciptions = []

    def read_referenced_file(self):
        # Read the Ground Truth file
        df_in = pd.read_csv(self.referenced_file, sep='\t')

        # Get the video format from the first row of the Ground Truth file
        video_format = df_in.iloc[0]['path'].split('.')[-1]
        print(f"Video format: {video_format}")

        # Iterate over the rows of the Ground Truth file
        for _, row in df_in.iterrows():        
            # Add the filename and the reference text to the list of transcriptions
            self.transciptions.append({
                'filename': row['path'].split('.')[0],
                'reference_text': row['sentence']
            })

        print(self.transciptions[:10])

        return video_format

    def transcribe_clips(self, video_format: str, limit: int = 0):
        cnt = 0

        if limit == 0:
            limit = len(self.transciptions)

        print(f"Total of {limit} clips to transcribe")

        # Iterate over the transcriptions and transcribe the clips
        for transription in self.transciptions[:limit]:
            if ((cnt+1) % 10 == 0):
                print(f"Clip {cnt+1} out of {limit}")
            
            # Transcribe the clip
            segs, _ = self.model.transcribe(
                os.path.join(self.base_clips_dir, f"{transription['filename']}.{video_format}"),
                language='he',
                temperature=0.0,
                beam_size=10)

            # Get the text from the transcriptions
            texts = [s.text for s in segs]

            # Join the text from the transcriptions into a single string
            transcribed_text = ' '.join(texts)

            # Add the transcribed text to the transcription
            self.transciptions[cnt]['transcribed_text'] = transcribed_text

            cnt+=1

    def save_transcriptions(self):
        # Create a DataFrame from the transcriptions
        df_out = pd.DataFrame(self.transciptions)

        # Save the DataFrame to a TSV file
        df_out.to_csv(self.output_file, sep='\t', index=False)

def main():
    # Create a new Part1 object
    part1 = Part1(
        referenced_file=os.path.join('..', 'cv-corpus-24.0-2025-12-05', 'he', 'test.tsv'),
        base_clips_dir=os.path.join('..', 'cv-corpus-24.0-2025-12-05', 'he', 'clips'),
        output_file=os.path.join('results', 'part1_transcriptions.tsv')
    )

    video_format = part1.read_referenced_file()
    part1.transcribe_clips(video_format)
    part1.save_transcriptions()

if __name__ == "__main__":
    main()