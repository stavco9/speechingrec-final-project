import pandas as pd
from modules.statistics_df import StatisticsDF
from modules.accuracy_statistics import AccuracyStatistics
from modules.normalize_text import NormalizeText
import os

class Part2:
    def __init__(self, input_transcriptions_file: str, output_statistics_file: str, output_transcriptions_file: str):
        self.input_transcriptions_file = input_transcriptions_file
        self.output_statistics_file = output_statistics_file
        self.output_transcriptions_file = output_transcriptions_file

        # Create a new NormalizeText object
        self.normalize = NormalizeText()

        self.statistics = []
        self.normalized_text = []

    def process_transcriptions(self, to_normalize: bool = False):
        df_in = pd.read_csv(self.input_transcriptions_file, sep='\t')

        # Create a new AccuracyStatistics object
        statistics_total = AccuracyStatistics()

        # Iterate over the rows of the transcriptions file
        for index, row in df_in.iterrows():

            if to_normalize:
                # Normalize the reference text
                reference_text = self.normalize.normalize_text(
                    text=row['reference_text'], 
                    cnt=index+1, 
                    type_of_text='Reference'
                ).split()
                
                # Normalize the transcribed text
                transcribed_text = self.normalize.normalize_text(
                    text=row['transcribed_text'], 
                    cnt=index+1, 
                    type_of_text='Transcribed'
                ).split()
            else:
                reference_text = row['reference_text'].split()
                transcribed_text = row['transcribed_text'].split()


            # Create a new AccuracyStatistics object for the current row
            accuracy_statistics = AccuracyStatistics(reference_text, transcribed_text)
            statistics_total += accuracy_statistics
            self.statistics.append({'filename': row['filename'], **accuracy_statistics.to_dict()})
            
            if to_normalize:
                self.normalized_text.append({
                    'filename': row['filename'],
                    'reference_text': ' '.join(reference_text),
                    'transcribed_text': ' '.join(transcribed_text)
                })

        return statistics_total

    def save_statistics(self, statistics_total: AccuracyStatistics):
        # Calculate the average statistics
        statistics_avg = {k: v / len(self.statistics) for k, v in statistics_total.to_dict().items()}
        self.statistics.append({'filename': 'TOTAL', **statistics_total.to_dict()})
        self.statistics.append({'filename': 'AVERAGE', **statistics_avg})

        # Create a new StatisticsDF object from the list of statistics and display and save it
        df_out = StatisticsDF(self.statistics)
        df_out.display()
        df_out.save(self.output_statistics_file)

        if len(self.normalized_text) > 0:
            # Create a new DataFrame from the normalized text and save it to a TSV file
            df_normalized = pd.DataFrame(self.normalized_text)
            df_normalized.to_csv(self.output_transcriptions_file, index=False, sep='\t')


def main():
    part2 = Part2(
        input_transcriptions_file=os.path.join('results', 'part1_transcriptions.tsv'),
        output_statistics_file=os.path.join('results', 'part2_statistics.csv'),
        output_transcriptions_file=os.path.join('results', 'part2_transcriptions.tsv')
    )
    statistics_total = part2.process_transcriptions(to_normalize=False)

    # Iterate over the frequent errors and print the most frequent errors
    for word_pair, num in statistics_total.frequent_errors(k=20):
        print('-> "%s" replaced by "%s" %d times.' %
        (word_pair[0], word_pair[1], num))

    part2.save_statistics(statistics_total)

if __name__ == "__main__":
    main()