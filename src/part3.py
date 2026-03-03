import pandas as pd
import os
from modules.statistics_df import StatisticsDF
from modules.accuracy_statistics import AccuracyStatistics
from modules.normalize_text import NormalizeText

# Create a new NormalizeText object
normalize = NormalizeText()

# Read the transcriptions file
df_in = pd.read_csv(os.path.join('results', 'part1_transcriptions.tsv'), sep='\t')

# List to store the statistics
statistics = []
normalized_text = []

# Create a new AccuracyStatistics object
statistics_total = AccuracyStatistics()

# Iterate over the rows of the transcriptions file
for index, row in df_in.iterrows():

    # Normalize the reference text
    reference_text = normalize.normalize_text(
        text=row['reference_text'], 
        cnt=index+1, 
        type_of_text='Reference'
    ).split()
    
    # Normalize the transcribed text
    transcribed_text = normalize.normalize_text(
        text=row['transcribed_text'], 
        cnt=index+1, 
        type_of_text='Transcribed'
    ).split()

    # Create a new AccuracyStatistics object for the current row
    accuracy_statistics = AccuracyStatistics(reference_text, transcribed_text)
    statistics_total += accuracy_statistics
    statistics.append({'filename': row['filename'], **accuracy_statistics.to_dict()})
    normalized_text.append({
        'filename': row['filename'],
        'reference_text': ' '.join(reference_text),
        'transcribed_text': ' '.join(transcribed_text)
    })

# Iterate over the frequent errors and print the most frequent errors
for word_pair, num in statistics_total.frequent_errors():
    print('-> "%s" replaced by "%s" %d times.' %
    (word_pair[0], word_pair[1], num))

# Calculate the average statistics
statistics_avg = {k: v / len(statistics) for k, v in statistics_total.to_dict().items()}
statistics.append({'filename': 'TOTAL', **statistics_total.to_dict()})
statistics.append({'filename': 'AVERAGE', **statistics_avg})

# Create a new StatisticsDF object from the list of statistics and display and save it
df_out = StatisticsDF(statistics)
df_out.display()
df_out.save(os.path.join('results', 'part3_statistics.csv'))

# Create a new DataFrame from the normalized text and save it to a TSV file

df_normalized = pd.DataFrame(normalized_text)
df_normalized.to_csv(os.path.join('results', 'part3_transcriptions.tsv'), index=False, sep='\t')