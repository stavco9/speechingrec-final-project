import pandas as pd
import os
from modules.statistics_df import StatisticsDF
from modules.accuracy_statistics import AccuracyStatistics

# Read the transcriptions file
df_in = pd.read_csv(os.path.join('results', 'part1_transcriptions.tsv'), sep='\t')

# List to store the statistics
statistics = []

# Create a new AccuracyStatistics object
statistics_total = AccuracyStatistics()

# Iterate over the rows of the transcriptions file
for _, row in df_in.iterrows():

    # Split the reference text and the transcribed text into words
    reference_text = row['reference_text'].split()
    transcribed_text = row['transcribed_text'].split()

    # Create a new AccuracyStatistics object for the current row
    accuracy_statistics = AccuracyStatistics(reference_text, transcribed_text)
    statistics_total += accuracy_statistics
    statistics.append({'filename': row['filename'], **accuracy_statistics.to_dict()})

# Iterate over the frequent errors and print the most frequent errors
for word_pair, num in statistics_total.frequent_errors(k=20):
    print('-> "%s" replaced by "%s" %d times.' %
    (word_pair[0], word_pair[1], num))

# Calculate the average statistics and add the total and average statistics to the list of statistics
statistics_avg = {k: v / len(statistics) for k, v in statistics_total.to_dict().items()}
statistics.append({'filename': 'TOTAL', **statistics_total.to_dict()})
statistics.append({'filename': 'AVERAGE', **statistics_avg})

# Create a new StatisticsDF object from the list of statistics and display and save it
df_out = StatisticsDF(statistics)
df_out.display()
df_out.save(os.path.join('results', 'part2_statistics.csv'))