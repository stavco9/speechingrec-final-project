import pandas as pd
from statistics_df import StatisticsDF
from accuracy_statistics import AccuracyStatistics

df_in = pd.read_csv('transcriptions.tsv', sep='\t')

statistics = []

statistics_total = AccuracyStatistics()

for _, row in df_in.iterrows():
    reference_text = row['reference_text'].split()
    transcribed_text = row['transcribed_text'].split()

    accuracy_statistics = AccuracyStatistics(reference_text, transcribed_text)
    statistics_total += accuracy_statistics
    statistics.append({'filename': row['filename'], **accuracy_statistics.to_dict()})

for word_pair, num in statistics_total.frequent_errors(k=20):
    print('-> "%s" replaced by "%s" %d times.' %
    (word_pair[0], word_pair[1], num))

statistics_avg = {k: v / len(statistics) for k, v in statistics_total.to_dict().items()}
statistics.append({'filename': 'TOTAL', **statistics_total.to_dict()})
statistics.append({'filename': 'AVERAGE', **statistics_avg})

df_out = StatisticsDF(statistics)

df_out.display()

df_out.save('statistics.csv')