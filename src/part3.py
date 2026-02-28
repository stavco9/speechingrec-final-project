import pandas as pd
import os
from modules.statistics_df import StatisticsDF
from modules.accuracy_statistics import AccuracyStatistics
from modules.normalize_text import NormalizeText

normalize = NormalizeText()

df_in = pd.read_csv(os.path.join('results', 'part1_transcriptions.tsv'), sep='\t')

statistics = []
normalized_text = []

statistics_total = AccuracyStatistics()

for index, row in df_in.iterrows():
    reference_text = normalize.normalize_text(
        text=row['reference_text'], 
        cnt=index+1, 
        type_of_text='Reference'
    ).split()
    transcribed_text = normalize.normalize_text(
        text=row['transcribed_text'], 
        cnt=index+1, 
        type_of_text='Transcribed'
    ).split()

    accuracy_statistics = AccuracyStatistics(reference_text, transcribed_text)
    statistics_total += accuracy_statistics
    statistics.append({'filename': row['filename'], **accuracy_statistics.to_dict()})
    normalized_text.append({
        'filename': row['filename'],
        'reference_text': ' '.join(reference_text),
        'transcribed_text': ' '.join(transcribed_text)
    })

for word_pair, num in statistics_total.frequent_errors():
    print('-> "%s" replaced by "%s" %d times.' %
    (word_pair[0], word_pair[1], num))

statistics_avg = {k: v / len(statistics) for k, v in statistics_total.to_dict().items()}

df_out = StatisticsDF(statistics)
df_out = df_out.sort_values(by=['wer'], ascending=False)
df_out_files = df_out.df['filename'].tolist()

filename_to_index = {filename: index for index, filename in enumerate(df_out_files)}
normalized_text = sorted(normalized_text, key=lambda x: filename_to_index.get(x['filename'], float('inf')))

df_summary = StatisticsDF([
   {'filename': 'TOTAL', **statistics_total.to_dict()},
   {'filename': 'AVERAGE', **statistics_avg}
])

df_out = df_out.concat(df_summary)

df_out.display()

df_out.save(os.path.join('results', 'part3_statistics.csv'))

df_normalized = pd.DataFrame(normalized_text)
df_normalized.to_csv(os.path.join('results', 'part3_transcriptions.tsv'), index=False, sep='\t')

with open(os.path.join('results', 'part3_corrections_tmp.txt'), 'w', encoding='utf-8') as f:
    for cnt, word, corrected_word in normalize.corrections:
        f.write(f"{str(cnt)}) {word} -> {corrected_word}\n")