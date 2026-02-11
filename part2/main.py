import pandas as pd
import os
import re
from IPython.display import display 
from align_sequences import align_sequences, get_difference
from edit_weights import NestedUniformWeights
from accuracy_statistics import AccuracyStatistics


debug = False

df = pd.read_csv(os.path.join('..', 'transcriptions.tsv'), sep='\t')

statistics = []

statistics_total = AccuracyStatistics()

print(statistics_total.to_dict())

cnt = 0

# Function to format floats as integers if they are whole numbers, otherwise as floats
def format_as_int_if_whole(val):
    if type(val) not in [float, int]:
        return val
    if pd.isna(val):
        return '' # Handle NaN values as empty strings for display
    if float(val).is_integer():
        return f'{int(val)}'
    else:
        return f'{val:.3f}' # Format non-whole numbers to one decimal place

for _, row in df.iterrows():
    #if cnt == 10:
    #    break

    reference_text = row['reference_text'].split()
    transcribed_text = row['transcribed_text'].split()

    align_score, aligned_pairs = align_sequences(reference_text, transcribed_text, NestedUniformWeights())
   
    if debug:
        print(f"Reference text: {reference_text}")
        print(f"Transcribed text: {transcribed_text}")
        print(f"Alignment score: {align_score}")
        print(f"Aligned pairs: {aligned_pairs}")
        print(f"Differences: {get_difference(aligned_pairs)}")
        print(f"Count of differences: {len(get_difference(aligned_pairs))}")
 
    accuracy_statistics = AccuracyStatistics(reference_text, transcribed_text)
    statistics_total += accuracy_statistics
    statistics.append({'filename': row['filename'], **accuracy_statistics.to_dict()})

    cnt += 1

statistics.append({'filename': 'TOTAL', **statistics_total.to_dict()})

statistics_avg = {k: v / cnt for k, v in statistics_total.to_dict().items()}
statistics.append({'filename': 'AVERAGE', **statistics_avg})

df_statistics = pd.DataFrame(statistics)
df_statistics = df_statistics.reset_index(drop=True)

for column in df_statistics:
    df_statistics[column] = df_statistics[column].apply(format_as_int_if_whole)

display(df_statistics)

df_statistics.to_csv(os.path.join('..', 'statistics.csv'), sep=',', index=False)