import re
import pandas as pd
from statistics_df import StatisticsDF
from accuracy_statistics import AccuracyStatistics
from num2words import num2words
import phunspell
pspell = phunspell.Phunspell('he_IL')

def normalize_text(text):
    print(f"Before: {text}")

    text = text.lower()
    text = re.sub('[!?.,:;()"’\']', '', text)
    text = text.replace('%', ' אחוזים ')
    text = re.sub('[-–־]', ' ', text)
    
    numbers_in_text = re.findall(r'\d+', text)
    for number in numbers_in_text:
        text = text.replace(number, num2words(number, lang='he'))

    text = re.sub('[-–־]', ' ', text)

    # Remove Hebrew Nikkud
    text = re.sub('[\u0591-\u05C7]+', '', text)

    list_correct = []
    for word in text.split():
        if not pspell.lookup(word):
            corrected_word = next(pspell.suggest(word), word)
            print(f"-> '%s' corrected to '%s'" % (word, corrected_word))
            list_correct.append(corrected_word)
        else:
            list_correct.append(word)
    text = ' '.join(list_correct)
    
    list_connected_words = []
    prefix = False
    for current_word, next_word in zip(text.split(), text.split()[1:]):
        if len(current_word) == 1 and current_word in ['כ', 'ב', 'ש', 'ל', 'מ'] and not prefix:
            prefix = True
            list_connected_words.append(current_word + next_word)
        else:
            prefix = False
            list_connected_words.append(current_word)
    if not prefix:
        list_connected_words.append(text.split()[-1])
    text = ' '.join(list_connected_words)

    text = re.sub('[!?.,:;()"’\']', '', text)
    text = re.sub('[-–־]', ' ', text)
    text = text.replace(' התה ', ' הייתה ')

    print(f"After: {text}")

    return text

df_in = pd.read_csv('transcriptions.tsv', sep='\t')

statistics = []
normalized_text = []

statistics_total = AccuracyStatistics()

for _, row in df_in.iterrows():
    reference_text = normalize_text(row['reference_text']).split()
    transcribed_text = normalize_text(row['transcribed_text']).split()

    accuracy_statistics = AccuracyStatistics(reference_text, transcribed_text)
    statistics_total += accuracy_statistics
    statistics.append({'filename': row['filename'], **accuracy_statistics.to_dict()})
    normalized_text.append({'filename': row['filename'], 'reference_text': ' '.join(reference_text), 'transcribed_text': ' '.join(transcribed_text)})

for word_pair, num in statistics_total.frequent_errors(k=20):
    print('-> "%s" replaced by "%s" %d times.' %
    (word_pair[0], word_pair[1], num))

statistics_avg = {k: v / len(statistics) for k, v in statistics_total.to_dict().items()}
statistics.append({'filename': 'TOTAL', **statistics_total.to_dict()})
statistics.append({'filename': 'AVERAGE', **statistics_avg})

df_out = StatisticsDF(statistics)

df_out.display()

df_out.save('statistics_normalized.csv')

df_normalized = pd.DataFrame(normalized_text)
df_normalized.to_csv('normalized_text.tsv', index=False, sep='\t')