import re
import pandas as pd
import os
from statistics_df import StatisticsDF
from accuracy_statistics import AccuracyStatistics
from num2words import num2words
from phunspell import Phunspell
from transformers import AutoTokenizer, AutoModel 
import torch

# Load Dicta's morphological model
tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-lex')
model = AutoModel.from_pretrained('dicta-il/dictabert-lex', trust_remote_code=True)
#tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-parse')
#model = AutoModel.from_pretrained('dicta-il/dictabert-large-parse', trust_remote_code=True)
model.eval()

#phunspell_storage_path = os.path.join(os.path.dirname(__file__), '..', 'phunspell-dict')
pspell = Phunspell('he_IL')

def get_lemmas(text):
    # This returns a dictionary with various morphological data
    output = model.predict([text], tokenizer)
    
    # Extract the 'lemma' for each token
    lemmas = [token['lex'] for sentence in output for token in sentence['tokens']]
    return " ".join(lemmas)

def get_base_forms(text):
    # This returns the 'lexemes' (the standardized base form)
    predictions = model.predict([text], tokenizer)
    # Extract the 'lex' field for each word
    return " ".join([token[1] for token in predictions[0]])

def correct_text(text):
    list_correct = []
    for word in text.split():
        if not pspell.lookup(word):
            corrected_word = next(pspell.suggest(word), word)
            print(f"-> '%s' corrected to '%s'" % (word, corrected_word))
            list_correct.append(corrected_word)
        else:
            list_correct.append(word)
    text = ' '.join(list_correct)
    return text

def handle_connected_words(text):
    list_connected_words = []
    prefix = False
    for current_word, next_word in zip(text.split(), text.split()[1:]):
        if prefix:
            prefix = False
            continue
        elif (len(current_word) == 1 and current_word in ['כ', 'ב', 'ש', 'ל', 'מ']) or \
            (len(current_word) == 2 and current_word in ['בכ', 'כב']):
            prefix = True
            list_connected_words.append(current_word + next_word)
        else:
            prefix = False
            list_connected_words.append(current_word)
    if not prefix:
        list_connected_words.append(text.split()[-1])
    text = ' '.join(list_connected_words)

    return text

def normalize_text(text: str, filename: str):
    #if filename == 'common_voice_he_39897724':
    print(f"Before: {text}")

    text = text.lower()
    #’\'
    text = re.sub('[!?.,:;()"״]', '', text)
    text = text.replace('%', ' אחוזים ')
    text = re.sub('[-–־—]', ' ', text)

    numbers_in_text = re.findall(r'\d+', text)
    for number in numbers_in_text:
        text = text.replace(number, num2words(number, lang='he'))

    text = re.sub('[-–־—]', ' ', text)

    # Remove Hebrew Nikkud
    text = re.sub('[\u0591-\u05C7]+', '', text)

    #text = correct_text(text)
    
    text = handle_connected_words(text)
    text = get_base_forms(text)

    text = text.replace('[BLANK]', '')

    text = re.sub('[!?.,:;()"״’\']', '', text)
    text = re.sub('[-–־—]', ' ', text)

    text = text.replace('התה', 'הייתה')

    text = " ".join(text.split())

    print(f"After: {text}")

    return text

df_in = pd.read_csv('transcriptions.tsv', sep='\t')

statistics = []
normalized_text = []

statistics_total = AccuracyStatistics()

for _, row in df_in.iterrows():

    reference_text = normalize_text(row['reference_text'], row['filename']).split()
    transcribed_text = normalize_text(row['transcribed_text'], row['filename']).split()

    accuracy_statistics = AccuracyStatistics(reference_text, transcribed_text)
    statistics_total += accuracy_statistics
    statistics.append({'filename': row['filename'], **accuracy_statistics.to_dict()})
    normalized_text.append({'filename': row['filename'], 'reference_text': ' '.join(reference_text), 'transcribed_text': ' '.join(transcribed_text)})

for word_pair, num in statistics_total.frequent_errors(k=20):
    print('-> "%s" replaced by "%s" %d times.' %
    (word_pair[0], word_pair[1], num))

statistics_avg = {k: v / len(statistics) for k, v in statistics_total.to_dict().items()}

df_out = StatisticsDF(statistics)

df_out = df_out.sort_values(by=['wer'], ascending=False)

df_additional = StatisticsDF([{'filename': 'TOTAL', **statistics_total.to_dict()},
   {'filename': 'AVERAGE', **statistics_avg}])
df_out = df_out.concat(df_additional)

df_out.display()

df_out.save('statistics_normalized.csv')

df_normalized = pd.DataFrame(normalized_text)
df_normalized.to_csv('normalized_text.tsv', index=False, sep='\t')