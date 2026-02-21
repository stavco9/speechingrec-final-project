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
tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-char-menaked')
model = AutoModel.from_pretrained('dicta-il/dictabert-large-char-menaked', trust_remote_code=True)
model.eval()

numbers_m_to_f = {
    "אחד": "אחת",
    "שניים": "שתיים",
    "שלושה": "שלוש",
    "ארבעה": "ארבע",
    "חמישה": "חמש",
    "שישה": "שש",
    "שבעה": "שבע",
    "שמונה": "שמונה",
    "תשעה": "תשע",
    "עשרה": "עשר",
    "אחד עשר": "אחת עשרה",
    "שנים עשר": "שתים עשרה",
    "שלושה עשר": "שלוש עשרה",
    "ארבעה עשר": "ארבע עשרה",
    "חמישה עשר": "חמש עשרה",
    "שישה עשר": "שש עשרה",
    "שבעה עשר": "שבע עשרה",
    "שמונה עשר": "שמונה עשרה",
    "תשעה עשר": "תשע עשרה"
}

def numbers_m_to_f_function(text):
    number_m_one_word = [key for key in numbers_m_to_f.keys() if ' ' not in key]
    number_f_one_word = [value for value in numbers_m_to_f.values() if ' ' not in value]
    number_m_two_words = [key for key in numbers_m_to_f.keys() if ' ' in key]
    number_f_two_words = [value for value in numbers_m_to_f.values() if ' ' in value]

    new_text = []

    skip = False

    for current_word, next_word in zip(text.split(), text.split()[1:]):
        if skip:
            skip = False
            continue
        
        for m_number, f_number in zip(number_m_two_words, number_f_two_words):
            if f"{current_word} {next_word}".endswith(f_number):
                skip = True
                current_word = current_word + ' ' + next_word
                break
            elif f"{current_word} {next_word}".endswith(m_number):
                skip = True
                current_word = current_word.replace(m_number.split()[0], f_number.split()[0])
                next_word = next_word.replace(m_number.split()[1], f_number.split()[1])
                current_word = current_word + ' ' + next_word
                break
        if not skip:
            for m_number, f_number in zip(number_m_one_word, number_f_one_word):
                if current_word.endswith(f_number):
                    break
                elif current_word.endswith(m_number):
                    current_word = current_word.replace(m_number, f_number)
                    break
        new_text.append(current_word)
    if not skip:
        current_word = text.split()[-1]
        for m_number, f_number in zip(number_m_one_word, number_f_one_word):
            if current_word.endswith(f_number):
                break
            elif current_word.endswith(m_number):
                current_word = current_word.replace(m_number, f_number)
                break
        new_text.append(current_word)
    return ' '.join(new_text)

common_errors = [{
       "error": "התה",
       "correction": "הייתה"
    },{
        "error": "יוסילי",
        "correction": "יוסי"
    },{
        "error": "מאד",
        "correction": "מאוד"
    },{
        "error": "אחוז",
        "correction": "אחוזים"
    },{
        "error": "בין לאמיים",
        "correction": "בינלאמיים"
    },{
        "error": "הבין לאמיים",
        "correction": "הבינלאמיים"
    },{
        "error": "מרצ",
        "correction": "מרץ"
    }]

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

def normalize_hours(text):
    for i, word in enumerate(text.split()):
        if word.endswith("שעה"):
            if i+1 < len(text.split()) and text.split()[i+1].isnumeric():
                original_number = int(text.split()[i+1])
                new_number = original_number if original_number <= 12 else original_number - 12
                text = text.replace(str(original_number), str(new_number))

    return text

def number_to_words(text):
    text = normalize_hours(text)
    numbers_in_text = re.findall(r'\d+', text)
    for number in numbers_in_text:
        text = text.replace(number, num2words(number, lang='he'))
    text = numbers_m_to_f_function(text)
    return text

def normalize_spelling(text):
    # The predict method returns the text with Niqqud, 
    # but it ALSO standardizes the spelling to a consistent Male/Hasar form.
    # By default, it removes extra Matres Lectionis.
    result = model.predict([text], tokenizer)
    
    # The output has Niqqud. To compare for WER, we strip the Niqqud back off.
    import re
    vocalized_text = result[0]
    normalized_text = re.sub(r"[\u0591-\u05C7]", '', vocalized_text)
    
    return normalized_text

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

def normalize_text(text: str, cnt: int):
    print(f"{str(cnt)}) Before: {text}")

    text = text.lower()
    #’\'
    text = re.sub('[!?.,:;()"״]', '', text)
    text = text.replace('%', ' אחוזים ')
    text = re.sub('[-–־—]', ' ', text)

    # Remove Hebrew Nikkud
    text = re.sub('[\u0591-\u05C7]+', '', text)

    text = number_to_words(text)
    text = re.sub('[-–־—]', ' ', text)

    #text = correct_text(text)
    
    text = handle_connected_words(text)
    text = normalize_spelling(text)
    #text = correct_text(text)

    #text = text.replace('[BLANK]', '')

    text = re.sub('[!?.,:;()"״’\']', '', text)
    text = re.sub('[-–־—]', ' ', text)

    for error in common_errors:
        text = text.replace(f" {error['error']} ", f" {error['correction']} ")

    text = " ".join(text.split())

    print(f"{str(cnt)}) After: {text}")

    return text

df_in = pd.read_csv('transcriptions_new.tsv', sep='\t')

statistics = []
normalized_text = []

statistics_total = AccuracyStatistics()

for index, row in df_in.iterrows():

    reference_text = normalize_text(row['reference_text'], index+1).split()
    transcribed_text = normalize_text(row['transcribed_text'], index+1).split()

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

df_out.save('statistics_normalized_new.csv')

df_normalized = pd.DataFrame(normalized_text)
df_normalized.to_csv('normalized_text_new.tsv', index=False, sep='\t')