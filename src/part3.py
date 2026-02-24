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
tokenizer_large_char_menaked = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-char-menaked')
model_large_char_menaked = AutoModel.from_pretrained('dicta-il/dictabert-large-char-menaked', trust_remote_code=True)
model_large_char_menaked.eval()

tokenizer_seg = AutoTokenizer.from_pretrained('dicta-il/dictabert-seg')
model_seg = AutoModel.from_pretrained('dicta-il/dictabert-seg', trust_remote_code=True)
model_seg.eval()

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

pre_normalization_corrections_force_equality = {
    "פניי": "הפנים שלי",
    "פנייך": "הפנים שלך"
}

pre_normalization_corrections = {
    "בביתו": "בבית שלו",
    "שבתוכם": "ש בתוכם",
    "הפעולה": "ה פעולה",
    "בין לאומיים": "בינלאומיים",
    "דוקטור": "'דר",
    "יהוה": "'ה",
    "ביילובסקי": "בריילובסקי",
    "דוגמא": "דוגמה",
    "גזירות": "גזרות"
}

post_normalization_corrections = {
    "יוסילי": "יוסי",
    "מאד": "מאוד",
    "אחוז": "אחוזים",
    "מרצ": "מרץ",
    "חנאלה": "חנה",
    "חנהלי": "חנה",
    "כשבארצות": "שבארצות",
    "ראיתיה": "ראיתי אותה",
    "אסמה": "אסם",
    "מענינים": "מעניינים",
    "מענינות": "מעניינות",
    "סושה": "סושי",
    "פסומים": "בשמים",
    "ספוטיפיי": "ספוטיפי",
    "סיטראון": "סיטרואן",
    "סיטואן": "סיטרואן",
    "קפסו": "קפצו",
    "ירסנית": "הרסנית",
    "מהי": "מה היא",
    "אזבסט": "אסבסט",
    "רגעיתון": "רגטון",
    "רגאטון": "רגטון",
    "מיקרוספוט": "מיקרוסופט",
    "מליארדים": "מיליארדים",
    "צמקוני": "צמחוני",
    "אלטרנטיבים": "אלטרנטיביים",
    "מונטריות": "מוניטריות",
    "פלישתים": "פלשתים",
    "סוריים": "סורים",
    "רסארים": "רסרים",
    "רבתים": "רבטים",
    "נורוגי": "נורבגי",
    "יספלו": "יסבלו",
    "תהא": "תהיה",
    "אינתיפדת": "אינתיפאדת",
    "לקסקלי": "לקסיקלי",
    "סודאן": "סודן",
    "מוסטפה": "מוסטפא",
    "צרניחובסקי": "טשנרחובסקי",
    "הולבורג": "אלבורג",
    "מליונים": "מיליונים",
    "צלאח": "סלאח",
    "אוטסאפ": "ווטסאפ",
    "גהנום": "גיהנום",
    "זקינה": "זקנה",
    "סרנבולות": "תרנגולות",
    "מששמעתי": "כאשר שמעתי",
    "כששמעתי": "כאשר שמעתי",
    "דריקה": "זריקה",
    "תטמיח": "תצמיח",
    "התהלק": "התהלך",
    "הריהו": "הרי הוא",
    "נשתתקו": "השתתקו",
    "בלביה": "בלגיה"
}

# Wrong post prefix segmentation corrections which we must fix manually
post_prefix_seg_corrections = {
    "ה כי": "הכי",
    "ב היל": "בהיל",
    "ל גרים": "לגרים",
    "ל הפתיד": "להפתיד",
    "ה חיש": "החיש",
    "ו אג": "ואג",
    "ול": "ו ל",
    "וה": "ו ה",
    "שה": "ש ה",
    "וב": "ו ב",
    "ומ": "ו מ",
    "כש": "כ ש",
    "משם": "מ שם",
    "מאז": "מ אז"
}

def handle_common_errors(text, error_dict, check_absolute_equality=False):
    # Separate single-word and multi-word errors
    errors_one_word = {error: correction for error, correction in error_dict.items() if ' ' not in error}
    errors_two_words = {error: correction for error, correction in error_dict.items() if ' ' in error}

    new_text = []

    skip = False

    for current_word, next_word in zip(text.split(), text.split()[1:]):
        old_word = current_word
        if skip:
            skip = False
            print(f"Replacing: {old_word} -> ''")
            continue
        
        for error, correction in errors_two_words.items():
            if (f"{current_word} {next_word}".endswith(correction) and not check_absolute_equality) \
            or (f"{current_word} {next_word}" == correction and check_absolute_equality):
                skip = True
                current_word = current_word + ' ' + next_word
                break
            elif (f"{current_word} {next_word}".endswith(error) and not check_absolute_equality) \
            or (f"{current_word} {next_word}" == error and check_absolute_equality):
                skip = True
                if len(correction.split()) == 1:
                    current_word = current_word.replace(error.split()[0], correction)
                else:
                    current_word = current_word.replace(error.split()[0], correction.split()[0])
                    next_word = next_word.replace(error.split()[1], correction.split()[1])
                    current_word = current_word + ' ' + next_word
                break
        
        # Check for single-word errors if no multi-word error was found
        if not skip:
            for error, correction in errors_one_word.items():
                if (current_word.endswith(error) and not check_absolute_equality) \
                or (current_word == error and check_absolute_equality):
                    current_word = current_word.replace(error, correction)
                    break
        if old_word != current_word:
            print(f"Replacing: {old_word} -> {current_word}")
        new_text.append(current_word)
    

    old_word = text.split()[-1]
    # Handle the last word
    if not skip:
        current_word = text.split()[-1]
        for error, correction in errors_one_word.items():
            if (current_word.endswith(error) and not check_absolute_equality) \
            or (current_word == error and check_absolute_equality):
                current_word = current_word.replace(error, correction)
                break
        if old_word != current_word:
            print(f"Replacing: {old_word} -> {current_word}")
        new_text.append(current_word)
    else:
        print(f"Replacing: {old_word} -> ''")
    return ' '.join(new_text)

#phunspell_storage_path = os.path.join(os.path.dirname(__file__), '..', 'phunspell-dict')
pspell = Phunspell('he_IL')

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
    text = handle_common_errors(text, numbers_m_to_f)
    return text

def normalize_spelling(text):
    # The predict method returns the text with Niqqud, 
    # but it ALSO standardizes the spelling to a consistent Male/Hasar form.
    # By default, it removes extra Matres Lectionis.
    result = model_large_char_menaked.predict([text], tokenizer_large_char_menaked)
    
    # The output has Niqqud. To compare for WER, we strip the Niqqud back off.
    vocalized_text = result[0]
    normalized_text = re.sub(r"[\u0591-\u05C7]", '', vocalized_text)
    
    return normalized_text

def normalize_spelling_seg(text):
    result = model_seg.predict([text], tokenizer_seg)
    return ' '.join([' '.join(tokens) for tokens in result[0]][1:-1])

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

def normalize_text(text: str, cnt: int, type_of_text: str):
    print(f"{str(cnt)}) {type_of_text} Before: {text}")

    text = text.lower()
 
    text = re.sub('[!?.,:;()"”“״]', '', text)
    text = text.replace('%', ' אחוזים ')
    text = re.sub('[-–־—]', ' ', text)

    # Remove Hebrew Nikkud
    text = re.sub('[\u0591-\u05C7]+', '', text)

    text = number_to_words(text)
    text = re.sub('[-–־—]', ' ', text)
    
    #print(f"{str(cnt)}) After number to words: {text}")

    text = handle_connected_words(text)

    #print(f"{str(cnt)}) After handle connected words: {text}")
   
    text = handle_common_errors(text, pre_normalization_corrections_force_equality, check_absolute_equality=True)
    text = handle_common_errors(text, pre_normalization_corrections)

    text = normalize_spelling(text)

    #print(f"{str(cnt)}) After normalize spelling: {text}")

    text = re.sub('[!?.,:;()"”“״’‘\']', '', text)
    text = re.sub('[-–־—]', ' ', text)
    
    # Add leading and trailing spaces to replace also words that are at the beginning or end of the text
    text = handle_common_errors(text, post_normalization_corrections)
    #print(f"{str(cnt)}) After common errors: {text}")

    text = normalize_spelling_seg(text)

    text = handle_common_errors(text, post_prefix_seg_corrections, check_absolute_equality=True)

    #print(f"{str(cnt)}) After normalize spelling seg: {text}")

    text = " ".join(text.split())

    #print(f"{str(cnt)}) {type_of_text} After: {text}")

    return text

df_in = pd.read_csv('transcriptions_new.tsv', sep='\t')

statistics = []
normalized_text = []

statistics_total = AccuracyStatistics()

for index, row in df_in.iterrows():
    reference_text = normalize_text(row['reference_text'], index+1, 'Reference').split()
    transcribed_text = normalize_text(row['transcribed_text'], index+1, 'Transcribed').split()

    accuracy_statistics = AccuracyStatistics(reference_text, transcribed_text)
    statistics_total += accuracy_statistics
    statistics.append({'filename': row['filename'], **accuracy_statistics.to_dict()})
    normalized_text.append({'filename': row['filename'], 'reference_text': ' '.join(reference_text), 'transcribed_text': ' '.join(transcribed_text)})

for word_pair, num in statistics_total.frequent_errors():
    print('-> "%s" replaced by "%s" %d times.' %
    (word_pair[0], word_pair[1], num))

statistics_avg = {k: v / len(statistics) for k, v in statistics_total.to_dict().items()}

df_out = StatisticsDF(statistics)

df_out = df_out.sort_values(by=['wer'], ascending=False)

df_out_files = df_out.df['filename'].tolist()

# Sort normalized_text to match the order of df_out_files
# Create a mapping from filename to index in df_out_files
filename_to_index = {filename: index for index, filename in enumerate(df_out_files)}
# Sort normalized_text based on the order in df_out_files
normalized_text = sorted(normalized_text, key=lambda x: filename_to_index.get(x['filename'], float('inf')))

df_additional = StatisticsDF([{'filename': 'TOTAL', **statistics_total.to_dict()},
   {'filename': 'AVERAGE', **statistics_avg}])
df_out = df_out.concat(df_additional)

df_out.display()

df_out.save('statistics_normalized_new.csv')

df_normalized = pd.DataFrame(normalized_text)
df_normalized.to_csv('normalized_text_new.tsv', index=False, sep='\t')