from transformers import AutoTokenizer, AutoModel 
import torch
from phunspell import Phunspell

# Load Dicta's morphological model
tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-char-menaked')
model = AutoModel.from_pretrained('dicta-il/dictabert-large-char-menaked', trust_remote_code=True)
model.eval()

tokenizer_seg = AutoTokenizer.from_pretrained('dicta-il/dictabert-seg')
model_seg = AutoModel.from_pretrained('dicta-il/dictabert-seg', trust_remote_code=True)

model_seg.eval()

pspell = Phunspell('he_IL')

texts = [
    "אלה מבטאים גם הישגים בתחומי הספורט והאומנות",
    "אלא מבטאים גם הישגים בתחומי הספורט והאמנות",
    "פייסבוק או חברת מטא היא הבעלים של אינסטגרם ווואטסאפ",
    "פיסבוק או חברת מטא היא הבעלים של אינסטגרם ואווטסאפ",
    "ראשן החל להניקנו",
    "עשן החל להניקנו",
    "לעיסת החמורים נשרתה מנוחה באסמה",
    "לעסת החמורים השרתה מנוחה באסם"
]

def correct_text(text):
    list_correct = []
    for word in text.split():
        if not pspell.lookup(word):
            corrected_word = next(pspell.suggest(word), word)
            #print(f"-> '%s' corrected to '%s'" % (word, corrected_word))
            list_correct.append(corrected_word)
        else:
            list_correct.append(word)
    text = ' '.join(list_correct)
    return text

def get_lemmas(text):
    # This returns a dictionary with various morphological data
    output = model.predict([text], tokenizer)
    
    # Extract the 'lemma' for each token
    lemmas = [token['lex'] for sentence in output for token in sentence['tokens']]
    return " ".join(lemmas)

def normalize_with_dicta(text):
    # Dicta models use a .predict method for their joint tasks
    with torch.no_grad():
        # output_style='json' gives us the 'lex' (lemma) field
        results = model.predict([text], tokenizer, output_style='json')
    
    lemmas = []
    for sentence in results:
        for token in sentence['tokens']:
            # 'lex' is the canonical dictionary form (the Lemma)
            lemmas.append(token['lex'])
            
    return " ".join(lemmas)

def get_base_forms(text):
    # This returns the 'lexemes' (the standardized base form)
    predictions = model.predict([text], tokenizer)
    # Extract the 'lex' field for each word
    return " ".join([token[1] for token in predictions[0]])

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

def normalize_spelling_seg(text):
    result = model_seg.predict([text], tokenizer_seg)
    return ' '.join([' '.join(tokens) for tokens in result[0]][1:-1])

for text in texts:
    #text = correct_text(text)
    text = normalize_spelling(text)
    text = normalize_spelling_seg(text)
    print(text)