from transformers import AutoTokenizer, AutoModel 
import torch

# Load Dicta's morphological model
tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-lex')
model = AutoModel.from_pretrained('dicta-il/dictabert-lex', trust_remote_code=True)
model.eval()

text1 = "אלה מבטאים גם הישגים בתחומי הספורט והאומנות"
text2 = "אלא מבטאים גם הישגים בתחומי הספורט והאמנות"

text3 = "פייסבוק או חברת מטא היא הבעלים של אינסטגרם ווואטסאפ"
text4 = "פיסבוק או חברת מטא היא הבעלים של אינסטגרם ואווטסאפ"

text5 = "ראשן החל להניקנו"
text6 = "עשן החל להניקנו"

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

print(get_base_forms(text1))
print(get_base_forms(text2))
print(get_base_forms(text3))
print(get_base_forms(text4))
print(get_base_forms(text5))
print(get_base_forms(text6))