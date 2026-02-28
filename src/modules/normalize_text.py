import re

from num2words import num2words
from transformers import AutoTokenizer, AutoModel
from consts.correction_dict import CorrectionDict
from phunspell import Phunspell

class NormalizeText:
    def __init__(self):
        # Load Dicta's morphological model
        self.tokenizer_large_char_menaked = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-char-menaked')
        self.model_large_char_menaked = AutoModel.from_pretrained('dicta-il/dictabert-large-char-menaked', trust_remote_code=True)
        self.model_large_char_menaked.eval()

        self.tokenizer_seg = AutoTokenizer.from_pretrained('dicta-il/dictabert-seg')
        self.model_seg = AutoModel.from_pretrained('dicta-il/dictabert-seg', trust_remote_code=True)
        self.model_seg.eval()

        self.phunspell = Phunspell('he_IL')

        self.corrections = []

        self.correction_dict = CorrectionDict()
    
    def _handle_common_errors(self, text: str, error_dict: dict, check_absolute_equality: bool = False) -> str:
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
   
    def _normalize_hours(self, text: str) -> str:
        for i, word in enumerate(text.split()):
            if word.endswith("שעה"):
                if i+1 < len(text.split()) and text.split()[i+1].isnumeric():
                    original_number = int(text.split()[i+1])
                    new_number = original_number if original_number <= 12 else original_number - 12
                    text = text.replace(str(original_number), str(new_number))

        return text
    
    def _number_to_words(self, text: str) -> str:
        text = self._normalize_hours(text)
        numbers_in_text = re.findall(r'\d+', text)
        for number in numbers_in_text:
            text = text.replace(number, num2words(number, lang='he'))
        text = self._handle_common_errors(text, self.correction_dict.numbers_m_to_f)
        return text

    def _normalize_spelling(self, text: str) -> str:
        # The predict method returns the text with Niqqud, 
        # but it ALSO standardizes the spelling to a consistent Male/Hasar form.
        # By default, it removes extra Matres Lectionis.
        result = self.model_large_char_menaked.predict([text], self.tokenizer_large_char_menaked)
        
        # The output has Niqqud. To compare for WER, we strip the Niqqud back off.
        vocalized_text = result[0]
        normalized_text = self._remove_nikkud(vocalized_text)
        
        return normalized_text

    def _correct_text(self, text: str, cnt: int) -> str:
        list_correct = []
        for word in text.split():
            # Check if word ends with any word in hebrew_correct_oov_words
            ends_with_oov = any(word.endswith(oov_word) for oov_word in self.correction_dict.hebrew_correct_oov_words)
            if not self.phunspell.lookup(word) and not ends_with_oov:
                corrected_word = next(self.phunspell.suggest(word), word)
                print(f"{str(cnt)}) -> '%s' corrected to '%s'" % (word, corrected_word))
                self.corrections.append((cnt, word, corrected_word))
                list_correct.append(corrected_word)
            else:
                list_correct.append(word)
        text = ' '.join(list_correct)
        return text

    def _normalize_spelling_seg(self, text: str) -> str:
        result = self.model_seg.predict([text], self.tokenizer_seg)
        return ' '.join([' '.join(tokens) for tokens in result[0]][1:-1])

    def _handle_connected_words(self, text: str) -> str:
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
  
    def _remove_nikkud(self, text: str) -> str:
        return re.sub('[\u0591-\u05C7]+', '', text)

    def normalize_text(self, text: str, cnt: int, type_of_text: str) -> str:
        print(f"{str(cnt)}) {type_of_text} Before: {text}")

        text = text.lower()
    
        text = re.sub('[!?.,:;()"”“״]', '', text)
        text = text.replace('%', ' אחוזים ')
        text = re.sub('[-–־—]', ' ', text)

        # Remove Hebrew Nikkud
        text = self._remove_nikkud(text)

        text = self._number_to_words(text)
        text = re.sub('[-–־—]', ' ', text)
        
        text = self._handle_connected_words(text)

        text = self._handle_common_errors(
            text, 
            self.correction_dict.pre_normalization_corrections_force_equality, 
            check_absolute_equality=True
        )
        text = self._handle_common_errors(
            text,
            self.correction_dict.pre_normalization_corrections
        )

        text = self._normalize_spelling(text)

        text = re.sub('[!?.,:;()"”“״’‘\']', '', text)
        text = re.sub('[-–־—]', ' ', text)
        
        text = self._handle_common_errors(
            text,
            self.correction_dict.post_normalization_corrections_force_equality,
            check_absolute_equality=True
        )

        # Add leading and trailing spaces to replace also words that are at the beginning or end of the text
        text = self._handle_common_errors(
            text,
            self.correction_dict.post_normalization_corrections
        )
        
        text = self._correct_text(text, cnt)

        text = self._normalize_spelling_seg(text)

        text = self._handle_common_errors(text, self.correction_dict.post_prefix_seg_corrections, check_absolute_equality=True)

        text = " ".join(text.split())

        print(f"{str(cnt)}) {type_of_text} After: {text}")

        return text