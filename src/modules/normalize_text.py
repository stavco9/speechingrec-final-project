import re

from num2words import num2words
from transformers import AutoTokenizer, AutoModel
from consts.correction_dict import CorrectionDict
from phunspell import Phunspell

# This class is used to normalize the text of the transcriptions or original text among various normalization steps.
class NormalizeText:
    def __init__(self):

        # Load Dicta's morphological model for normalizing spelling (Ktiv Male to Ktiv Haser form)
        self.tokenizer_large_char_menaked = AutoTokenizer.from_pretrained('dicta-il/dictabert-large-char-menaked')
        self.model_large_char_menaked = AutoModel.from_pretrained('dicta-il/dictabert-large-char-menaked', trust_remote_code=True)
        self.model_large_char_menaked.eval()
       
        # Load Dicta's morphological model for normalizing spelling segmentation (Seperate word prefixes from the rest of the word)
        self.tokenizer_seg = AutoTokenizer.from_pretrained('dicta-il/dictabert-seg')
        self.model_seg = AutoModel.from_pretrained('dicta-il/dictabert-seg', trust_remote_code=True)
        self.model_seg.eval()

        # Load Phunspell for correcting spelling (OOV words)
        self.phunspell = Phunspell('he_IL')

        # List to store the corrections made to the text
        self.corrections = []

        # Correction dictionary for common errors
        self.correction_dict = CorrectionDict()
    
    # This function is used to handle common errors in the text by replacing the errors with the corrections.
    # It can also check for absolute equality between the error and the correction.
    def _handle_common_errors(self, text: str, error_dict: dict, check_absolute_equality: bool = False) -> str:
        
        # Separate single-word and multi-word errors into two dictionaries
        errors_one_word = {error: correction for error, correction in error_dict.items() if ' ' not in error}
        errors_two_words = {error: correction for error, correction in error_dict.items() if ' ' in error}

        new_text = []

        # Flag to skip the current word if it is an error and the correction is a multi-word error
        skip = False

        # Iterate over the words in the text
        for current_word, next_word in zip(text.split(), text.split()[1:]):
            
            # Save the original word for printing the replacement
            old_word = current_word

            # If the skip flag is True, skip the current word and print the replacement
            if skip:
                skip = False
                print(f"Replacing: {old_word} -> ''")
                continue
            
            # Iterate over the multi-word errors
            for error, correction in errors_two_words.items():

                # Check if the current word and the next word matches the correction
                if (f"{current_word} {next_word}".endswith(correction) and not check_absolute_equality) \
                or (f"{current_word} {next_word}" == correction and check_absolute_equality):

                    # Set the skip flag to True and replace the current word and the next word with the correction
                    skip = True
                    current_word = current_word + ' ' + next_word
                    break

                # Check if the current word and the next word matches the error
                elif (f"{current_word} {next_word}".endswith(error) and not check_absolute_equality) \
                or (f"{current_word} {next_word}" == error and check_absolute_equality):
                    
                    # Set the skip flag to True and replace the current word and the next word with the correction
                    skip = True
                    if len(correction.split()) == 1:
                        current_word = current_word.replace(error.split()[0], correction)
                    else:
                        current_word = current_word.replace(error.split()[0], correction.split()[0])
                        next_word = next_word.replace(error.split()[1], correction.split()[1])
                        current_word = current_word + ' ' + next_word
                    break
                    
            # Iterate over the single-word errors in case no multi-word error was found
            if not skip:
                for error, correction in errors_one_word.items():

                    # Check if the current word matches the error
                    if (current_word.endswith(error) and not check_absolute_equality) \
                    or (current_word == error and check_absolute_equality):
                        current_word = current_word.replace(error, correction)
                        break

            # If the original word is not the same as the current word, print the replacement
            if old_word != current_word:
                print(f"Replacing: {old_word} -> {current_word}")

            # Add the current word to the new text
            new_text.append(current_word)
        
        # Save the last word for printing the replacement
        old_word = text.split()[-1]

        # If the skip flag is False, handle the last word
        if not skip:
            current_word = text.split()[-1]
            for error, correction in errors_one_word.items():

                # Check if the current word matches the error
                if (current_word.endswith(error) and not check_absolute_equality) \
                or (current_word == error and check_absolute_equality):
                    current_word = current_word.replace(error, correction)
                    break

            # If the original word is not the same as the current word, print the replacement
            if old_word != current_word:
                print(f"Replacing: {old_word} -> {current_word}")
            new_text.append(current_word)
        else:
            # If the skip flag is True, print the replacement
            print(f"Replacing: {old_word} -> ''")

        # Return the new text as a single string
        return ' '.join(new_text)
   
    # This function is used to normalize the hours in the text by converting them to the 12-hour format.
    def _normalize_hours(self, text: str) -> str:

        # Iterate over the words in the text and check if they end with "שעה" and are followed by a number
        for i, word in enumerate(text.split()):
            if word.endswith("שעה"):
                if i+1 < len(text.split()) and text.split()[i+1].isnumeric():

                    # If the number represents an hour in the 24-hour format, convert it to the 12-hour format
                    original_number = int(text.split()[i+1])
                    new_number = original_number if original_number <= 12 else original_number - 12
                    text = text.replace(str(original_number), str(new_number))
        
        # Return the new text as a single string
        return text
    
    # This function is used to convert numbers to words and handle common errors in the text.
    def _number_to_words(self, text: str) -> str:

        # Normalize the hours in the text
        text = self._normalize_hours(text)

        # Find all the numbers in the text
        numbers_in_text = re.findall(r'\d+', text)

        # Iterate over the numbers in the text and convert them to words
        for number in numbers_in_text:
            text = text.replace(number, num2words(number, lang='he'))

        # Convert all masculine Hebrew numbers to feminine numbers
        text = self._handle_common_errors(text, self.correction_dict.numbers_m_to_f)
        return text

    # The predict method returns the text with Niqqud, 
    # but it ALSO standardizes the spelling to a consistent Male/Hasar form.
    # By default, it removes extra Matres Lectionis.
    # This function is used to normalize the spelling of the text by converting it to the Male/Hasar form.
    def _normalize_spelling(self, text: str) -> str:
        result = self.model_large_char_menaked.predict([text], self.tokenizer_large_char_menaked)
        
        # The output has Niqqud. To compare for WER, we strip the Niqqud back off.
        vocalized_text = result[0]
        normalized_text = self._remove_nikkud(vocalized_text)
        
        return normalized_text
    
    # This function is used to correct the spelling of the text by using Phunspell.
    # For OOV words that are not legal Hebrew words, it will use the Phunspell suggestions.
    def _correct_text(self, text: str, cnt: int) -> str:
        list_correct = []
        for word in text.split():
            # Check if word ends with any word in the correction dictionary of OOV words (Exceptions for words that are not legal Hebrew words)
            ends_with_oov = any(word.endswith(oov_word) for oov_word in self.correction_dict.hebrew_correct_oov_words)

            # If the word is not in the Phunspell dictionary and is not an exception for words that are not legal Hebrew words
            # use the Phunspell suggestions to correct the word
            if not self.phunspell.lookup(word) and not ends_with_oov:
                corrected_word = next(self.phunspell.suggest(word), word)
                print(f"{str(cnt)}) -> '%s' corrected to '%s'" % (word, corrected_word))
                self.corrections.append((cnt, word, corrected_word))
                list_correct.append(corrected_word)
            else:
                list_correct.append(word)
        text = ' '.join(list_correct)
        return text

    # This function is used to normalize the spelling segmentation of the text
    # by separating word prefixes from the rest of the word.
    def _normalize_spelling_seg(self, text: str) -> str:
        result = self.model_seg.predict([text], self.tokenizer_seg)
        return ' '.join([' '.join(tokens) for tokens in result[0]][1:-1])

    # This function is used to handle connected words in the text by connect prefix words with the next word.
    def _handle_connected_words(self, text: str) -> str:
        list_connected_words = []
        prefix = False

        # Iterate over the words in the text and check if they are connected words
        for current_word, next_word in zip(text.split(), text.split()[1:]):

            # If the prefix flag is True, skip the current word and set the prefix flag to False
            if prefix:
                prefix = False
                continue
            
            # If the current word is a prefix word, set the prefix flag to True and add the current word and the next word to the list of connected words
            elif (len(current_word) == 1 and current_word in ['כ', 'ב', 'ש', 'ל', 'מ']) or \
                (len(current_word) == 2 and current_word in ['בכ', 'כב']):
                prefix = True
                list_connected_words.append(current_word + next_word)
            
            # If the current word is not a prefix word, set the prefix flag to False and add the current word to the list of connected words
            else:
                prefix = False
                list_connected_words.append(current_word)

        # If the prefix flag is False, add the last word to the list of connected words
        if not prefix:
            list_connected_words.append(text.split()[-1])
        text = ' '.join(list_connected_words)

        return text
  
    # This function is used to remove Hebrew Nikkud from the text.
    def _remove_nikkud(self, text: str) -> str:
       return re.sub('[\u0591-\u05C7]+', '', text)

    # This function is used to normalize the text by applying all the normalization steps.
    def normalize_text(self, text: str, cnt: int, type_of_text: str) -> str:

        # Print the original text
        print(f"{str(cnt)}) {type_of_text} Before: {text}")

        # Remove punctuation and special characters and replace percentage with "אחוזים"
        text = re.sub('[!?.,:;()"”“״]', '', text)
        text = text.replace('%', ' אחוזים ')
        text = re.sub('[-–־—]', ' ', text)

        # Remove Hebrew Nikkud
        text = self._remove_nikkud(text)

        # Convert numbers to words
        text = self._number_to_words(text)

        # Replace dashes with spaces
        text = re.sub('[-–־—]', ' ', text)
        
        # Connect prefix words with the next word
        text = self._handle_connected_words(text)

        # Handle common errors in the text by replacing the errors with the corrections before normalizing the spelling.
        text = self._handle_common_errors(
            text, 
            self.correction_dict.pre_normalization_corrections_force_equality, 
            check_absolute_equality=True
        )
        text = self._handle_common_errors(
            text,
            self.correction_dict.pre_normalization_corrections
        )
        
        # Normalize the spelling of the text by converting it from Ktiv Male to Ktiv Hasar form.
        text = self._normalize_spelling(text)

        # Remove punctuation and special characters and replace dashes with spaces
        text = re.sub('[!?.,:;()"”“״’‘\']', '', text)
        text = re.sub('[-–־—]', ' ', text)
        
        # Handle common errors in the text by replacing the errors with the corrections after normalizing the spelling.
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
        
        # Correct OOV words that are not legal Hebrew words by using Phunspell.
        text = self._correct_text(text, cnt)

        # Normalize the spelling segmentation of the text by separating word prefixes from the rest of the word.
        text = self._normalize_spelling_seg(text)
        
        # Handle common errors in the text by replacing the errors with the corrections after normalizing the spelling segmentation.
        text = self._handle_common_errors(text, self.correction_dict.post_prefix_seg_corrections, check_absolute_equality=True)

        # Join the words in the text with spaces
        text = " ".join(text.split())

        # Print the normalized text
        print(f"{str(cnt)}) {type_of_text} After: {text}")

        return text