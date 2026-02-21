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
    numbers_m = list(numbers_m_to_f.keys())
    number_m_one_word = [key for key in numbers_m_to_f.keys() if ' ' not in key]
    number_m_two_words = [key for key in numbers_m_to_f.keys() if ' ' in key]

    new_text = []

    skip = False

    for current_word, next_word in zip(text.split(), text.split()[1:]):
        if skip:
            skip = False
            continue
        
        for number in number_m_two_words:
            if f"{current_word} {next_word}".endswith(number):
                female_form = numbers_m_to_f[number]
                skip = True
                current_word = current_word.replace(number.split()[0], female_form.split()[0])
                next_word = next_word.replace(number.split()[1], female_form.split()[1])
                current_word = current_word + ' ' + next_word
                break
        if not skip:
            for number in number_m_one_word:
                if current_word.endswith(number):
                    current_word = current_word.replace(number, numbers_m_to_f[number])
                    break
        new_text.append(current_word)
    if not skip:
        current_word = text.split()[-1]
        for number in number_m_one_word:
            if current_word.endswith(number):
                current_word = current_word.replace(number, numbers_m_to_f[number])
                break
        new_text.append(current_word)
    return ' '.join(new_text)


text1 = "הצעדה תתחיל בשעה ארבעה ליד התחנה המרכזית"
text2 = "עשרים ותשעה באוקטובר אלפיים ואחד"

print(numbers_m_to_f_function(text1))
print(numbers_m_to_f_function(text2))