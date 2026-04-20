import re


def clean_text(text):
    # remove tags from xml
    text = re.sub(r"<.*>", ' ', text)
    # remove indication of beginning of paragraph
    text = re.sub(r"^§ [\.\w]*\s*", ' ', text)
    # remove anything in [] 
    text = re.sub(r"\[.*\]", '', text)
    return text.lstrip().rstrip()
