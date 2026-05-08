import re
import json

def clean_text(text):
    # remove tags from xml
    text = re.sub(r"<.*>", ' ', text)
    # remove indication of beginning of paragraph
    text = re.sub(r"^§ [\.\w]*\s*", ' ', text)
    # remove anything in [] 
    text = re.sub(r"\[.*\]", '', text)
    return text.lstrip().rstrip()

def to_json(dictionary, filename):
    with open(filename,'w') as fp:
        json.dump(dictionary, fp,sort_keys=True, indent=4,ensure_ascii=False)