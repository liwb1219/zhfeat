# zhfeat


# Usage
```python
from extractor.feat_extractor import Extractor
import re


"""Input text, separated by "#" between paragraphs"""
text = """人有两件，双手和大脑，双手会做工，大脑会思考。#用手不用脑，事情做不好，用脑不用手，啥也办不到，用手又用脑，才能有创造。一切创造靠劳动，劳动要用手和脑。"""


""" (1) Only need character, word, sentence and paragraph features """
zh_extractor = Extractor(is_topic_features=False)
chinese_linguistic_features = zh_extractor.extract_chinese_linguistic_features(text)
print('chinese linguistic features:', chinese_linguistic_features)
print('features num:', len(chinese_linguistic_features))


""" (2) Also Need topic features """
zh_extractor = Extractor(is_topic_features=True)
chinese_linguistic_features = zh_extractor.extract_chinese_linguistic_features(text)
print('chinese linguistic features:', chinese_linguistic_features)
print('features num:', len(chinese_linguistic_features))


""" (3) Extract separately """
zh_extractor = Extractor(is_topic_features=True)
filtered_text = re.sub(r'\s+|\r|#', '', text)
paragraph_text = re.sub(r'\s+|\r', '', text)
char_features = zh_extractor.extract_character_features(filtered_text)
word_features = zh_extractor.extract_word_features(filtered_text)
sentence_features = zh_extractor.extract_sentence_features(filtered_text)
paragraph_features = zh_extractor.extract_paragraph_features(paragraph_text)
topic_features = zh_extractor.extract_difficulty_aware_topic_features(filtered_text)
print('char features:', char_features)
print('word features:', word_features)
print('sentence features:', sentence_features)
print('paragraph features:', paragraph_features)
print('topic features:', topic_features)
```
