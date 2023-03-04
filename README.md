# zhfeat: A Toolkit for Extracting Chinese Linguistic Features


## Usage
If you need topic features, please click the link https://disk.pku.edu.cn:443/link/CC30242BAFC6A7E930D4CB8515B5263E download the pre-trained model and put it in the checkpoint folder.
```python
from extractor.feat_extractor import Extractor
import re

# Input text, separated by "#" between paragraphs
text = """人有两件，双手和大脑，双手会做工，大脑会思考。#用手不用脑，事情做不好，用脑不用手，啥也办不到，用手又用脑，才能有创造。"""

# (1) Only need character, word, sentence and paragraph features
zh_extractor = Extractor(is_topic_features=False)
chinese_linguistic_features = zh_extractor.extract_chinese_linguistic_features(text)
print('chinese linguistic features:', chinese_linguistic_features)
print('features num:', len(chinese_linguistic_features))

# (2) Also Need topic features
zh_extractor = Extractor(is_topic_features=True)
chinese_linguistic_features = zh_extractor.extract_chinese_linguistic_features(text)
print('chinese linguistic features:', chinese_linguistic_features)
print('features num:', len(chinese_linguistic_features))

# (3) Extract separately
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

## Feature Definition
| Idx | Dim | Feature description |
|---|---|---|
| 1   | 1   | Total number of characters |

## Citation
```
@inproceedings{li-etal-2022-unified,
    title = "A Unified Neural Network Model for Readability Assessment with Feature Projection and Length-Balanced Loss",
    author = "Li, Wenbiao  and
      Ziyang, Wang  and
      Wu, Yunfang",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.504",
    pages = "7446--7457"}
```
