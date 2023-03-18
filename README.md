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
| 序号 | 类型 | 特征 | 维度 | 特征描述 |
|-|-|-|-|-|
| 1 | 字 | 总字数 | 1 | 文章中的汉字总数 |
| 2 | 字 | 字种数 | 1 | 文章中不同汉字的总数 |
| 3 | 字 | TTR | 1 | 字总数与总字数之比 |
| 4 | 字 | 平均笔画数 | 1 | 每个汉字的平均笔画数 |
| 5 | 字 | 加权平均笔画数 | 1 | 对特征4进行负对数字频加权 |
| 6 | 字 | 不同笔画字数量 | 25 | 不同笔画的字出现次数 |
| 7 | 字 | 不同笔画字比例 | 25 | 不同笔画的字出现频率 |
| 8 | 字 | 平均字频 | 1 | 平均每个字出现的次数 |
| 9 | 字 | 加权平均字频 | 1 | 对特征8进行负对数字频加权 |
| 10 | 字 | 单次字数量 | 1 | 文章中只出现一次的字的总数 |
| 11 | 字 | 单次字比例 | 1 | 文章中只出现一次的字的比例 |
| 12 | 字 | 常用字数量 | 1 | 文章中常用字的总数 |
| 13 | 字 | 常用字比例 | 1 | 文章中常用字的比例 |
| 14 | 字 | 未登录字数量 | 1 | 未出现在字表中的字的总数 |
| 15 | 字 | 未登录字比例 | 1 | 未出现在字表中的字的比例 |
| 16 | 字 | 一级字数量 | 1 | 难度为一级的字的总数 |
| 17 | 字 | 一级字比例 | 1 | 难度为一级的字的比例 |
| 18 | 字 | 二级字数量 | 1 | 难度为二级的字的总数 |
| 19 | 字 | 二级字比例 | 1 | 难度为二级的字的比例 |
| 20 | 字 | 三级字数量 | 1 | 难度为三级的字的总数 |
| 21 | 字 | 三级字比列 | 1 | 难度为三级的字的比例 |
| 22 | 字 | 四级字数量 | 1 | 难度为四级的字的总数 |
| 23 | 字 | 四级字比例 | 1 | 难度为四级的字的比例 |
| 24 | 字 | 平均汉字等级 | 1 | 按难度等级对不同等级字加权 |
| 25 | 词 | 总词数 | 1 | 文章中的词语总数 |
| 26 | 词 | 词种数 | 1 | 文章中不同词语的总数 |
| 27 | 词 | TTR | 1 | 词种数与总词数之比 |
| 28 | 词 | 平均词语长度 | 1 | 平均每个词语所含汉字的个数 |
| 29 | 词 | 加权平均词语长度 | 1 | 对特征28进行负对数词频加权 |
| 30 | 词 | 平均词频 | 1 | 平均每个词语出现的次数 |
| 31 | 词 | 加权平均词频 | 1 | 对特征30进行负对数词频加权 |
| 32 | 词 | 单字词数量 | 1 | 一个字组成的词语的总数 |
| 33 | 词 | 单字词比例 | 1 | 一个字组成的词语的比例 |
| 34 | 词 | 二字词数量 | 1 | 两个字组成的词语的总数 |
| 35 | 词 | 二字词比例 | 1 | 两个字组成的词语的比例 |
| 36 | 词 | 三字词数量 | 1 | 三个字组成的词语的总数 |
| 37 | 词 | 三字词比例 | 1 | 三个字组成的词语的比例 |
| 38 | 词 | 四字词数量 | 1 | 四一个字组成的词语的总数 |
| 39 | 词 | 四字词比例 | 1 | 四一个字组成的词语的比例 |
| 40 | 词 | 多字词数量 | 1 | 大于四个字组成的词语的总数 |
| 41 | 词 | 多字词比例 | 1 | 大于四个字组成的词语的比例 |
| 42 | 词 | 成语数量 | 1 | 文章中的词语总数 |
| 43 | 词 | 单次词数量 | 1 | 文章中只出现一次的词的总数 |
| 44 | 词 | 单次词比例 | 1 | 文章中只出现一次的词的总数 |
| 45 | 词 | 未登录词数量 | 1 | 未出现在词表中的词的总数 |
| 46 | 词 | 未登录词比例 | 1 | 未出现在词表中的词的比例 |
| 47 | 词 | 甲（一）级词数量 | 1 | 难度为甲的词的总数 |
| 48 | 词 | 甲（一）级词比例 | 1 | 难度为甲的词的比例 |
| 49 | 词 | 乙（二）级词数量 | 1 | 难度为乙的词的总数 |
| 50 | 词 | 乙（二）级词比例 | 1 | 难度为乙的词的比例 |
| 51 | 词 | 丙（三）级词数量 | 1 | 难度为丙的词的总数 |
| 52 | 词 | 丙（三）级次比例 | 1 | 难度为丙的词的比例 |
| 53 | 词 | 丁（四）级词数量 | 1 | 难度为丁的词的总数 |
| 54 | 词 | 丁（四）级次比例 | 1 | 难度为丁的词的比例 |
| 55 | 词 | 平均词语等级 | 1 | 按难度等级对不同等级词加权 |
| 56 | 词 | 词性分布数量 | 57 | 不同词性的词的总数 |
| 57 | 词 | 词性分布比例 | 57 | 不同词性的词的比例 |
| 58 | 句 | 句子数 | 1 | 文章中的句子总数 |
| 59 | 句 | 句子平均汉字数 | 1 | 平均每个句子所含的字数 |
| 60 | 句 | 句子平均词语数 | 1 | 平均每个句子所含的词数 |
| 61 | 句 | 句子最大汉字数 | 1 | 句子所含字数的最大值 |
| 62 | 句 | 句子最大词语数 | 1 | 句子所含词数的最大值 |
| 63 | 句 | 分句数 | 1 | 文章中的分句总数 |
| 64 | 句 | 分句平均汉字数 | 1 | 平均每个分句所含的字数 |
| 65 | 句 | 分句平均词语数 | 1 | 平均每个分句所含的词数 |
| 66 | 句 | 分句最大汉字数 | 1 | 分句所含字数的最大值 |
| 67 | 句 | 分句最大词语数 | 1 | 分句所含词数的最大值 |
| 68 | 句 | 句长分布 | 30 | 长度为特定区间的句子数 |
| 69 | 句 | 平均句法树高度 | 1 | 平均每个句子的句法树高度 |
| 70 | 句 | 最大句法树高度 | 1 | 句法树高度中的最大值 |
| 71 | 句 | 句法树高度<=5的比例 | 1 | 句法树高度<=5的句子比例 |
| 72 | 句 | 句法树高度<=10的比例 | 1 | 5<句法树高度<=10的句子比例 |
| 73 | 句 | 句法树高度<=15的比例 | 1 | 10<句法树高度<=15的句子比例 |
| 74 | 句 | 句法树高度>=16的比例 | 1 | 句法树高度>=16的句子比例 |
| 75 | 句 | 依存关系分布 | 14 | 每种依存关系的占比 |
| 76 | 段 | 总段数 | 1 | 文章中的段落总数 |
| 77 | 段 | 段落平均汉字数 | 1 | 平均每个段落所含的字数 |
| 78 | 段 | 段落平均词语数 | 1 | 平均每个段落所含的词数 |
| 79 | 段 | 段落最大汉字数 | 1 | 段落所含字数的最大值 |
| 80 | 段 | 段落最大词语数 | 1 | 段落所含词数的最大值 |
| 81 | 主题 | 难度感知的主题特征 | 160 | 其中一半主题锚定难度词汇 |

## Retrain the corex model yourself
cd BERT-FP-LBL/features-main
python run_corex.py

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
