# coding=utf-8
import json


# 字频字典{字:频率}
def load_chinese_character_frequency_dict(dir_path):
    return json.load(open(dir_path + '/table/ChineseCharacterFrequency.json', 'r', encoding='utf-8'))


# 笔画数字典{字:笔画数}
def load_chinese_character_strokes_dict(dir_path):
    return json.load(open(dir_path + '/table/ChineseCharacterStrokes.json', 'r', encoding='utf-8'))


# 现代汉语常用字
def load_common_chinese_character_table(dir_path):
    return set(json.load(open(dir_path + '/table/CommonChineseCharacters.json', 'r', encoding='utf-8')))


# 一级字
def load_first_level_character_table(dir_path):
    return set(json.load(open(dir_path + '/table/First-levelCharacters.json', 'r', encoding='utf-8')))


# 二级字
def load_second_level_character_table(dir_path):
    return set(json.load(open(dir_path + '/table/Second-levelCharacters.json', 'r', encoding='utf-8')))


# 三级字
def load_third_level_character_table(dir_path):
    return set(json.load(open(dir_path + '/table/Third-levelCharacters.json', 'r', encoding='utf-8')))


# 四级字
def load_fourth_level_character_table(dir_path):
    return set(json.load(open(dir_path + '/table/Fourth-levelCharacters.json', 'r', encoding='utf-8')))


# 成语
def load_idiom_table(dir_path):
    return set(json.load(open(dir_path + '/table/Idiom18744.json', 'r', encoding='utf-8')))


# 词频字典{词:频率}
def load_chinese_word_frequency_dict(dir_path):
    return json.load(open(dir_path + '/table/ChineseWordFrequency.json', 'r', encoding='utf-8'))


# 一级词
def load_first_level_word_table(dir_path):
    return set(json.load(open(dir_path + '/table/First-levelWords.json', 'r', encoding='utf-8')))


# 二级词
def load_second_level_word_table(dir_path):
    return set(json.load(open(dir_path + '/table/Second-levelWords.json', 'r', encoding='utf-8')))


# 三级词
def load_third_level_word_table(dir_path):
    return set(json.load(open(dir_path + '/table/Third-levelWords.json', 'r', encoding='utf-8')))


# 四级词
def load_fourth_level_word_table(dir_path):
    return set(json.load(open(dir_path + '/table/Fourth-levelWords.json', 'r', encoding='utf-8')))


def fill_depth(parents, node, depth):
    if depth[node] != 0:
        return
    if parents[node] == -1:
        depth[node] = 1
        return
    if depth[parents[node]] == 0:
        fill_depth(parents, parents[node], depth)
    depth[node] = depth[parents[node]] + 1


def load_stopwords_table(dir_path):
    return set(json.load(open(dir_path + '/table/ChineseStopWords.json', 'r', encoding='utf-8')))
