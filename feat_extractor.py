# coding=utf-8
import math
from extractor.utils import *
import jieba
import jieba.posseg
import re
from ddparser import DDParser
from LAC import LAC
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# current path
dir_path = os.path.dirname(os.path.realpath(__file__))


class Extractor:
    def __init__(self, is_topic_features=False):
        self.is_topic_features = is_topic_features
        self.chinese_character_frequency_dict = load_chinese_character_frequency_dict(dir_path)
        self.chinese_character_strokes_dict = load_chinese_character_strokes_dict(dir_path)
        self.common_chinese_character_table = load_common_chinese_character_table(dir_path)
        self.first_level_character_table = load_first_level_character_table(dir_path)
        self.second_level_character_table = load_second_level_character_table(dir_path)
        self.third_level_character_table = load_third_level_character_table(dir_path)
        self.fourth_level_character_table = load_fourth_level_character_table(dir_path)
        self.idiom_table = load_idiom_table(dir_path)
        self.chinese_word_frequency_dict = load_chinese_word_frequency_dict(dir_path)
        self.first_level_word_table = load_first_level_word_table(dir_path)
        self.second_level_word_table = load_second_level_word_table(dir_path)
        self.third_level_word_table = load_third_level_word_table(dir_path)
        self.fourth_level_word_table = load_fourth_level_word_table(dir_path)
        self.ddp = DDParser()
        self.stopwords_table = load_stopwords_table(dir_path)
        if self.is_topic_features:
            self.vectorizer = CountVectorizer(
                decode_error="replace",
                vocabulary=pickle.load(open('extractor/checkpoint/vocab.pkl', "rb")))
            self.anchored_topic_model = joblib.load('extractor/checkpoint/anchored_topic_model.pkl')

    def extract_chinese_linguistic_features(self, text):
        filtered_text = re.sub(r'\s+|\r|#', '', text)
        paragraph_text = re.sub(r'\s+|\r', '', text)
        char_features = self.extract_character_features(filtered_text)
        word_features = self.extract_word_features(filtered_text)
        sentence_features = self.extract_sentence_features(filtered_text)
        paragraph_features = self.extract_paragraph_features(paragraph_text)
        linguistic_features = char_features + word_features + sentence_features + paragraph_features
        if self.is_topic_features:
            topic_features = self.extract_difficulty_aware_topic_features(filtered_text)
            linguistic_features += topic_features
        return linguistic_features

    def extract_character_features(self, text):
        features_list = list()

        # 总字数
        feature0 = len(text)
        if feature0 == 0:
            feature0 = 1
        features_list.append(feature0)

        # 字种数
        char_table = {}
        for char in text:
            if char in char_table:
                char_table[char] += 1
            else:
                char_table[char] = 1
        feature1 = len(char_table)
        features_list.append(feature1)

        # TTR
        feature2 = feature1 / feature0
        features_list.append(feature2)

        # 平均笔画数 加权平均笔画数 不同笔画字数量(25维) 不同笔画字比例(25维)
        total_char = 0  # 总字数,不包含标点
        total_stroke = 0  # 总笔画数
        weight_total_stroke = 0  # 加权总笔画数
        stroke_number_cnt = [0 for _ in range(25)]
        for char in text:
            if char in self.chinese_character_strokes_dict and char in self.chinese_character_frequency_dict:
                total_char += 1
                total_stroke += self.chinese_character_strokes_dict[char]
                weight_total_stroke += -math.log(self.chinese_character_frequency_dict[char] + 1e-8) * self.chinese_character_strokes_dict[char]
                if self.chinese_character_strokes_dict[char] > 25:
                    stroke_number_cnt[24] += 1
                else:
                    stroke_number_cnt[self.chinese_character_strokes_dict[char] - 1] += 1
        if total_char == 0:
            total_char = 1
        feature3 = total_stroke / total_char
        features_list.append(feature3)
        feature4 = weight_total_stroke / total_char
        features_list.append(feature4)
        stroke_number_ratio = [i / total_char for i in stroke_number_cnt]
        features_list += stroke_number_cnt
        features_list += stroke_number_ratio

        # 平均字频
        total_char_type = 0
        total_char_freq = 0
        weight_total_char_freq = 0
        for char, freq in char_table.items():
            if char in self.chinese_character_frequency_dict:
                total_char_type += 1
                total_char_freq += freq
                weight_total_char_freq += -math.log(self.chinese_character_frequency_dict[char] + 1e-8) * freq

        if total_char_type == 0:
            total_char_type = 1
        feature7 = total_char_freq / total_char_type
        feature8 = weight_total_char_freq / total_char_type
        features_list.append(feature7)
        features_list.append(feature8)

        # 单次字数量 单次字比例
        char_once_num = 0
        for char, freq in char_table.items():
            if freq == 1:
                char_once_num += 1
        feature9 = char_once_num
        feature10 = char_once_num / feature0
        features_list.append(feature9)
        features_list.append(feature10)

        # 常用字比例
        common_char_num = 0
        for char in text:
            if char in self.common_chinese_character_table:
                common_char_num += 1
        feature11 = common_char_num
        feature12 = common_char_num / feature0
        features_list.append(feature11)
        features_list.append(feature12)

        # 未登录字数量 未登录字比例 X级字数量 X级字比例
        total_char_num = 0
        one_level_char_num = 0
        two_level_char_num = 0
        three_level_char_num = 0
        four_level_char_num = 0
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                total_char_num += 1
                if char in self.first_level_character_table:
                    one_level_char_num += 1
                elif char in self.second_level_character_table:
                    two_level_char_num += 1
                elif char in self.third_level_character_table:
                    three_level_char_num += 1
                elif char in self.fourth_level_character_table:
                    four_level_char_num += 1
        feature13 = total_char_num - one_level_char_num - two_level_char_num - three_level_char_num - four_level_char_num
        if total_char_num == 0:
            total_char_num = 1
        feature14 = feature13 / total_char_num
        features_list.append(feature13)
        features_list.append(feature14)
        feature15 = one_level_char_num
        feature16 = one_level_char_num / total_char_num
        features_list.append(feature15)
        features_list.append(feature16)
        feature17 = two_level_char_num
        feature18 = two_level_char_num / total_char_num
        features_list.append(feature17)
        features_list.append(feature18)
        feature19 = three_level_char_num
        feature20 = three_level_char_num / total_char_num
        features_list.append(feature19)
        features_list.append(feature20)
        feature21 = four_level_char_num
        feature22 = four_level_char_num / total_char_num
        features_list.append(feature21)
        features_list.append(feature22)

        feature23 = feature16 * 1 + feature18 * 2 + feature20 * 3 + feature22 * 4
        features_list.append(feature23)

        return features_list

    def extract_word_features(self, text):
        features_list = list()
        seg_text = jieba.lcut(text, cut_all=False)

        # 总词数
        feature0 = len(seg_text)
        if feature0 == 0:
            feature0 = 1
        features_list.append(feature0)

        # 词种数
        word_table = {}
        for word in seg_text:
            if word in word_table:
                word_table[word] += 1
            else:
                word_table[word] = 1
        feature1 = len(word_table)
        features_list.append(feature1)

        # TTR
        feature2 = feature1 / feature0
        features_list.append(feature2)

        # 平均词语长度 加权平均词语长度
        total_word = 0
        total_word_len = 0
        weight_total_word_len = 0
        for word in seg_text:
            total_word_len += len(word)
            if word in self.chinese_word_frequency_dict:
                total_word += 1
                weight_total_word_len += -math.log(self.chinese_word_frequency_dict[word] + 1e-8) * len(word)
        if total_word == 0:
            total_word = 1
        feature3 = total_word_len / feature0
        feature4 = weight_total_word_len / total_word
        features_list.append(feature3)
        features_list.append(feature4)

        # 平均词频 加权平均词频
        total_word_type = 0
        total_word_freq = 0
        weight_total_word_freq = 0
        for word, freq in word_table.items():
            if word in self.chinese_word_frequency_dict:
                total_word_type += 1
                total_word_freq += freq
                weight_total_word_freq += -math.log(self.chinese_word_frequency_dict[word] + 1e-8) * freq
        if total_word_type == 0:
            total_word_type = 1
        feature5 = total_word_freq / total_word_type
        feature6 = weight_total_word_freq / total_word_type
        features_list.append(feature5)
        features_list.append(feature6)

        # X字词数量 X字词比例
        word_len1_num = 0
        word_len2_num = 0
        word_len3_num = 0
        word_len4_num = 0
        word_len5_num = 0
        idiom_num = 0
        for word in seg_text:
            if word in self.idiom_table:
                idiom_num += 1
            if len(word) == 1:
                word_len1_num += 1
            elif len(word) == 2:
                word_len2_num += 1
            elif len(word) == 3:
                word_len3_num += 1
            elif len(word) == 4:
                word_len4_num += 1
            elif len(word) >= 5:
                word_len5_num += 1
        feature7 = word_len1_num
        feature8 = word_len1_num / feature0
        features_list.append(feature7)
        features_list.append(feature8)
        feature9 = word_len2_num
        feature10 = word_len2_num / feature0
        features_list.append(feature9)
        features_list.append(feature10)
        feature11 = word_len3_num
        feature12 = word_len3_num / feature0
        features_list.append(feature11)
        features_list.append(feature12)
        feature13 = word_len4_num
        feature14 = word_len4_num / feature0
        features_list.append(feature13)
        features_list.append(feature14)
        feature15 = word_len5_num
        feature16 = word_len5_num / feature0
        features_list.append(feature15)
        features_list.append(feature16)
        feature17 = idiom_num
        features_list.append(feature17)

        # 单次词数量 单次词比例
        word_once_num = 0
        for word, freq in word_table.items():
            if freq == 1:
                word_once_num += 1
        feature18 = word_once_num
        feature19 = word_once_num / feature0
        features_list.append(feature18)
        features_list.append(feature19)

        # 未登录词数量 未登录词比例 X级词数量 X级词比例
        one_level_word_num = 0
        two_level_word_num = 0
        three_level_word_num = 0
        four_level_word_num = 0
        for word in seg_text:
            if word in self.first_level_word_table:
                one_level_word_num += 1
            elif word in self.second_level_word_table:
                two_level_word_num += 1
            elif word in self.third_level_word_table:
                three_level_word_num += 1
            elif word in self.fourth_level_word_table:
                four_level_word_num += 1

        feature20 = feature0 - one_level_word_num - two_level_word_num - three_level_word_num - four_level_word_num
        feature21 = feature20 / feature0
        features_list.append(feature20)
        features_list.append(feature21)

        feature22 = one_level_word_num
        feature23 = one_level_word_num / feature0
        features_list.append(feature22)
        features_list.append(feature23)
        feature24 = two_level_word_num
        feature25 = two_level_word_num / feature0
        features_list.append(feature24)
        features_list.append(feature25)
        feature26 = three_level_word_num
        feature27 = three_level_word_num / feature0
        features_list.append(feature26)
        features_list.append(feature27)
        feature28 = four_level_word_num
        feature29 = four_level_word_num / feature0
        features_list.append(feature28)
        features_list.append(feature29)

        feature30 = feature23 * 1 + feature25 * 2 + feature27 * 3 + feature29 * 4
        features_list.append(feature30)

        pos_tag = jieba.posseg.lcut(text)
        pos2id = {'vi': 0, 'uz': 1, 'ag': 2, 'a': 3, 'ug': 4, 'rg': 5, 'i': 6, 'yg': 7, 'nz': 8, 'o': 9, 'ns': 10,
                  's': 11, 'r': 12, 'dg': 13, 'rz': 14, 'ud': 15, 'nt': 16, 't': 17, 'd': 18, 'nrt': 19, 'mq': 20,
                  'eng': 21, 'j': 22, 'g': 23, 'h': 24, 'mg': 25, 'uv': 26, 'b': 27, 'ad': 28, 'ul': 29, 'u': 30,
                  'an': 31, 'df': 32, 'vg': 33, 'rr': 34, 'nr': 35, 'm': 36, 'uj': 37, 'n': 38, 'l': 39, 'y': 40,
                  'p': 41, 'vd': 42, 'zg': 43, 'c': 44, 'nrfg': 45, 'tg': 46, 'z': 47, 'vq': 48, 'e': 49, 'f': 50,
                  'x': 51, 'vn': 52, 'v': 53, 'q': 54, 'ng': 55, 'k': 56}
        pos_distribute_cnt = [0 for _ in range(57)]
        for word in pos_tag:
            pos_distribute_cnt[pos2id[word.flag]] += 1
        pos_distribute_ratio = [i / len(pos_tag) for i in pos_distribute_cnt]
        features_list += pos_distribute_cnt
        features_list += pos_distribute_ratio

        return features_list

    def extract_sentence_features(self, text):
        features_list = list()
        sentences = re.split('(。|？|！|……)', text)
        sentences = [sentences[2 * i] + sentences[2 * i + 1] for i in range(int(len(sentences) / 2))]

        # 句子数
        feature0 = len(sentences)
        if feature0 == 0:
            feature0 = 1
        features_list.append(feature0)

        # 句子平均汉字数 句子平均词语数 句子最大汉字数 句子最大词语数 句长分布(30维)
        ddp = self.ddp
        sen_char_total = 0
        sen_word_total = 0
        sen_char_max = 0
        sen_word_max = 0
        len_cnt = [0 for _ in range(30)]
        # 平均句法树高度 最大句法树高度 句法树高度<=5/<=10/<=15/>=16的比例 依存关系分布(14维)
        syntax_tree_height_total = 0
        syntax_tree_height_max = 0
        syntax_tree_height_1 = 0
        syntax_tree_height_2 = 0
        syntax_tree_height_3 = 0
        syntax_tree_height_4 = 0
        dependency_cnt = [0 for _ in range(14)]
        for sen in sentences:
            parse_dict = ddp.parse(sen)[0]
            sen_char_total += len(sen)
            sen_char_max = max(sen_char_max, len(sen))
            sen_word_total += len(parse_dict['word'])
            sen_word_max = max(sen_word_max, len(parse_dict['word']))
            if len(sen) <= 100:
                len_cnt[int((len(sen) - 1) / 5)] += 1
            elif 101 <= len(sen) <= 190:
                len_cnt[int((len(sen) - 1) / 10) + 10] += 1
            elif len(sen) >= 191:
                len_cnt[29] += 1

            parents = [-1] + parse_dict['head']
            depth = [0 for _ in range(len(parents))]
            for i in range(len(parents)):
                fill_depth(parents, i, depth)

            max_depth = max(depth)
            syntax_tree_height_total += max_depth
            syntax_tree_height_max = max(syntax_tree_height_max, max_depth)
            if max_depth <= 5:
                syntax_tree_height_1 += 1
            elif 6 <= max_depth <= 10:
                syntax_tree_height_2 += 1
            elif 11 <= max_depth <= 15:
                syntax_tree_height_3 += 1
            elif max_depth >= 15:
                syntax_tree_height_4 += 1

            dependency_dict = {'SBV': 0, 'VOB': 1, 'POB': 2, 'ADV': 3, 'CMP': 4, 'ATT': 5, 'F': 6, 'COO': 7, 'DBL': 8,
                               'DOB': 9, 'VV': 10, 'IC': 11, 'MT': 12, 'HED': 13}
            for dependency in parse_dict['deprel']:
                dependency_cnt[dependency_dict[dependency]] += 1

        feature1 = sen_char_total / feature0
        feature2 = sen_word_total / feature0
        feature3 = sen_char_max
        feature4 = sen_word_max
        features_list.append(feature1)
        features_list.append(feature2)
        features_list.append(feature3)
        features_list.append(feature4)
        features_list += len_cnt

        feature11 = syntax_tree_height_total / feature0
        feature12 = syntax_tree_height_max
        feature13 = syntax_tree_height_1 / feature0
        feature14 = syntax_tree_height_2 / feature0
        feature15 = syntax_tree_height_3 / feature0
        feature16 = syntax_tree_height_4 / feature0
        features_list.append(feature11)
        features_list.append(feature12)
        features_list.append(feature13)
        features_list.append(feature14)
        features_list.append(feature15)
        features_list.append(feature16)
        features_list += dependency_cnt

        # 分句数 分句平均汉字数 分句平均词语数 分句最大汉字数 分句最大词语数
        clauses = re.split('(。|？|！|，|；|……)', text)
        clauses = [clauses[2 * i] + clauses[2 * i + 1] for i in range(int(len(clauses) / 2))]
        cla_char_total = 0
        cla_word_total = 0
        cla_char_max = 0
        cla_word_max = 0
        lac = LAC(mode='seg')
        for cla in clauses:
            cla_char_total += len(cla)
            cla_char_max = max(cla_char_max, len(cla))
            seg_cla = lac.run(cla)
            cla_word_total += len(seg_cla)
            cla_word_max = max(cla_word_max, len(seg_cla))
        feature5 = len(clauses)
        if feature5 == 0:
            feature5 = 1
        features_list.append(feature5)
        feature6 = cla_char_total / feature5
        feature7 = cla_word_total / feature5
        feature8 = cla_char_max
        feature9 = cla_word_max
        features_list.append(feature6)
        features_list.append(feature7)
        features_list.append(feature8)
        features_list.append(feature9)

        return features_list

    @staticmethod
    def extract_paragraph_features(text):
        features_list = list()
        paras = re.split('#', text)

        # 总段数
        feature0 = len(paras)
        if feature0 == 0:
            feature0 = 1
        features_list.append(feature0)

        # 段落平均汉字数 段落平均词语数 段落最大汉字数 段落最大词语数
        para_char_total = 0
        para_word_total = 0
        para_char_max = 0
        para_word_max = 0
        for para in paras:
            para_char_total += len(para)
            para_char_max = max(para_char_max, len(para))
            seg_para = jieba.lcut(para, cut_all=False)
            para_word_total += len(seg_para)
            para_word_max = max(para_word_max, len(seg_para))

        feature1 = para_char_total / feature0
        feature2 = para_word_total / feature0
        feature3 = para_char_max
        feature4 = para_word_max
        features_list.append(feature1)
        features_list.append(feature2)
        features_list.append(feature3)
        features_list.append(feature4)

        return features_list

    def extract_difficulty_aware_topic_features(self, text):
        seg_text = jieba.lcut(text, cut_all=False)
        # 只保留词语长度大于1的
        topic_words = [word for word in seg_text if len(word) > 1]
        # 去停用词
        topic_words = [word for word in topic_words if word not in self.stopwords_table]
        model_inputs = [' '.join(topic_words)]
        tf = self.vectorizer.transform(model_inputs)
        p_y_given_x, log_z = self.anchored_topic_model.transform(tf, details=True)
        topic_distribution = p_y_given_x.tolist()
        difficulty_aware_topic_features = topic_distribution[0]
        return difficulty_aware_topic_features
