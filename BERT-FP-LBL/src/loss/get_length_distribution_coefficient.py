# coding=utf-8
import ast
import re


def get_weighted_length(len_list, w1=0.25, w2=0.25, w3=0.25, w4=0.25):
    if not len_list:
        return (0, 0, 0, 0), 0
    len_list.sort()
    avg_len = sum(len_list) / len(len_list)
    min_len = len_list[0]
    mid_len = len_list[int(len(len_list) * 0.5)]
    max_len = len_list[-1]
    weighted_length = w1 * avg_len + w2 * min_len + w3 * mid_len + w4 * max_len
    return (avg_len, min_len, mid_len, max_len), weighted_length


if __name__ == '__main__':
    for data_name in ['Cambridge', 'OneStopE', 'WeeBit']:
        dataset_path = '../../data/' + data_name + '/train.txt'
        print('data name:', data_name)
        length_list_0, length_list_1, length_list_2, length_list_3, length_list_4 = [], [], [], [], []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for data in f:
                data_dict = ast.literal_eval(data)
                label = int(data_dict['label']) - 1
                text = re.sub(r'\n', ' ', data_dict['text'])
                word_num = len(text.split())
                if label == 0:
                    length_list_0.append(word_num)
                elif label == 1:
                    length_list_1.append(word_num)
                elif label == 2:
                    length_list_2.append(word_num)
                elif label == 3:
                    length_list_3.append(word_num)
                elif label == 4:
                    length_list_4.append(word_num)
        _, weighted_length_0 = get_weighted_length(length_list_0)
        _, weighted_length_1 = get_weighted_length(length_list_1)
        _, weighted_length_2 = get_weighted_length(length_list_2)
        _, weighted_length_3 = get_weighted_length(length_list_3)
        _, weighted_length_4 = get_weighted_length(length_list_4)
        weighted_length_list = [weighted_length_0, weighted_length_1, weighted_length_2,
                                weighted_length_3, weighted_length_4]
        print('weighted_length_list:', weighted_length_list)
        length_coefficient = [round(i / sum(weighted_length_list), 4) for i in weighted_length_list]
        print('length_coefficient:', length_coefficient)
        print('coefficient sum:', sum(length_coefficient))
        print()
