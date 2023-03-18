# coding=utf-8
import ast
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from corextopic import corextopic as ct
import joblib
import os


def read_stopwords(file_path):
    stop_words = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stop_words = ast.literal_eval(line)
    return stop_words


def read_pretrain_wikipedia_data(file_path, stop_words):
    nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for data in tqdm(f.readlines()):
            seg_text = [token.text for token in nlp(data.strip())]
            # 只保留词语长度大于1的
            seg_text = [word for word in seg_text if len(word) > 1]
            # 去停用词
            seg_text = [word for word in seg_text if word not in stop_words]
            data_list.append(seg_text)
    return data_list


if __name__ == '__main__':
    stopwords_set = read_stopwords(file_path='stopwords/EnglishStopWords.txt')
    pretrain_data = read_pretrain_wikipedia_data(file_path='pretrain_data/corpus_en.txt', stop_words=stopwords_set)
    corpus_en = [' '.join(d) for d in pretrain_data]

    # return document-term matrix(term frequency)   replace必须加 用于保存训练集的特征   max_features词表大小
    # max_df=0.5, min_df=20, max_features=127238
    vectorizer = CountVectorizer(max_features=None, max_df=0.5, min_df=20, binary=True, decode_error='replace')
    tf = vectorizer.fit_transform(corpus_en)  # Learn the vocabulary dictionary and return document-term matrix(train)
    vocab = vectorizer.get_feature_names()

    if not os.path.exists('pretrain_model'):
        os.mkdir('pretrain_model')
    with open('pretrain_model/vocab.pkl', 'wb') as fw:
        pickle.dump(vectorizer.vocabulary_, fw)

    """
    The anchor strength controls how much weight CorEx puts towards 
    maximizing the mutual information between the anchor words and their respective topics. 

    Anchor strength should always be set at a value greater than 1, 
    since setting anchor strength between 0 and 1 only recovers the unsupervised CorEx objective.

    Empirically, setting anchor strength from 1.5 - 3 seems to nudge the topic model towards the anchor words. 

    Setting anchor strength greater than 5 is strongly enforcing that the CorEx topic 
    model find a topic associated with the anchor words.

    We encourage users to experiment with the anchor strength and determine what values are best for their needs.
    """
    anchor_words = [
        ['people', 'they'],
        ['day', 'days', 'months', 'time', 'week'],
        ['brother', 'son'],
        ['lead', 'led'],
        ['met', 'told'],
        ['as', 'but', 'it', 'one', 'that', 'there', 'this', 'to', 'when', 'with'],
        ['late', 'later', 'lost', 'named', 'ten'],
        ['change', 'close', 'hand', 'move', 'takes'],
        ['book', 'story', 'written'],
        ['win', 'won'],
        ['level', 'low'],
        ['life', 'love'],
        ['mother', 'wife'],
        ['he', 'she'],
        ['set', 'start'],
        ['all', 'some', 'these'],

        ['held', 'moved'],
        ['force', 'forces', 'military'],
        ['final', 'fourth', 'victory'],
        ['addition', 'based', 'included', 'including', 'leading', 'major'],
        ['career', 'success', 'successful'],
        ['release', 'released'],
        ['allowed', 'half', 'leaving', 'position', 'rest', 'while'],
        ['record', 'recorded'],
        ['brought', 'called', 'decided', 'finally', 'forced', 'reached', 'started'],
        ['law', 'public'],
        ['after', 'appeared', 'during', 'earlier', 'initially', 'stated', 'wrote'],
        ['building', 'built'],
        ['relationship', 'role'],
        ['death', 'father'],
        ['control', 'system'],
        ['modern', 'style'],
        ['play', 'playing'],
        ['received', 'return', 'returned'],
        ['created', 'form', 'process', 'result'],
        ['attack', 'battle'],
        ['died', 'killed'],

        ['appearance', 'influence', 'presence'],
        ['discovered', 'evidence', 'reported', 'revealed'],
        ['contract', 'prior'],
        ['designed', 'features'],
        ['continued', 'despite', 'remained'],
        ['operation', 'operations'],
        ['according', 'referred', 'stating', 'supported'],
        ['considered', 'declared', 'established', 'intended', 'involved', 'majority'],
        ['debut', 'produced', 'production'],
        ['construction', 'structure'],
        ['captured', 'eventually', 'ultimately'],
        ['attempt', 'attempted'],
        ['economic', 'government'],
        ['critical', 'response'],

        ['collaboration', 'demonstrated'],
        ['incorporated', 'retained', 'subsequent', 'subsequently', 'transferred'],
        ['alternative', 'citing', 'instance', 'interpretation', 'perspective', 'whilst'],
        ['influential', 'prominent'],
        ['depicted', 'portrayed'],
        ['apparent', 'attributed', 'characterized', 'circumstances', 'controversy', 'pursuit', 'substantial'],
        ['instrumental', 'tribute'],
        ['alleged', 'investigation'],
        ['establishment', 'regime']]
    anchored_topic_model = ct.Corex(n_hidden=120, words=vocab, max_iter=10)
    anchored_topic_model.fit(tf, words=vocab, anchors=anchor_words, anchor_strength=4)

    joblib.dump(anchored_topic_model, 'pretrain_model/anchored_topic_model.pkl')
