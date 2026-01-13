# This is a sample Python script.
import csv
import time
import tensorflow as tf
import pandas as pd
import numpy as np
from gensim.models import Word2Vec, word2vec, KeyedVectors
from gensim.models.word2vec import LineSentence
import jieba
import jieba.analyse
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
nltk.download("punkt")

def test_mal():
    with open('malware.csv', 'r', encoding='gb2312') as f:
        reader = csv.reader(f)
        print(reader)
        rows = [row for row in reader]
        str_test =''
        # file_handle = open('test_mal.txt', mode='w')
        file_handle = open('api_mal.txt', mode='w')
        for index, i in enumerate(rows):
            for indexs, item in enumerate(rows[index]):
                rows[index][indexs] = (item)
                if indexs <= 1:
                    continue
                str_test = str_test +' ' + str(rows[index][indexs])
            file_handle.write(str_test+'\n')
            print('第' + str(index) + '行提取完成')
            str_test = ''
            if index >= 2000:
                break
        file_handle.close()

def test_beg():
    with open('goodware.csv', 'r', encoding='gb2312') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        str_test =''
        file_handle = open('api_beg.txt', mode='w')
        for indexs, item in enumerate(rows):
            if indexs == 0:
                continue
            if indexs >= 2001:
                break
            matches = re.findall(r"'(.+?)'",rows[indexs][10])
            for index,i in enumerate(matches):
                str_test = str_test +' ' + str(matches[index])
            file_handle.write(str_test+'\n')
            print('第' + str(indexs) + '行提取完成')
            str_test = ''
        file_handle.close()

def test():
    path = "./test_beg.txt"
    # path = "./test_mal.txt"
    num = 1
    sentences = []
    sen = [[]]
    total_nor = 0
    total_felnor = 0
    total_x = 0
    with open(path) as f:
        for line in f.readlines():
            if line.strip() != "":
                tokens = nltk.word_tokenize(line.strip())
                sentences.append(tokens)
                # print(sentences)
                # print(len(sentences[0]))
                sen[0] = line.strip()
                vectorizer = CountVectorizer(max_features=1000)
                tf_idf_transformer = TfidfTransformer()
                tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(sen))
                x_train_weight = tf_idf.toarray()
                if len(sentences[0]) >= 150:
                    lens = 150
                elif len(sentences[0]) <= 1:
                    print('第{}个向量无效'.format(num))
                    total_x += 1
                    num += 1
                    sentences = []
                    continue
                else:
                    lens = len(sentences[0])
                w2v = word2vec.Word2Vec(sentences, hs=1, sg=1, min_count=0, window=5, vector_size=lens, workers=4)
                # print('向量数量：')
                if len(w2v.wv.vectors) == len(x_train_weight[0]):
                    print('第{}个向量正常!'.format(num))
                    total_nor += 1
                else:
                    print('第{}个向量不正常'.format(num))
                    total_felnor += 1

            sentences = []
            num += 1
        print('有{}个向量正常'.format(total_nor))
        print('有{}个向量不正常'.format(total_felnor))
        print('有{}个向量无效'.format(total_x))

def preHandel(path):
    #st = time.time()
    num = 0
    sentences = []
    with open(path) as f:
        for line in f.readlines():
            if line.strip() != "":
                # `[^\w\s]` 匹配除了字母、数字和空格之外的所有字符
                content = re.sub('[^\w\s]', '', line.strip())
                content_seq = list(jieba.cut(content))
                sentences.append(content_seq)
                print(sentences)
                sentences = []
                num += 1
            if num >= 5:
                break
    #end = time.time()
    #print("PreHandel End Num:%s Cost:%ss" % (num, (end - st)))
    return sentences

def getSimilarSeq(key, model, top=10):
    print("Top For %s ======================" % key)
    sims = model.wv.most_similar(key, topn=top)
    for i in sims:
        print(i)
    print("End Sim For %s ======================" % key)

def Vec():
    #path = "./api_beg.txt"
    path = "./api_mal.txt"
    num = 1
    sentences = []
    with open(path) as f:
        for line in f.readlines():
            if line.strip() != "":
                content = re.sub('[^\w\s]', '', line.strip())
                content_seq = list(jieba.cut(content))
                sentences.append(content_seq)
                w2v = word2vec.Word2Vec(sentences, hs=1, sg=1, min_count=0,
                                        window=5, vector_size=150,workers=4)
                with open('./api_vec/mal/Vec_mal_{}.csv'.format(num), 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(w2v.wv.vectors)
                print('第{}个向量写入成功！'.format(num))
                sentences = []
                num += 1

def save_Vec(ls):
    with open('Vec_one.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(ls)

def doc_vec_TF():
    path = "./api_beg.txt"
    # path = "./api_mal.txt"
    num = 1
    sentences = []
    sen = [[]]
    doc = []
    totals = 0
    with open(path) as f:
        for line in f.readlines():
            if line.strip() != "":
                tokens = nltk.word_tokenize(line.strip())
                sentences.append(tokens)
                sen[0] = line.strip()
                vectorizer = CountVectorizer(max_features=1000)
                tf_idf_transformer = TfidfTransformer()
                tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(sen))
                x_train_weight = tf_idf.toarray()
                if len(sentences[0]) >= 150:
                    lens = 150
                elif len(sentences[0]) <= 1:
                    num += 1
                    sentences = []
                    continue
                else:
                    lens = len(sentences[0])

                w2v = word2vec.Word2Vec(sentences, hs=1, sg=1, min_count=0, window=5, vector_size=lens, workers=4)
                if len(w2v.wv.vectors) != len(x_train_weight[0]):
                    sentences = []
                    num += 1
                    continue
                total = 0 * w2v.wv.vectors[0]
                for index, i in enumerate(w2v.wv.vectors):
                    total = total + x_train_weight[0][index] * i
                    #print(index)
                # print('第{}个句子向量获取成功'.format(num))

                doc = total
                with open('./doc_vec_TF.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    # 恶意标签为1，良性标签为0
                    writer.writerow(['0', doc])
                print('第{}个向量写入成功！'.format(num))
                totals += 1
                sentences = []
                num += 1
        print('总计写入{}个向量'.format(totals))


if __name__ == '__main__':
    # test_beg()
    # test_mal()
    # test()
    # doc_vec_TF()
    # word_vec()
    # Vec()
    print('end!')

