import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec, word2vec, KeyedVectors
from gensim.models.word2vec import LineSentence
import jieba
import jieba.analyse
import re
import nltk

def doc_vec():
    path = "./TF_beg_api.txt"
    # path = "./TF_mal_api.txt"
    num = 1
    sentences = []
    totals = 0
    with open(path) as f:
        for line in f.readlines():
            if line.strip() != "":
                terms = line.strip().split()
                sentences.append(terms)

                vec_len = len(sentences[0])
                if len(sentences[0]) <= 1:
                    sentences[0].append(" ")
                    vec_len = len(sentences[0])
                elif len(sentences[0]) > 150:
                    vec_len = 150

                w2v = word2vec.Word2Vec(sentences, hs=1, sg=1, min_count=0, window=5, vector_size=vec_len, workers=4)
                with open('./api_vec/beg/Vec_beg_{}.csv'.format(num), 'w', newline='') as file:
                # with open('./api_vec/mal/Vec_mal_{}.csv'.format(num), 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(w2v.wv.vectors)
                print('第{}个样本写入成功！'.format(num))
                sentences = []
                num += 1
                totals += 1
        print('总计写入{}个样本'.format(totals))

def DCT_quantify(data, block_size=8, set_mask=None):
    h, w = data.shape
    if type(set_mask) == np.ndarray:
        mask = set_mask
    else:
        mask = np.ones([block_size, block_size])

    # if (h_to_pad := (h % block_size)) != 0:
    #     data = np.pad(data, ((0, block_size - h_to_pad), (0, 0)))
    # if (w_to_pad := (w % block_size)) != 0:
    #     data = np.pad(data, ((0, 0), (0, block_size - w_to_pad)))

    h_to_pad = (h % block_size)
    w_to_pad = (w % block_size)
    if h_to_pad != 0:
        data = np.pad(data, ((0, block_size - h_to_pad), (0, 0)))
    if w_to_pad != 0:
        data = np.pad(data, ((0, 0), (0, block_size - w_to_pad)))

    new_h, new_w = data.shape

    v_slices_num = new_h // block_size
    h_slices_num = new_w // block_size

    hori_data = np.vsplit(data, indices_or_sections=v_slices_num)

    for i, row in enumerate(hori_data):

        vert_data = np.hsplit(row, indices_or_sections=h_slices_num)
        first_v_block = cv2.dct(vert_data[0].astype(np.float32))

        first_v_block = np.multiply(first_v_block, mask)

        first_v_iblock = cv2.idct(first_v_block)
        dct_block_rows = first_v_block
        idct_block_rows = first_v_iblock
        for j, block in enumerate(vert_data[1:]):

            single_block = cv2.dct(block.astype(np.float32))

            single_block = np.multiply(single_block, mask)

            single_iblock = cv2.idct(single_block)

            dct_block_rows = np.hstack([dct_block_rows, single_block])
            idct_block_rows = np.hstack([idct_block_rows, single_iblock])

        if i == 0:
            dct_img = dct_block_rows
            idct_img = idct_block_rows
        else:
            dct_img = np.vstack([dct_img, dct_block_rows])
            idct_img = np.vstack([idct_img, idct_block_rows])

    return dct_img, idct_img

def phash():

    with open('/Vec_phash.csv', 'a', newline='', encoding='gb18030') as file:
        writer = csv.writer(file)
        max_num = 0
        min_num = 0
        for x in range(301,1001):
            # if x >= 10:
            #     break
            str = ''
            with open('./api_vec/mal/Vec_mal_{}.csv'.format(x), 'r', encoding='gb18030') as f:
            # with open('./api_vec/beg/Vec_beg_{}.csv'.format(x), 'r', encoding='gb18030') as f:
                reader = csv.reader(f)
                rows1 = [row for row in reader]
            rows = np.array(rows1)
            # print(rows.shape)
            mask = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

            B_dct, B_idct = DCT_quantify(rows)
            mean = 0
            for i, index in enumerate(B_dct):
                for j, indexs in enumerate(B_dct[i]):
                    mean += B_dct[i][j]
                # break
            mean = mean / (len(B_dct) * len(B_dct[0]))

            for i, index in enumerate(B_dct):
                for j, indexs in enumerate(B_dct[i]):
                    if B_dct[i][j] >= mean:
                        max_num += 1
                    else:
                        min_num += 1
                if max_num >= min_num:
                    str += "1"
                else:
                    str += "0"
                max_num = 0
                min_num = 0
            # writer.writerow(['0', str])
            writer.writerow(['1', str])
            print("第{}个样本写入!".format(x))
            # print(str)
            str = ''
            
#将API句向量和参数句向量结合
def api_C():
    rows = []
    with open('/TF_api.csv', 'r', encoding='gb18030') as f:
        reader = csv.reader(f)
        rows1 = [row for row in reader]
    with open('/TF_api_c.csv', 'r', encoding='gb18030') as f:
        reader = csv.reader(f)
        rows2 = [row for row in reader]
    with open('./doc_TF.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        # 恶意标签为1，良性标签为0
        for i in range(2000):
            rows = rows1[i][1] + rows2[i][1]
            # 融合的向量根据良性向量数量划分，前i个向量为良性向量,从0开始
            if i < 1000:
               writer.writerow(['0', rows])
            else:
                writer.writerow(['1', rows])
            print("第{}个向量写入!".format(i))

if __name__ == "__main__":
    # doc_vec()
    # phash()
    #api_c()
    print("end")
