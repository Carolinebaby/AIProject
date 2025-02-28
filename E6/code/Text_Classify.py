from time import time
import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def compute_cosine_distance(v1, v2):
    return cosine(v1, v2)


def main():
    start_time = time()
    train_words_list = []
    train_words_tags = []
    test_words_list = []
    test_words_tags = []
    emotion_dict = {}
    # 数据预处理
    # 处理训练数据
    train_file = open('train.txt', 'r')
    train_txt = train_file.readlines()
    for line in train_txt[1:]:
        line_words = line.split(' ')
        e_id = int(line_words[1])
        e_str = line_words[2]
        emotion_dict[e_id] = e_str
        train_words_tags.append(e_id)
        words = ' '.join(line_words[3:])
        train_words_list.append(words)

    # 处理测试数据
    test_file = open('test.txt', 'r')
    test_txt = test_file.readlines()
    for line in test_txt[1:]:
        line_words = line.split(' ')
        e_id = int(line_words[1])
        test_words_tags.append(e_id)
        words = ' '.join(line_words[3:])
        test_words_list.append(words)

    # k-NN 算法
    # 对 train.txt 的所有句子进行特征获取
    tfidf_vec = TfidfVectorizer(norm='l1')
    tfidf_matrix = tfidf_vec.fit_transform(train_words_list)
    train_size = tfidf_matrix.shape[0]  # 得到训练的句子数量
    # print(tfidf_matrix.shape)
    # 得到 numpy 的 array 格式的 tfidf矩阵
    train_tfidf = tfidf_matrix.toarray()

    k = 15  # 参数
    # 得到 所有测试句子的 tfidf向量 的值
    test_tfidf = tfidf_vec.transform(test_words_list).toarray()
    test_size = test_tfidf.shape[0]  # 测试的句子数量
    success_cnt, fail_cnt = 0, 0  # 成功分类的数量和失败的数量
    # 遍历所有的 测试案例
    for i in range(test_size):
        # 计算 测试句子 和 所有训练的句子 的 tf-idf 的向量的 “距离”
        # 余弦距离  # 在处理 i=20 的数据的时候会出现除法异常
        # distances = np.array([compute_cosine_distance(train_tfidf[j], test_tfidf[i]) for j in range(train_size)])
        # 欧氏距离
        # distances = np.array([compute_euclidean_distance(train_tfidf[j], test_tfidf[i]) for j in range(train_size)])
        # 曼哈顿距离，利用矩阵运算进行优化
        distances = np.sum(np.abs(train_tfidf-test_tfidf[i]), axis=1)
        # 获得前k短距离的训练样例的下标
        nearest_neighbors = np.argsort(distances)[:k]
        # 得到前k短距离的训练样例对应的 情感标签
        neighbor_tags = [train_words_tags[j] for j in nearest_neighbors]
        # 得到 情感标签最多的对应的 情感标签
        common_tag = Counter(neighbor_tags).most_common(1)[0][0]
        if common_tag == test_words_tags[i]:  # 如果情感标签相同
            # print(i, emotion_dict[common_tag], "success")
            success_cnt += 1
        else:
            # print(i, emotion_dict[common_tag], "fail", emotion_dict[test_words_tags[i]])
            fail_cnt += 1

    end_time = time()
    print("total time:", end_time - start_time)
    print("success total:", success_cnt, " fail total:", fail_cnt)
    print("Probability of success:", success_cnt / test_size)
    return 0


if __name__ == '__main__':
    main()
