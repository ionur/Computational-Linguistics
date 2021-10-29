import pandas as pd
import scipy.stats as stats
from pymagnitude import *


def compute_kt(file):
    vectors = Magnitude(file)
    df = pd.read_csv('SimLex-999.txt', sep='\t')[['word1', 'word2', 'SimLex999']]
    human_scores = []
    vector_scores = []
    for word1, word2, score in df.values.tolist():
        human_scores.append(score)
        similarity_score = vectors.similarity(word1, word2)
        vector_scores.append(similarity_score)
        # print(f'{word1},{word2},{score},{similarity_score:.4f}')

    correlation, p_value = stats.kendalltau(human_scores, vector_scores)
    print(f'Correlation = {correlation}, P Value = {p_value}')


def main():
    vectors = Magnitude('GoogleNews-vectors-negative300.magnitude')
    df = pd.read_csv('SimLex-999.txt', sep='\t')[['word1', 'word2', 'SimLex999']]
    human_scores = []
    vector_scores = []
    for word1, word2, score in df.values.tolist():
        human_scores.append(score)
        similarity_score = vectors.similarity(word1, word2)
        vector_scores.append(similarity_score)
        #print(f'{word1},{word2},{score},{similarity_score:.4f}')

    correlation, p_value = stats.kendalltau(human_scores, vector_scores)
    print(f'Correlation = {correlation}, P Value = {p_value}')

    # Least two pairs
    print('1. Least Two Pairs')

    print('Human Judgement')
    res = 100
    hum_index = -1
    i = 0
    for word1, word2, score in df.values.tolist():
        if (score < res):
            res = score
            hum_index = i
        i = i + 1

    print(df.iloc[[hum_index]])
    print('Vector Similarity')
    res = 100
    vec_index = -1
    i = 0
    for word1, word2, score in df.values.tolist():
        if (res > vectors.similarity(word1, word2)):
            res = vectors.similarity(word1, word2)
            vec_index = i
        i = i + 1

    print(df.iloc[[vec_index]])

    # Most two pairs
    print('1. Most Two Pairs')

    print('Human Judgement')
    res = -100
    hum_index = -1
    i = 0
    for word1, word2, score in df.values.tolist():
        if (res < score):
            res = score
            hum_index = i
        i = i + 1

    print(df.iloc[[hum_index]])
    print('Vector Similarity')
    res = -100
    vec_index = -1
    i = 0
    for word1, word2, score in df.values.tolist():
        if (res < vectors.similarity(word1, word2)):
            res = vectors.similarity(word1, word2)
            vec_index = i
        i = i + 1

    print(df.iloc[[vec_index]])

    # 1.

    print('glove.6B.50d.magnitude')
    compute_kt('glove.6B.50d.magnitude')

    # 2
    print('glove.6B.100d.magnitude')
    compute_kt('glove.6B.100d.magnitude')

    # 3
    print('glove.6B.200d.magnitude')
    compute_kt('glove.6B.200d.magnitude')

    # 4
    print('glove.6B.300d.magnitude')
    compute_kt('glove.6B.300d.magnitude')

    # 5
    print('glove.840B.300d.magnitude')
    compute_kt('glove.840B.300d.magnitude')

    # More Thorough Analysis

if __name__ == '__main__':
    main()