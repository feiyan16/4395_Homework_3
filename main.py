import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import numpy as np


def process(name):
    file = open(name)
    text = file.read()
    text = text.lower()  # make all to lower case
    text.replace('--', '')  # remove all '--' 
    text = re.sub(r'[0-9]', '', text)  # remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuations
    return nltk.word_tokenize(text)  # tokenize text


def stem_dictionary(tuples):
    dictionary = {}
    for word, stem in tuples:  # 7
        if stem not in dictionary.keys():
            dictionary[stem] = [word]
        else:
            dictionary[stem].append(word)
    return dictionary


def levenshteinDistance(token1, token2):
    row_size = len(token1) + 1  # number of rows
    col_size = len(token2) + 1  # number of columns
    distances = np.zeros((row_size, col_size))  # matrix to hold distances
    for col in range(col_size):
        distances[0][col] = col
    for row in range(row_size):
        distances[row][0] = row
    for row in range(1, row_size):
        for col in range(1, col_size):
            if token1[row - 1] == token2[col - 1]:
                distances[row][col] = distances[row - 1][col - 1]
            else:
                diag = distances[row - 1][col - 1]
                prev = distances[row][col - 1]
                top = distances[row-1][col]
                if diag <= prev and diag <= top:
                    distances[row][col] = diag + 1
                elif prev <= diag and prev <= top:
                    distances[row][col] = prev + 1
                elif top <= diag and top <= prev:
                    distances[row][col] = top + 1
    return distances[row_size - 1][col_size - 1]


def pos_dictionary(array):
    pos_dict = {}
    for tkn, tag in array:
        if tag not in pos_dict:
            pos_dict[tag] = 1
        else:
            pos_dict[tag] += 1
    print('12)')
    for pos in pos_dict:
        print(pos, ':', pos_dict[pos])


def edit_distance(dictionary):
    distances = []
    for key in dictionary:
        if key == 'continu':
            for word in dictionary[key]:
                distances.append(levenshteinDistance('continue', word))
            break
    print('10)', distances, '\n')


if __name__ == '__main__':
    tokens = process(sys.argv[1])
    print('3)', len(tokens), '\n')
    unique_tokens = set(tokens)
    print('4)', len(unique_tokens), '\n')
    impt_tokens = [t for t in unique_tokens if t not in stopwords.words('english')]
    print('5)', len(impt_tokens), '\n')
    word_stemmed_tuples = [(tkn, PorterStemmer().stem(tkn)) for tkn in impt_tokens]  # 6
    stem_dict = stem_dictionary(word_stemmed_tuples)  # 7
    print('8)', len(stem_dict), '\n')  # 8
    print('9)')
    for stem in sorted(stem_dict, key=lambda k: len(stem_dict[k]), reverse=True)[:25]:
        print(stem, ':', stem_dict[stem])  # 9
    print()
    edit_distance(stem_dict)
    pos_dictionary(nltk.pos_tag(tokens))
