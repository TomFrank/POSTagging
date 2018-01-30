import csv
import os
import re
import time
import numpy as np
from multiprocessing import Pool
from functools import partial

corpora_filename = 'people201401.txt'
input_filename = 'test_data_input.txt'
output_filename = 'test_output.txt'
regular_expression_of_word_tag_pair = '([^ \[\]]+)/(\w+)'

correct_ans_filename = 'test_data_ans.txt'


def get_trans_matrix(tag2index_dict):
    trans_mat_file_path = './model_data/trans_matrix.npy'
    if not os.path.exists(trans_mat_file_path):
        with open('./corp/' + corpora_filename, 'r', encoding='utf-8') as f:
            trans_mat = np.zeros([len(tag2index_dict), len(tag2index_dict)], np.float32)
            p = re.compile(regular_expression_of_word_tag_pair)
            for line in f:
                tags = p.findall(line)
                for i in range(1, len(tags)):
                    index_i = tag2index_dict[tags[i][1]]
                    index_im1 = tag2index_dict[tags[i - 1][1]]
                    trans_mat[index_i][index_im1] += 1
            for row in trans_mat:
                row /= np.sum(row)
            np.save(trans_mat_file_path, trans_mat)
            return trans_mat
    else:
        trans_mat = np.load(trans_mat_file_path)
        return trans_mat


def get_emission_matrix(tag2index_dict, word2index_dict):
    emission_file_path = './model_data/emission_matrix.npy'
    if not os.path.exists(emission_file_path):
        with open('./corp/' + corpora_filename, 'r', encoding='utf-8') as f:
            p = re.compile(regular_expression_of_word_tag_pair)
            emission_mat = np.zeros([len(tag2index_dict), len(word2index_dict)], dtype=np.float32)
            for line in f:
                word_tag_pairs = p.findall(line)
                for pair in word_tag_pairs:
                    word_index = word2index_dict[pair[0]]
                    tag_index = tag2index_dict[pair[1]]
                    emission_mat[tag_index][word_index] += 1
            e_mat_trans = emission_mat.T
            for row in e_mat_trans:
                row /= np.sum(row)
            np.save(emission_file_path, emission_mat)
            return emission_mat
    else:
        emission_mat = np.load(emission_file_path)
        return emission_mat


def get_tag2index_dict():
    tag_file_path = './model_data/tags.txt'
    if not os.path.exists(tag_file_path):
        try:
            tags = set()
            with open('./corp/' + corpora_filename, 'r', encoding='utf-8') as f:
                p = re.compile(regular_expression_of_word_tag_pair)
                for line in f:
                    ts = p.findall(line)
                    for t in ts:
                        tags.add(t[1])
            tags = list(sorted(tags))
            with open(tag_file_path, 'w', encoding='utf-8', newline='') as f:
                for t in tags:
                    f.write(t + '\n')
            tags = {t: i for i, t in enumerate(tags)}
            return tags
        except OSError:
            print('Can not open corpora file:' + corpora_filename)
    else:
        with open(tag_file_path, 'r', encoding='utf-8', newline='\n') as f:
            tags = {t.strip('\n'): i for i, t in enumerate(f)}
            return tags


def get_word2index_dict():
    vocab_file_path = './model_data/vocab.csv'
    if not os.path.exists(vocab_file_path):
        try:
            vocab = {}
            with open('./corp/' + corpora_filename, 'r', encoding='utf-8') as f:
                p = re.compile(regular_expression_of_word_tag_pair)
                for line in f:
                    ws = p.findall(line)
                    for w in ws:
                        if w[0] not in vocab:
                            vocab[w[0]] = 1
                        else:
                            vocab[w[0]] += 1
            vocab = sorted(vocab.items(), key=lambda v: v[1], reverse=True)
            vocab = [(w[0], i) for i, w in enumerate(vocab)]
            with open(vocab_file_path, 'w', encoding='utf-8', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerows(vocab)
            return {w: i for w, i in vocab}
        except OSError:
            print('Can not open corpora file:' + corpora_filename)
    else:
        with open(vocab_file_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            return {row[0]: int(row[1]) for row in csv_reader}


def read_data(word2index_dict):
    try:
        input_list = []
        with open('./' + input_filename, 'r', encoding='utf-8') as f:
            for line in f:
                one_line = []
                for word in line.split():
                    if word in word2index_dict:
                        one_line.append(word2index_dict[word])
                    else:
                        one_line.append(-1)
                if len(one_line) > 0:
                    input_list.append(one_line)
        return input_list
    except OSError:
        print('Can not open input data file')


def del_data_tag():
    if not os.path.exists('./' + input_filename):
        with open('./test_data_ans.txt', 'r', encoding='utf-8') as fin:
            with open('./' + input_filename, 'w', encoding='utf-8') as fout:
                p = re.compile(regular_expression_of_word_tag_pair)
                for line in fin:
                    ws = p.findall(line)
                    fout.writelines(list(map(lambda w: w[0] + ' ', ws)))
                    fout.write('\n')


def get_init_prob(tag2index_dict):
    init_prob_file_path = './model_data/init_prob.npy'
    if not os.path.exists(init_prob_file_path):
        with open('./corp/' + corpora_filename, 'r', encoding='utf-8') as fin:
            p = re.compile(regular_expression_of_word_tag_pair)
            init_prob = np.zeros([len(tag2index_dict)], dtype=np.float32)
            for line in fin:
                words = line.split()
                if len(words) != 0:
                    fst_tag = p.findall(line)[0][1]
                    init_prob[tag2index_dict[fst_tag]] += 1
            init_prob /= np.sum(init_prob)
            np.save(init_prob_file_path, init_prob)
            return init_prob
    else:
        init_prob = np.load(init_prob_file_path)
        return init_prob


def viterbi(tag2index_dict, trans_mat, ems_mat, init_prob, input_data):
    T = len(input_data)
    K = len(tag2index_dict)
    t1 = np.zeros([K, T], dtype=np.float32)
    t2 = np.zeros([K, T], dtype=np.int)
    log_ems_mat = np.log(ems_mat)
    log_trans_mat = np.log(trans_mat)
    t1.T[0] = np.log(init_prob) + log_ems_mat.T[input_data[0]]
    # with Pool(processes=os.cpu_count()) as pool:
    for i in range(1, T):
        for j in range(0, K):
            max_idx = np.argmax(t1.T[i - 1] + log_trans_mat.T[j])
            t1[j][i] = t1[max_idx][i - 1] + log_trans_mat[max_idx][j] + log_ems_mat[j][input_data[i]]
            t2[j][i] = max_idx
    index2tag_dict = dict(zip(tag2index_dict.values(), tag2index_dict.keys()))
    z = np.zeros([T], dtype=np.int)
    x = []
    z[T - 1] = np.argmax(t1.T[T - 1])
    x.append(index2tag_dict[z[T - 1]])
    for i in range(T - 1, 0, -1):
        z[i - 1] = t2[z[i]][i]
        x.append(index2tag_dict[z[i - 1]])
    return list(reversed(x))


def POS_tagging(tag2index_dict, trans_mat, ems_mat, init_prob, input_data):
    # with Pool(processes=4) as pool:
    tag_one_sentence = partial(viterbi, tag2index_dict, trans_mat, ems_mat, init_prob)
    tags_lists = list(map(tag_one_sentence, input_data))
    return tags_lists


def combine_input_and_tag(input_data, tags_array, word2index_dict):
    output = []
    index2word_dict = dict(zip(word2index_dict.values(), word2index_dict.keys()))
    for one_line_data, one_line_tags in zip(input_data, tags_array):
        output.append([index2word_dict[data] + '/' + tag for data, tag in zip(one_line_data, one_line_tags)])
    return output


def save_ans(ans):
    with open('./' + output_filename, 'w', encoding='utf-8') as f:
        for line in ans:
            print(*line, file=f, sep=' ')


def eval_accuracy(predict_tag):
    ans_tag_filename = './ans_tag.txt'
    ans_tags = []
    if not os.path.exists(ans_tag_filename):
        with open('./' + correct_ans_filename, 'r', encoding='utf-8') as f:
            p = re.compile(regular_expression_of_word_tag_pair)
            for line in f:
                ms = p.findall(line)
                if len(ms) != 0:
                    ans_tags.append([m[1] for m in ms])
        with open(ans_tag_filename, 'w', encoding='utf-8') as fout:
            for l in ans_tags:
                fout.writelines(list(map(lambda x: x + ' ', l)))
                fout.write('\n')
    else:
        with open(ans_tag_filename, 'r', encoding='utf-8') as f:
            for line in f:
                ans_tags.append(line.split())

    cmp_mat = list(
        map(lambda xa, xb: list(map(lambda xxa, xxb: 1. if xxa == xxb else 0., xa, xb)), ans_tags, predict_tag))
    word_count = sum([len(l) for l in cmp_mat])
    accuracy = sum([sum(l) for l in cmp_mat])
    accuracy /= word_count
    return accuracy


def main():
    t0 = time.process_time()
    # del_data_tag()
    word2index_dict = get_word2index_dict()
    # print('word dict:\n', list(word_dict.items())[:10])
    tag2index_dict = get_tag2index_dict()
    # print('tag dict:\n', len(tags_dict), tags_dict)
    input_data = read_data(word2index_dict=word2index_dict)
    # print('input: \n', input_data[:50])
    tran_mat = get_trans_matrix(tag2index_dict=tag2index_dict)
    # print('trans mat:\n', tran_mat)
    ems_mat = get_emission_matrix(tag2index_dict=tag2index_dict,
                                  word2index_dict=word2index_dict)
    # print('emi mat:\n', emi_mat)
    init_prob = get_init_prob(tag2index_dict=tag2index_dict)

    predict_tags = POS_tagging(tag2index_dict=tag2index_dict,
                               trans_mat=tran_mat,
                               ems_mat=ems_mat,
                               init_prob=init_prob,
                               input_data=input_data,)
    # print(predict_tags[:50])
    ans = combine_input_and_tag(word2index_dict=word2index_dict,
                                tags_array=predict_tags,
                                input_data=input_data)
    accuracy = eval_accuracy(predict_tags)
    delta_t = time.process_time() - t0
    # save_ans(ans)

    print(ans[:50])
    print('accuracy: ', accuracy)
    print('time: ', delta_t)

if __name__ == '__main__':
    main()
