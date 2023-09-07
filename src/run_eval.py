#!/usr/bin/env python


"""
Name: run_eval.py
"""

import argparse
import json
import numpy as np

"""usage
python src/run_eval.py --pred_file xxx --gold_file xxx
"""

def get_token_thread(dialogue):
    sentences = dialogue['sentences']
    replies = dialogue['replies']

    sentence_ids = [[i] * len(w.split()) for i, w in enumerate(sentences)]
    sentence_ids = [w for sent in sentence_ids for w in sent]

    thread_list = [[0]]
    cur_thread = []
    for i, r in enumerate(replies):
        if i == 0: continue
        if r > replies[i - 1]:
            cur_thread.append(i)
        else:
            thread_list.append(cur_thread)
            cur_thread = [i]
    if len(cur_thread) > 0:
        thread_list.append(cur_thread)
    
    dis_matrix = np.zeros([len(replies), len(replies)], dtype=int)
    for i in range(len(thread_list)):
        first_list = thread_list[i]
        for ii in range(len(first_list)):
            for j in range(i, len(thread_list)):
                second_list = thread_list[j]
                for jj in range(len(second_list)):
                    if i == j:
                        dis_matrix[first_list[ii], second_list[jj]] = abs(ii - jj)
                        dis_matrix[second_list[jj], first_list[ii]] = abs(ii - jj)
                    elif i * j == 0:
                        dis_matrix[first_list[ii], second_list[jj]] = ii + jj + 1
                        dis_matrix[second_list[jj], first_list[ii]] = ii + jj + 1
                    else:
                        dis_matrix[first_list[ii], second_list[jj]] = ii + jj + 2
                        dis_matrix[second_list[jj], first_list[ii]] = ii + jj + 2
    
    return dis_matrix, sentence_ids

def get_utterance_distance(sentence_ids, dis_matrix, index0, index1, index2):
    def get_pair_distance(id0, id1):
        sent0 = sentence_ids[id0]
        sent1 = sentence_ids[id1]
        return dis_matrix[sent0, sent1]
    dis0 = get_pair_distance(index0, index1)
    dis1 = get_pair_distance(index1, index2)
    dis2 = get_pair_distance(index0, index2)
    return max(dis0, dis1, dis2)

class Template:
    def __init__(self, config):
        self.pred_file = config.pred_file
        self.gold_file = config.gold_file

    def read_data(self, path, mode='pred'):
        with open(path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            content = {w['doc_id']: w for w in content}
        if mode == 'pred': return content

        new_content = {}
        for k, line in content.items():
            triplets = line['triplets']
            ta = [tuple(w[:4]) for w in triplets if all(z != -1 for z in w[:4])]
            to = [tuple(w[0:2] + w[4:6]) for w in triplets if all(z != -1 for z in w[:2] + w[4:6])]
            ao = [tuple(w[2:6]) for w in triplets if all(z != -1 for z in w[2:6])]

            dis_matrix, sentence_ids = get_token_thread(line)
            line.update({'ta': ta, 'to': to, 'ao': ao, 'dis_matrix': dis_matrix, 'sentence_ids': sentence_ids})
            new_content[k] = line

        return new_content

    def post_process(self, line, key='quad'):
        if key in ['targets', 'aspects', 'opinions']:
            return [tuple(w[:2]) for w in line[key]]
        if key in ['ta', 'to', 'ao']:
            return [tuple(w[:4]) for w in line[key]]

        res = []
        if key in ['quad', 'iden']:
            for comb in line['triplets']:
                if any(w == -1 for w in comb[:6]): continue
                assert all(isinstance(w, int) for w in comb[:6])
                assert isinstance(comb[6], str) 
                assert len(comb) == 10
                comb[6] = comb[6] if comb[6] in ['pos', 'neg'] else 'other'
                res.append(tuple(comb[:6 if key == 'iden' else 7]))
            return res
        if key in ['intra', 'inter']:
            for comb in line['triplets']:
                if any(w == -1 for w in comb[:6]): continue
                comb[6] = comb[6] if comb[6] in ['pos', 'neg'] else 'other'
                distance = get_utterance_distance(line['sentence_ids'], line['dis_matrix'], comb[0], comb[2], comb[4])
                if key == 'intra' and distance > 0: continue
                if key == 'inter' and distance == 0: continue
                res.append(tuple(comb[:7]))
            return res
        if key in ['cross-1', 'cross-2', 'cross-3']:
            for comb in line['triplets']:
                if any(w == -1 for w in comb[:6]): continue
                comb[6] = comb[6] if comb[6] in ['pos', 'neg'] else 'other'
                distance = get_utterance_distance(line['sentence_ids'], line['dis_matrix'], comb[0], comb[2], comb[4])
                if key == 'cross-1' and distance != 1: continue
                if key == 'cross-2' and distance != 2: continue
                if key == 'cross-3' and distance <3: continue
                res.append(tuple(comb[:7]))
            return res
        raise ValueError('Invalid key: {}'.format(key))


    def compute_score(self, mode='quad'):

        tp, fp, fn = 0, 0, 0
        for doc_id in self.gold_res:
            pred_line = self.pred_res[doc_id]
            gold_line = self.gold_res[doc_id]

            pred_line['sentence_ids'] = gold_line['sentence_ids']
            pred_line['dis_matrix'] = gold_line['dis_matrix']

            pred_line = self.post_process(pred_line, mode)
            gold_line = self.post_process(gold_line, mode)

            fp += len(set(pred_line) - set(gold_line))
            fn += len(set(gold_line) - set(pred_line))
            tp += len(set(pred_line) & set(gold_line))

        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        scores = [p, r, f1]
        return scores

    def forward(self, print_line=False):
        self.pred_res = self.read_data(self.pred_file, 'pred')
        self.gold_res = self.read_data(self.gold_file, 'gold')

        assert len(self.pred_res) == len(self.gold_res)
        assert all(k in self.gold_res for k in self.pred_res)

        scores = []
        res = 'Item\tPrec.\tRec.\tF1\n'
        items = ['targets', 'aspects', 'opinions', 'ta', 'to', 'ao', 'quad', 'iden', 'intra', 'inter', 'cross-1', 'cross-2', 'cross-3']
        item_name = ['Target', 'Aspect', 'Opinion', 'TA', 'TO', 'AO', 'Micro', 'Iden', 'Intra', 'Inter', 'Cross-1', 'Cross-2', 'Cross-3']
        num_format = lambda x: '\t' + '\t'.join([f'{w*100:.2f}' if i < 3 else str(w) for i, w in enumerate(x)]) + '\n'
        line_indices = [0, 3, 6, 8]

        for i, item in enumerate(items):
            if i in line_indices : res += '-'*30 + '\n'
            score = self.compute_score(item)
            scores.append(score)

            res += item_name[i] + num_format(score)
        res += '-'*30
    
        if print_line:
            print(res)

        micro_score, iden_score = scores[6], scores[7]
        return micro_score, iden_score, res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_file', default='data/save/pred_zh_test.json', type=str, required=False, help='pred file name with directory name')
    parser.add_argument('-g', '--gold_file', default='data/dataset/jsons_zh/test.json', type=str, required=False, help='gold file name with directory name')
    args = parser.parse_args()
    template = Template(args)
    template.forward(True)