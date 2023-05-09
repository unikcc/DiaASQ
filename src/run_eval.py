#!/usr/bin/env python


"""
Name: run_eval.py
"""

import argparse
import json

"""usage
python src/run_eval.py --pred_file xxx --gold_file xxx
"""

class Template:
    def __init__(self, config):
        self.pred_file = config.pred_file
        self.gold_file = config.gold_file

    def read_data(self, path, pd=None):
        with open(path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        res = {}
        if pd is not None:
            content = [w for w in content if w['doc_id'] in pd]
        for line in content:
            doc_id = line['doc_id']
            triplets = line['triplets']
            cur_res = set()
            for comb in triplets:
                if any(w == -1 for w in comb[:6]): continue
                assert all(isinstance(w, int) for w in comb[:6])
                assert isinstance(comb[6], str)
                assert len(comb) == 10
                comb[6] = comb[6] if comb[6] in ['pos', 'neg'] else 'other'
                cur_res.add(tuple(comb[:7]))
            cur_res = sorted(list(cur_res), key=lambda x: (x[0], x[2], x[4]))
            res[doc_id] = cur_res

        return res

    def compute_score(self, preds, golds, mode='quad'):

        # assert all(doc_id in golds for doc_id in preds)
        # assert all(doc_id in preds for doc_id in golds)
        tp, fp, fn = 0, 0, 0
        for doc_id in preds:
            pred_line = preds[doc_id]
            gold_line = golds[doc_id]

            if mode != 'quad':
                pred_line = [w[:6] for w in pred_line]
                gold_line = [w[:6] for w in gold_line]

            fp += len(set(pred_line) - set(gold_line))
            fn += len(set(gold_line) - set(pred_line))
            tp += len(set(pred_line) & set(gold_line))

        p = tp / (tp + fp) if tp + fp > 0 else 0
        r = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        scores = [p, r, f1, tp, tp + fp, tp + fn]
        return scores

    def forward(self, print_line=False):
        pred_res = self.read_data(self.pred_file)
        gold_res = self.read_data(self.gold_file, pd=pred_res)
        micro_score = self.compute_score(pred_res, gold_res)
        iden_score = self.compute_score(pred_res, gold_res, mode='iden')
        res = 'Item\tPrec.\tRec.\tF1\tTP\tPred.\tGold.\n'
        res += 'Micro\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}\n'.format(*micro_score)
        res += 'Iden\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}'.format(*iden_score)
        if print_line:
            print(res)
        return micro_score, iden_score, res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred_file', default='data/save/pred_zh_test.json', type=str, required=False, help='pred file name with directory name')
    parser.add_argument('-g', '--gold_file', default='data/dataset/jsons_zh/test.json', type=str, required=False, help='gold file name with directory name')
    args = parser.parse_args()
    template = Template(args)
    template.forward(True)