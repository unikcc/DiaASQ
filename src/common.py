
import numpy as np
from collections import defaultdict

import os
import random
import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class WordPair:
    def __init__(self, max_sequence_len=512):
        self.max_sequence_len = max_sequence_len

        self.entity_dic = {"O": 0, "ENT-T": 1, "ENT-A": 2, "ENT-O": 3}

        self.rel_dic = {"O": 0, "h2h": 1, "t2t": 2}

        self.polarity_dic = {"O": 0, "pos": 1, "neg": 2, 'other': 3}

    def encode_entity(self, elements, entity_type='ENT-T'):
        '''
        Convert the elements in the dataLoader to a list of entities rel_list.
        The format is [(starting position, ending position, entity type in the dictionary)].
        '''
        entity_list = []
        for line in elements:
            start, end = line[:2]
            entity_list.append((start, end, self.entity_dic[entity_type]))
        return entity_list

    def encode_relation(self, triplets):
        '''
        Convert the triplets in the dataLoader to a list of relations `rel_list`.
        Each relation is represented as a tuple with three elements: the starting position, the ending position, and the relation type in the dictionary.
        '''
        rel_list = []
        for triplet in triplets:
            s_en, e_en, s_as, e_as, s_op, e_op, polar = triplet
            # Add head-to-head relations for the quadruples to head_rel_list
            # Add relation from entity to aspect

            if s_en != -1 and s_as != -1:
                rel_list.append((s_en, s_as, self.rel_dic['h2h']))
                rel_list.append((e_en, e_as, self.rel_dic['t2t']))

            # Add relation from entity to opinion
            if s_en != -1 and s_op != -1:
                rel_list.append((s_en, s_op, self.rel_dic['h2h']))
                rel_list.append((e_en, e_op, self.rel_dic['t2t']))

            # Add relation from aspect to opinion
            if s_as != -1 and s_op != -1:
                rel_list.append((s_as, s_op, self.rel_dic['h2h']))
                rel_list.append((e_as, e_op, self.rel_dic['t2t']))

        return rel_list
    
    def encode_polarity(self, triplets):
        '''
        Convert triplets in the dataLoader to polarity.
        Each polarity is represented as a tuple with three elements: the starting position, the ending position, and the polarity category.
        '''
        rel_list = []
        for triplet in triplets:
            s_en, e_en, s_as, e_as, s_op, e_op, polar = triplet
            # Add head-to-head relations for the quadruples->head_rel_list
            # Add relation entity->opinion
            rel_list.append((s_en, s_op, polar))
            rel_list.append((e_en, e_op, polar))

        return rel_list

    def list2rel_matrix4batch(self, batch_rel_list, seq_len=512):
        '''
        Convert a sentence's relation list to a matrix.
        batch_rel_matrix:[batch_size, seq_len, seq_len]
        '''
        rel_matrix = np.zeros([len(batch_rel_list), seq_len, seq_len], dtype=int)
        for batch_id, rel_list in enumerate(batch_rel_list):
            for rel in rel_list:
                rel_matrix[batch_id, rel[0], rel[1]] = rel[2]
        return rel_matrix.tolist()

    # Decoding section
    def rel_matrix2list(self, rel_matrix):
        '''
        Convert a (512*512) matrix to a list of relations.
        '''
        rel_list = []
        nonzero = rel_matrix.nonzero()
        for x_index, y_index in zip(*nonzero):
            dic_key = int(rel_matrix[x_index][y_index].item())
            rel_elem = (x_index, y_index, dic_key)
            rel_list.append(rel_elem)
        return rel_list

    def get_triplets(self, ent_matrix, rel_matrix, pol_matrix, token2sents):
        ent_list = self.rel_matrix2list(ent_matrix)
        rel_list = self.rel_matrix2list(rel_matrix)
        pol_list = self.rel_matrix2list(pol_matrix)
        res, pair = self.decode_triplet(ent_list, rel_list, pol_list, token2sents)
        return res, pair
    
    def decode_triplet(self, ent_list, rel_list, pol_list, token2sents):
        # Entity dictionary, with structure (head: [(tail, relation type)])
        entity_elem_dic = defaultdict(list)
        entity2type = {}
        for entity in ent_list:
            if token2sents[entity[0]] != token2sents[entity[1]]: continue
            entity_elem_dic[entity[0]].append((entity[1], entity[2]))
            entity2type[entity[:2]] = entity[2]
        
        # Decoding polarity matrix
        pol_entity_elem = defaultdict(list)
        for h2h_pol in pol_list:
            pol_entity_elem[h2h_pol[0]].append((h2h_pol[1], h2h_pol[2]))

        # (boundary,boundary -> polarity) set
        b2b_relation_set = {}
        for rel in pol_list:
            b2b_relation_set[rel[:2]] = rel[-1]
        
        # tail2tail set
        t2t_relation_set = set()
        for rel in rel_list:
            if rel[2] == self.rel_dic['t2t']:
                t2t_relation_set.add(rel[:2])

        # head2head dictionary, with structure (head1: [(head2, relation type)])
        h2h_entity_elem = defaultdict(list)
        for h2h_rel in rel_list:
            # for each head-to-head relationship, mark its entity as 0
            if h2h_rel[2] != self.rel_dic['h2h']: continue
            h2h_entity_elem[h2h_rel[0]].append((h2h_rel[1], h2h_rel[2]))
        
        # for all head-to-head relations
        triplets = []
        for h1, values in h2h_entity_elem.items():
            if h1 not in entity_elem_dic: continue
            for h2, rel_tp in values:
                if h2 not in entity_elem_dic: continue
                for t1, ent1_tp in entity_elem_dic[h1]:
                    for t2, ent2_tp in entity_elem_dic[h2]:
                        if (t1, t2) not in t2t_relation_set: continue
                        triplets.append((h1, t1, h2, t2))

        # if there is a (0,0,0,0) in triplets, remove it
        if (0, 0, 0, 0) in triplets:
            triplets.remove((0, 0, 0, 0))
        
        triplet_set = set(triplets)
        ele2list = defaultdict(list)
        for line in triplets:
            e0, e1 = line[:2], line[2:]
            ele2list[e0].append(e1)
        
        tetrad = []
        for subj, obj_list in ele2list.items():
            for obj in obj_list:
                if obj not in ele2list: continue
                for third in ele2list[obj]:
                    if (*subj, *third) not in triplet_set: continue
                    tp0 = b2b_relation_set.get((subj[0], third[0]), -1)
                    tp1 = b2b_relation_set.get((subj[1], third[1]), -1)
                    if (tp0 == tp1 or tp0 == -1) and tp1 != -1:
                        tetrad.append((*subj, *obj, *third, tp1))
                    elif tp0 != -1 and tp1 == -1:
                        tetrad.append((*subj, *obj, *third, tp0))
                    else:
                        tetrad.append((*subj, *obj, *third, 1))
        
        pairs = {'ta': [], 'to': [], 'ao': []}
        for line in triplets:
            h1, t1, h2, t2 = line
            tp1 = entity2type[(h1, t1)]
            tp2 = entity2type[(h2, t2)]
            if tp1 == 1 and tp2 == 2:
                pairs['ta'].append(line)
            elif tp1 == 2 and tp2 == 3:
                pairs['ao'].append(line)
            elif tp1 == 1 and tp2 == 3:
                pairs['to'].append(line)
        return set(tetrad), pairs

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

class ScoreManager:
    def __init__(self) -> None:
        self.score = []
        self.line = []
    
    def add_instance(self, score, res):
        self.score.append(score)
        self.line.append(res)
    
    def get_best(self):
        best_id = np.argmax(self.score)
        res = self.line[best_id]
        return self.score[best_id], res

def update_config(config):
    lang = config.lang
    keys = ['json_path']
    for k in keys:
        config[k] = config[k] + '_' + lang
    keys = ['cls', 'sep', 'pad', 'unk', 'bert_path']
    for k in keys:
        config[k] = config['bert-' + config.lang][k]
    return config


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None and len(mask.shape) == 3:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
