#!/usr/bin/env python
# _*_ coding:utf-8 _*_


"""
@filename: prepare_data.py
@dateTime: 2022-03-13 15:36:30
"""

import os
import re
import json
import string
import random
import unicodedata
import argparse
from urllib.parse import quote

import yaml
import jieba
import spacy
import numpy as np
import requests as rq

from attrdict import AttrDict
from loguru import logger
from tqdm import tqdm


class Prepare(object):
    def __init__(self, lang):
        config = AttrDict(yaml.load(open('src/config.yaml', 'r', encoding='utf-8').read(), Loader=yaml.FullLoader))
        config.lang = lang

        keys = ['annotation_dir', 'json_path', 'preprocessed_dir', 'target_dir']
        for k in keys:
            config[k] = config[k] + '_' + config['lang']

        self.config = config

        self.set_seed(config.seed)
        self.opinion_dict = {'Opinion_pos': 'pos', 'Opinion_neg': 'neg', 'Opinion_mid': 'neu', 'Opinion_ambiguous': 'amb', 'Opinion_doubt': 'doubt', 'Opinion1_pos': 'pos', "Opinion1_neg": 'neg'}
        self.spacy = spacy.load('en_core_web_sm')
    
    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
    
    def tohalfwitth(self, token):
        return unicodedata.normalize('NFKC', token)
    
    def parse_dialog(self, document):
        doc_id, content, reply_speark, triplets, targets, aspects, opinions, dependencies = document
        s2c = {}
        for i, line in enumerate(content):
            for j, w in enumerate(line):
                s2c[(i, j)] = len(s2c)
        
        # Convert utterance-level token index into dialogue-level index
        global_dependencies = [[(s2c[(i, u)], w if w == -1 else s2c[(i, w)], x, y, z) for u, w, x, y, z in line] for i, line in enumerate(dependencies)]
        global_dependencies = [w for line in global_dependencies for w in line]

        # Convert speaker and replying infos.
        replies = [int(w[0]) - 1 for w in reply_speark]
        speakers = [ord(w[2]) - ord('A') for w in reply_speark]

        # The sentence number of each word in the entire document
        sentence_ids = []
        for i, line in enumerate(content):
            sentence_ids += [i] * len(line)
        triples = []

        # Extract the triplet of the whole dialogue.
        for target, aspect, opinion in triplets:
            line = []
            if target is not None:
                line += target[1:3]
            else:
                line += [-1, -1]
            if aspect is not None:
                line += aspect[1:3]
            else:
                line += [-1, -1]
            if opinion is not None:
                # print(type(line), type(opinion[1:3]))
                line = line + list(opinion[1:3]) + [self.opinion_dict[opinion[0]]]
            else:
                line += [-1, -1, -1]
            line.append('' if target is None else target[-1])
            line.append('' if aspect is None else aspect[-1])
            line.append('' if opinion is None else opinion[-1])
            triples.append(tuple(line))
        
        # Deduplicate the triplet and keep the original orders.
        triples = sorted(set(triples),key=triples.index)

        # Extraction Opinions
        opinions = [(*w[1:4], self.opinion_dict[w[0]]) for w in opinions]
        res = {'doc_id': doc_id, 'sentences': content, 'replies': replies, 'speakers': speakers, 'sentence_ids': sentence_ids, 'triplets': triples, 'targets': targets, 'aspects': aspects, 'opinions': opinions, 'dependency': global_dependencies, 'local_dependency': dependencies}
        return res
    
    def parse_subdialog(self, dialog):

        # Process the dialogue threads and convert the global index to the index in the thread
        doc_id, replies, speakers, sentence_ids, triplets, targets, aspects, opinions, local_dependency, contents = [dialog[w] for w in '\
        doc_id, replies, speakers, sentence_ids, triplets, targets, aspects, opinions, local_dependency, sentences'.strip().split(', ')]
        sub_dialogs = []

        # The IDs of the sentences in the three threads
        new_sentence_collections = [[0], [0], [0]]

        # The current thread number
        sidx = -1
        for i, idx in enumerate(replies):
            if i == 0: continue

            # Once the first utterance is replied to, a new thread begins
            if idx == 0:
                sidx += 1
            # Add the current sentence number to the thread 
            new_sentence_collections[sidx].append(i)

        # Remove empty threads 
        new_sentence_collections = [w for w in new_sentence_collections if len(w) > 1]
        
        # Convert the global index of each character in the document to (thread number, index in the thread)
        char2char = {-1: (-2, -1), -2: (-2, -1)}

        # Convert the first utterance 
        for i in range(len(contents[0])):
            char2char[i] = (-1, i)

        total_lens = len(contents[0])
        # Then convert each thread
        for c_id, sentence_collections in enumerate(new_sentence_collections):
            # The current thread length
            cur_lens = len(contents[0])
            # Traverse the utterance in the current thread
            for j, k in enumerate(sentence_collections):
                if k == 0: continue # Skip the first sentence
                # Traverse the j-th sentence in the current thread
                for m in range(len(contents[k])):
                    # Global index -> (thread number, index in thread)
                    char2char[m + total_lens] = (c_id, m + cur_lens)
                # After traversing the sentence, the global and current thread lengths are increased accordingly
                total_lens, cur_lens = total_lens + len(contents[k]), cur_lens + len(contents[k])

        # After obtaining the char2char conversion relationship, begin the index conversion operation

        # Traverse each thread
        for i, line in enumerate(new_sentence_collections):

            # Cur Document ID: document ID + thread ID
            cur_doc_id = '{}_{}'.format(doc_id, i)

            # The sentences in the current thread
            cur_sentences = [contents[w] for w in line]
            
            # The speakers in the current thread
            cur_speakers = [speakers[w] for w in line]

            # The sentence IDs of the characters in the current thread
            cur_sentence_ids = []
            for j, sent in enumerate(cur_sentences):
                cur_sentence_ids += [j] * len(sent)
            
            # The dependency analysis results for the sentences in the current thread
            cur_dependency = [local_dependency[w] for w in line]

            # s2c: (sentence ID, index of character in sentence) -> global index
            s2c = {}
            for i, sent in enumerate(cur_sentences):
                for j, w in enumerate(sent):
                    s2c[(i, j)] = len(s2c)
            
            # After obtaining s2c, convert the index of the sentence-level dependency analysis to the index in the entire thread
            cur_dependency = [[(s2c[(i, u)], w if w == -1 else s2c[(i, w)], x, y, z) for u, w, x, y, z in line] for i, line in enumerate(cur_dependency)]
            cur_dependency = [w for line in cur_dependency for w in line]

            # After obtaining s2c, convert the index of the sentence-level dependency analysis to the index in the entire thread
            cur_line = {'doc_id': cur_doc_id, 'sentences': cur_sentences, 'speakers': cur_speakers, 'triplets': [], 'targets': [], 'aspects': [], 'opinions': [], 'dependency': cur_dependency}
            sub_dialogs.append(cur_line)

        # Implement the distribution of entities (Target, Aspect, Object) from the global scope to each branch.
        def divide(objects, mode='targets'):
            for line in objects:
                # Obtain the start and end indexes and text of the entity
                start, end, text = line[:3]

                # If the entity is an opinion, there is also a polarity
                if mode == 'opinions':
                    polarity = line[-1]

                # Convert the global indexes to thread numbers and indexes in the thread
                c_id0, nstart = char2char[start]
                c_id1, nend = char2char[end - 1]

                # Determine that the start and end words are in the same thread
                assert c_id0 == c_id1

                if c_id0 == -1:
                    # If the thread number is -1, it means that it is the first sentence, so the element belongs to three threades at the same time
                    # Traverse the three dialogue threades
                    for k in range(len(sub_dialogs)):
                        # Determine that the text and index correspond
                        if not ''.join([w for line in sub_dialogs[k]['sentences'] for w in line][nstart:nend+1]) == text.replace(' ', ''):
                            print([w for line in sub_dialogs[k]['sentences'] for w in line][nstart:nend+1], text.replace(' ', ''))
                            print(sub_dialogs[k]['doc_id'])
                            xx = 1
                        assert ''.join([w for line in sub_dialogs[k]['sentences'] for w in line][nstart:nend+1]) == text.replace(' ', '')
                        if mode == 'opinions':
                            # If it is an opinion, add the polarity
                            sub_dialogs[k][mode].append((nstart, nend + 1, text, polarity))
                        else:
                            # Otherwise, add the index in the thread directly to the corresponding entity list
                            sub_dialogs[k][mode].append((nstart, nend + 1, text))
                
                else:
                    # Otherwise, it belongs to a certain thread, so add it to the corresponding thread
                    if not ''.join([w for line in sub_dialogs[c_id0]['sentences'] for w in line][nstart:nend+1]) == text.replace(' ', ''):
                        print([w for line in sub_dialogs[c_id0]['sentences'] for w in line][nstart:nend+1], text.replace(' ', ''))
                        print(sub_dialogs[c_id0]['doc_id'])
                        x = 1
                    assert ''.join([w for line in sub_dialogs[c_id0]['sentences'] for w in line][nstart:nend+1]) == text.replace(' ', '')
                    # assert ''.join(sub_dialogs[c_id0]['sentences'])[nstart:nend+1] == text
                    if mode == 'opinions':
                        sub_dialogs[c_id0][mode].append((nstart, nend + 1, text, polarity))
                    else:
                        sub_dialogs[c_id0][mode].append((nstart, nend + 1, text))
        
        # Add three entities to the corresponding thread list
        divide(targets, 'targets')
        divide(aspects, 'aspects')
        divide(opinions, 'opinions')

        # Traverse the triplets and assign them to the three threades
        for t_s, t_e, a_s, a_e, o_s, o_e, polarity, t_t, a_t, o_t in triplets:

            # Subscript conversion, global subscript -> (thread number, subscript in thread)
            nts, nas, nos = [char2char[w] for w in [t_s, a_s, o_s]]
            nte, nae, noe = [char2char[w - 1] for w in [t_e, a_e, o_e]]

            # Make sure each entity is in the same thread
            assert nts[0] == nte[0] and nas[0] == nae[0] and nos[0] == noe[0]

            # Get the thread set of the entities in the triplet
            effective_ids = [(w[0], z) for w, z in zip([nts, nas, nos], [t_s, a_s, o_s]) if w[0] != - 2]

            if all(w[0] == -1 for w in effective_ids):
                # If all entities in the triplet are present in the first sentence, then distribute them to three branches.
                nts, nas, nos = [w[1] for w in [nts, nas, nos]]
                nte, nae, noe = [w[1] + 1 if w[1] != -1 else w[1] for w in [nte, nae, noe]]
                cur_line = [nts, nte, nas, nae, nos, noe, polarity, t_t, a_t, o_t]
                for k in range(len(sub_dialogs)):
                    sub_dialogs[k]['triplets'].append(cur_line)
            else:
                # Otherwise, distribute them to a unique branch, ignoring the first sentence.
                effective_ids = [w for w in effective_ids if w[0] != -1]

                # Ignore if there is a cross-branch occurrence.
                if not all(w[0] == effective_ids[0][0] for w in effective_ids):
                    ids = sorted(list(set([sentence_ids[w[1]] for w in effective_ids])))
                    continue

                # assert all(w == effective_ids[0] for w in effective_ids)
                nts, nas, nos = [w[1] for w in [nts, nas, nos]]
                nte, nae, noe = [w[1] + 1 if w[1] != -1 else w[1] for w in [nte, nae, noe]]
                cur_line = [nts, nte, nas, nae, nos, noe, polarity, t_t, a_t, o_t]
                effective_id = effective_ids[0][0]
                sub_dialogs[effective_id]['triplets'].append(cur_line)
                cur_doc = [w for line in sub_dialogs[effective_id]['sentences'] for w in line]

                if not t_t.replace(' ', '') == ''.join(cur_doc[nts:nte]):
                    print(t_t.replace(' ', ''), ''.join(cur_doc[nts:nte]))
                assert t_t.replace(' ', '') == ''.join(cur_doc[nts:nte])

                if not a_t.replace(' ', '') == ''.join(cur_doc[nas:nae]):
                    print(a_t.replace(' ', ''), ''.join(cur_doc[nas:nae]))
                assert a_t.replace(' ', '') == ''.join(cur_doc[nas:nae])

                if not o_t.replace(' ', '') == ''.join(cur_doc[nos:noe]):
                    print(o_t.replace(' ', ''), ''.join(cur_doc[nos:noe]))
                assert o_t.replace(' ', '') == ''.join(cur_doc[nos:noe])

        return sub_dialogs

    def read_files(self, file_list):
        # Read all files
        def sub_replace(text):
            # Replace special characters
            text = text.replace('…', '.').replace('“', '"').replace('”', '"').replace('4⃣️', '4..').replace('￣', '-').replace('℃', '度')
            return text 
        
        def get_half(filename):
            # Read text and convert to half-width encoding
            half_a = []
            lines = open(filename, 'r', encoding='utf-8').read().splitlines()
            for line in lines:
                half_line = self.tohalfwitth(sub_replace(line))
                assert len(line) == len(half_line)
                half_a.append(half_line)
            return half_a

        res = []
        for file_index, ann_file in enumerate(tqdm(file_list)):
            txt_file = ann_file.replace('.ann', '.txt')

            # Get half-width encoded text from ann and txt files
            ann_content = get_half(ann_file)
            txt_content = get_half(txt_file)

            # If there are no entities (i.e., the text has not been annotated), skip
            if all(not w.startswith('T') for w in ann_content): continue

            char2char, content, short_lens, long_lens, reply_speaker = {}, [], 0, 0, []

            # Iterate over text
            for idx, line in enumerate(txt_content):
                # Extract conversation utterance text
                content.append(line[4:])

                # Extract speaker and reply relationships
                reply_speaker.append(line[:3])

                # Get string mapping, short_len: index of new text (without reply number and speaker), long_len: index of original text
                for j in range(len(line[4:])):
                    char2char[j + long_lens + 4] = short_lens + j 
                char2char[long_lens + len(line[4:]) + 4] = short_lens + len(line[4:])
                long_lens, short_lens = long_lens + len(line) + 1, short_lens + len(line) - 4
            
            new_content, word2dict = self.parse_content(content)

            # Get dependencies parsing
            dependencies = self.parse_document(new_content)

            # Parse ann file, annotation is a triplet, entities are entity elements (target, aspect, and opinion)
            annotation, entities = self.parse_ann(ann_content, char2char, ann_file, word2dict)

            # Separate target, aspect, and opinion
            targets = list(set([entities[w][1:] for w in entities if entities[w][0].startswith('object')]))
            aspects = list(set([entities[w][1:] for w in entities if entities[w][0].startswith('Aspect')]))
            opinions = list(set([entities[w] for w in entities if entities[w][0].startswith('Opinion')]))

            # Remove duplicate targets
            targets = self.remove_duplicate(annotation, targets)

            # Summarize results
            res.append((ann_file.split('/')[-1][:-4], new_content, reply_speaker, annotation, targets, aspects, opinions, dependencies))
        return res
    
    def parse_content(self, content):
        punctation = string.punctuation + '。'
        new_content = []
        if self.config.lang == 'zh':
            res, char2dict = [], {}
            char_index, word_index = 0, 0
            prev_length, word_length = 0, 0
            
            for line in content:
                cut_line = list(jieba.cut(line))
                new_cut = []
                for word in cut_line:
                    if word == ' ': continue
                    if len(word) == 1:
                        new_cut.append(word)
                    elif len(re.findall(r'[0-9a-zA-Z]', word)) > 0 or any(w in word for w in string.punctuation):

                        pattern0 = '[^0-9a-zA-Z'+string.punctuation + ']+|'
                        pattern1 = '[0-9]+|[a-zA-Z]+|[' + string.punctuation + ']+'

                        pattern = re.compile(pattern0+pattern1)
                        if word in ['bugmiui']:
                            new_cut.append(word[:3])
                            new_cut.append(word[3:])
                        elif word in ['MIUIbug']:
                            new_cut.append(word[:-3])
                            new_cut.append(word[-3:])
                        else:
                            lst = re.findall(pattern, word)
                            if not len(''.join(lst)) == len(word):
                                xx = 1
                            assert len(''.join(lst)) == len(word)
                            new_cut += lst
                    else:
                        new_cut += list(word)
                assert all(w != ' ' for w in word)
                new_content.append(new_cut)
                i, j = 0, 0
                while i < len(line) and j < len(new_cut):
                    if line[i] == ' ':
                        i += 1
                        continue
                    if line[i] == new_cut[j]:
                        char2dict[prev_length + i] = word_length + j
                        i += 1
                        j += 1
                    else:
                        lens = len(new_cut[j])
                        if lens <= 1:
                            xx = 1
                        assert lens > 1
                        assert line[i:i+lens] == new_cut[j]
                        for k in range(lens):
                            char2dict[prev_length + i] = word_length + j
                            i += 1
                        j += 1
                assert i == len(line) and j == len(new_cut)
                prev_length += len(line)
                word_length += len(new_cut)
        else:
            res, char2dict = [], {}
            prev_length, word_length = 0, 0
            
            for line in content:
                if '≥' in line:
                    xx = 1
                cut_line = [w.text for w in self.spacy(line)]
                # cut_line = line.split()
                # assert all(w != ' ' for w in word)
                new_content.append(cut_line)
                i, j = 0, 0
                while i < len(line) and j < len(cut_line):
                    if line[i] == ' ':
                        i += 1
                        continue
                    if line[i] == cut_line[j]:
                        char2dict[prev_length + i] = word_length + j
                        i += 1
                        j += 1
                    else:
                        lens = len(cut_line[j])
                        if lens <= 1:
                            xx = 1
                        assert lens > 1
                        if not line[i:i+lens] == cut_line[j]:
                            print('-{}-{}='.format(line[i:i+lens], cut_line[j]))
                            xx = 1
                        assert line[i:i+lens] == cut_line[j]
                        for k in range(lens):
                            char2dict[prev_length + i] = word_length + j
                            i += 1
                        j += 1
                if not i == len(line) or not j == len(cut_line):
                    print(len(line))
                    xx = 1
                assert i == len(line) and j == len(cut_line)
                prev_length += len(line)
                word_length += len(cut_line)
        
        return new_content, char2dict
     
    def parse_ann(self, ann_content, char2char, ann_file, char2dict):
        # Decode ann file and extract entities (Target, Aspect, Object) and triples
        annotation = []
        ann_content = sorted(ann_content, key=lambda x:x[0], reverse=True)
        entities = {}
        for line in ann_content:
            if line.startswith('T'):
                try:
                    tid, (tp, start, end), text = line.split('\t')[0], line.split('\t')[1].split(), line.split('\t')[2]
                except:
                    tid, (tp, start, end), text = line.split('\t')[0], (line.split('\t')[1].split()[0], *line.split('\t')[1].split(';')[-1].split()), line.split('\t')[2].split()[-1]
                start, end = int(start), int(end)
                nstart, nend = char2char[start], char2char[end]
                if not nstart in char2dict:
                    xx = 1
                nstart, nend = char2dict[nstart], char2dict[nend - 1] + 1

                newline = (tp, nstart, nend, text)
                entities[tid] = newline 
            elif line.startswith('E'):
                # E1  object1:T1 Aspect1-Arg:T4 Opinion1-Arg:T39
                eid, line_args = line.split('\t')[0], line.split('\t')[1].split()
                obj_id, asp_id, opi_id = None, None, None
                for w in line_args:
                    if w.startswith('obj'):
                        assert obj_id is None
                        obj_id = w.split(':')[1]
                    elif w.startswith('Asp'):
                        if not asp_id is None:
                            logger.warning((ann_file, asp_id, 'Duplicate'))
                            continue
                        assert asp_id is None
                        asp_id = w.split(':')[1]
                    elif w.startswith('Opi'):
                        if not opi_id is None:
                            logger.warning((ann_file, opi_id, 'Duplicate'))
                            continue
                        assert opi_id is None
                        opi_id = w.split(':')[1]
                try:
                    trigger, aspect, opinion = [entities[w] if w is not None else None for w in [obj_id, asp_id, opi_id]]
                except:
                    logger.info(ann_file, [obj_id, asp_id, opi_id], 'wrong')
                    exit(0)
                annotation.append((trigger, aspect, opinion))
            elif line.startswith('R'):
                rid, line_args, _ = line.split('\t')
                rname, t0, t1 = [w if i == 0 else w.split(':')[1] for i, w in enumerate(line_args.split())]
                trigger, aspect, opinion = [entities[w] if w is not None else None for w in [None, t0, t1]]
                annotation.append((trigger, aspect, opinion))
        return annotation, entities
    
    def remove_duplicate(self, annotation, targets):
        # In the annotation results, there are two incorrect targets: (0, 3, 18p, target) and (1, 3, 8p, target). The latter needs to be removed.
        res, r = [], set()
        for tgt, asp, opi in annotation:
            if tgt is None: continue
            start, end = tgt[1:3]
            r.add((start, end))
        
        targets = sorted(targets, key=lambda x:x[0] - x[1])

        mark = [0] * 4096
        for i, (start, end, txt) in enumerate(targets):
            if (start, end) in r:
                res.append((start, end, txt))
                for k in range(start, end):
                    mark[k] =  1

        for i, (start, end, txt) in enumerate(targets):
            if any(mark[w] == 1 for w in range(start, end)): continue
            res.append((start, end, txt))
            for k in range(start, end):
                mark[k] =  1
        res = sorted(res, key=lambda x:x[0])
        return res
    
    def parse_document(self, document):
        # dependency parsing, char-level subscripts 
        # no dependency parsing
        if True:
        # if not self.config.use_dep: 
            res = []
            for line in document:
                # line = line.replace(' ', '')
                line_res = []
                for i, w in enumerate(line):
                    if i == 0:
                        line_res.append((i, - 1, 'ROOT', line[i], 'R'))
                    else:
                        line_res.append((i, - 1, 'ROOT', line[i], 'R'))
                        # line_res.append((i, 0, 'dep', line[i], line[0]))
                res.append(line_res)
            return res
            # return [[(i, i - 1, 'ROOT' if i == 0 else 'root', line[i], line[i-1]) for i, w in enumerate(line)] for line in document]
        res = []
        url = 'http://{}/'.format(self.config['ip_port'])  + '?properties={"annotators":"tokenize, ssplit,pos, lemma, depparse","outputFormat":"json", "pipelineLanguage":"zh"}'
        for sentence in document:
            response = rq.post(url, data=quote(sentence.encode('utf-8')))
            response = json.loads(response.text)['sentences']

            word_dep = []
            father_child = set()

            dependency = [w['basicDependencies'] for w in response]
            dependencies = self.get_dependency(sentence, dependency)

            for line in dependencies:
                word_dep.append(line[:5])
                father_child.add((line[-1][0], line[-2][0]))

            word_dep = sorted(word_dep, key=lambda x:x[0])
            word_dep = [w for w in word_dep]

            char_dep = []
            word2char = {-1: -1}
            for word_id, (cur_idx, root_idx, dep_name, cur_word, root_word) in enumerate(word_dep):
                char_dep.append((len(char_dep), root_idx, dep_name, cur_word[0], root_word[0]))
                word2char[word_id] = len(char_dep) - 1
                for j in range(1, len(cur_word)):
                    char_dep.append((len(char_dep), cur_idx, 'consecutive', cur_word[j], cur_word[0]))

            char_dep = [(c_id, word2char[r_id], dep, c_w, r_w) for c_id, r_id, dep, c_w, r_w in char_dep]
            res.append(char_dep)

            # Verification
            for c_id, (c_id, f_id, dep, c_w, r_w) in enumerate(char_dep):
                if f_id == -1: 
                    assert dep == 'ROOT' or dep == 'Empty'
                    continue
                if dep != 'consecutive':
                    father_token = sentence[f_id]
                    cur_token = sentence[c_id]
                    assert father_token == r_w
                    assert cur_token == c_w
                    assert (father_token, cur_token) in father_child
        return res
    
    def get_dependency(self, sentence, dependencies):
        res = []
        # (start_idx, end_idx, dep_type)
        # (3, 0, ROOT)

        for dependency in dependencies:
            cur_length = len(res)
            for line in dependency:
                start_idx = line['governor'] - 1 + cur_length
                end_idx = line['dependent'] - 1 + cur_length
                dep_type = line['dep']
                if dep_type == 'ROOT': start_idx = -1
                res.append((end_idx, start_idx, dep_type, line['dependentGloss'], line['governorGloss']))
        res = sorted(res, key=lambda x:x[0])
        new_dep = []
        i, j, empty_num = 0, 0, 0
        token2token = {-1: -1}
        while i < len(sentence) and j < len(res):
            if sentence[i] == ' ':
                cur_line = (j + empty_num, -1, 'Empty', ' ', 'Empty')
                new_dep.append(cur_line)
                empty_num += 1
                i += 1
            else:
                token2token[j] = j + empty_num
                lens = len(res[j][-2])
                i += lens
                j += 1

        for e, s, d, dg, gg in res:
            cur_line =(token2token[e], token2token[s], d, dg, gg)
            new_dep.append(cur_line)
        return new_dep

    def get_data(self):
        files = os.listdir(self.config.annotation_dir)
        files = filter(lambda x:x.endswith('.ann'), files)
        files = [w for w in files]
        files = sorted(files)
        files = [os.path.join(self.config.annotation_dir, w) for w in files]
        return files
    
    def split_data(self, dataset):
        dataset = [w for w in dataset if len(w[3]) > 0]
        train_data = [dataset[i] for i in range(len(dataset)) if i % 10 > 1]
        valid_data = [dataset[i] for i in range(len(dataset)) if i % 10 == 1]
        test_data = [dataset[i] for i in range(len(dataset)) if i % 10 == 0]
        return train_data, valid_data, test_data
    
    def post_parse(self, dialogue):
        assert all(w != ' ' for line in dialogue['sentences'] for w in line)
        dialogue['sentences'] = [' '.join(w) for w in dialogue['sentences']]
        res_sub = []

        # for sub in dialogue['sub_dialogs']:
        #     assert all(w != ' ' for line in sub['sentences'] for w in line)
        #     sub['sentences'] = [' '.join(w) for w in sub['sentences']]
        #     res_sub.append(sub)

        # dialogue['sub_dialogs'] = res_sub
        return dialogue
    
    def forward(self):
        # Get all ann files
        files = self.get_data()

        # Read annotation data, including utterances, speakers, replying relation, and (Target, Aspect, Opinion, Sentiment) quadruples.
        res = self.read_files(files)

        # Split the dataset into train, valid and test as the rate of 8:1:1. The files whose index ends with 0/1 will be grouped into test/valid set, and the other files belong to train set.
        modes = 'train valid test'
        split_data = self.split_data(res)

        for i, w in enumerate(modes.split()):
            data = split_data[i]

            # Convert linear dialogues into tree-like dialogue, including the whole dialogue tree and sub-dialogue thread.
            new_data = []
            for line in tqdm(data):

                # Get the whole dialogue tree
                dialog = self.parse_dialog(line)

                # Get the sub-dialogue tree
                del dialog['dependency']
                del dialog['local_dependency']
                # sub_dialog = self.parse_subdialog(dialog)
                # dialog['sub_dialogs'] = sub_dialog
                # del dialog['sentence_ids']

                dialog = self.post_parse(dialog)

                new_data.append(dialog)

            # Write the tree-like dialogue into JONS format file. 
            res_path = os.path.join(self.config.json_path, '{}.json'.format(w))
            logger.info("Save {:5} file to {}...".format(w, res_path))

            # This line will generate a more readable JSON file but occupy too much space.
            # js = json.dumps(new_data, indent=4, separators=(',', ':'), ensure_ascii=False)

            # This line will generate a compact JSON file whose space is relative small.
            js = json.dumps(new_data, ensure_ascii=False)

            if not os.path.exists(self.config.json_path):
                os.makedirs(self.config.json_path)

            with open(res_path, 'w', encoding='utf-8') as f:
                f.write(js)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', default='zh')
    args = parser.parse_args()
    template = Prepare(lang=args.lang)
    template.forward()