import warnings
import os

import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import BertConfig, BertTokenizer
from transformers import BertForSequenceClassification, BertForMaskedLM
import copy
import argparse
import numpy as np

import jieba
import re
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
# 将虚词给滤除
chinese_filter_words = ['的','地','得','着','了','过','[UNK]']
filter_words = set(filter_words)


def get_sim_embed(embed_path, sim_path):
    id2word = {}
    word2id = {}

    with open(embed_path, 'r', encoding='utf-8') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in id2word:
                id2word[len(id2word)] = word
                word2id[word] = len(id2word) - 1

    cos_sim = np.load(sim_path)
    return cos_sim, word2id, id2word

USE_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

def get_data_cls(data_path):
    lines = open(data_path, 'r', encoding='utf-8').readlines()[0:]
    features = []
    for i, line in enumerate(lines):
        if i>2:
            break
        split = line.split("_!_")
        label = int(split[1])
        label -= 100
        seq = split[-2]

        features.append([seq, label])
    return features


class Feature(object):
    def __init__(self, seq_a, label):
        self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []


def chinese_tokenize(seq):
    keys = []
    words = []
    seg_keys = []
    segs = jieba.cut(seq)
    index = 0
    for i, seg in enumerate(segs):
        seg_keys.append([index, index+len(seg)])
        for j in range(len(seg)):
            words.append(seg[j])
            keys.append([index,index+1])
            index+=1
    return words, keys

def _get_masked(words):
    len_text = len(words)
    masked_words = []
    for i in range(len_text - 1):
        masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
    masked_words.append(words[:len_text-1]+['[UNK]'])
    # list of words
    return masked_words

def _get_deleted(words):
    len_text = len(words)
    deleted_words = []
    for i in range(len_text - 1):
        deleted_words.append(words[0:i] + words[i + 1:])
    deleted_words.append(words[:len_text-1])
    # list of words
    return deleted_words

def get_important_scores(words, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length, mode):
    if mode == 'mask':    
        masked_words = _get_masked(words)
        texts = [''.join(words) for words in masked_words]  # list of text of masked words
    elif mode == 'delete':
        deleted_words = _get_deleted(words)
        texts = [''.join(words) for words in deleted_words]  # list of text of deleted words
    all_input_ids = []
    all_masks = []
    all_segs = []
    for text in texts:
        inputs = tokenizer(text, None, max_length=max_length,truncation=True,
                 padding='max_length', return_token_type_ids=True)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = inputs['attention_mask']
        assert(len(input_ids)==max_length)
        assert(len(token_type_ids)==max_length)
        assert(len(attention_mask)==max_length)
        all_input_ids.append(input_ids)
        all_masks.append(attention_mask)
        all_segs.append(token_type_ids)
    seqs = torch.tensor(all_input_ids, dtype=torch.long)
    masks = torch.tensor(all_masks, dtype=torch.long)
    segs = torch.tensor(all_segs, dtype=torch.long)
    seqs = seqs.to('cuda')

    eval_data = TensorDataset(seqs)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    leave_1_probs = []
    for batch in eval_dataloader:
        masked_input, = batch
        bs = masked_input.size(0)

        leave_1_prob_batch = tgt_model(masked_input)[0]  # B num-label
        leave_1_probs.append(leave_1_prob_batch)
    leave_1_probs = torch.cat(leave_1_probs, dim=0)  # words, num-label
    leave_1_probs = torch.softmax(leave_1_probs, -1)  #
    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
    import_scores = (orig_prob
                     - leave_1_probs[:, orig_label]
                     +
                     (leave_1_probs_argmax != orig_label).float()
                     * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                     ).data.cpu().numpy()

    return import_scores

def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words
        
    elif sub_len == 1:
        for (i,j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k

    substitutes = substitutes[0:12, 0:4] # maximum BPE candidates

    # find all possible candidates 

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0] # N L vocab-size
    ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer.convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words

def insert_mask_based_index(index, mode, raw_text, mask_token):
    if mode == 0: # before index
        masked_text = raw_text[:index] + [mask_token] + raw_text[index:]
    if mode == 1: # after index
        if index + 1 < len(raw_text):
            masked_text = raw_text[:index+1] + [mask_token] + raw_text[index+1:]
        else:
            masked_text = raw_text[:index+1] + [mask_token]
    return masked_text

def merge_mask_based_index(index, raw_text, mask_token):
    return raw_text[:index] + [mask_token] + raw_text[index+2:] if index+2 < len(raw_text) else raw_text[:index] + [mask_token]

def attack(feature, tgt_model, mlm_model, tokenizer, k, batch_size, max_length=128, attack_name='BertAttack', attack_mode=None, cos_mat=None, w2i={}, i2w={}, use_bpe=1, threshold_pred_score=0.3):
    # MLM-process
    words, keys = chinese_tokenize(feature.seq)

    # original label
    inputs = tokenizer(feature.seq, None, max_length=max_length,
        padding='max_length', truncation=True, return_token_type_ids=True)
    input_ids, token_type_ids = torch.tensor(inputs["input_ids"]), torch.tensor(inputs["token_type_ids"])
    attention_mask = torch.tensor(inputs['attention_mask'])
    seq_len = input_ids.size(0)
    orig_probs = tgt_model(input_ids.unsqueeze(0).to('cuda'),
                           attention_mask.unsqueeze(0).to('cuda'),
                           token_type_ids.unsqueeze(0).to('cuda')
                           )[0].squeeze()
    orig_probs = torch.softmax(orig_probs, -1)
    orig_label = torch.argmax(orig_probs)
    current_prob = orig_probs.max()

    if orig_label != feature.label:
        feature.success = 3
        return feature

    words = ['[CLS]'] + words[:max_length - 2] + ['[SEP]']
    keys = ([[0,1]]) + keys + ([[keys[-1][1], keys[-1][1]+1]])
    
    if attack_name == 'BertAttack':
        important_scores = get_important_scores(words, tgt_model, current_prob, orig_label, orig_probs,
                                            tokenizer, batch_size, max_length, mode='mask')
    if attack_name == 'TextFooler' or attack_name == 'BAE':
        important_scores = get_important_scores(words, tgt_model, current_prob, orig_label, orig_probs,
                                            tokenizer, batch_size, max_length, mode='delete')
    feature.query += int(len(words))
    list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
    final_words = copy.deepcopy(words)

    # --------------------REPLACE----------------------
    replace_input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(words)])
    replace_word_predictions = mlm_model(replace_input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab
    replace_word_pred_scores_all, replace_word_predictions = torch.topk(replace_word_predictions, k, -1)  # seq-len k
    # print(word_predictions)
    replace_word_predictions = replace_word_predictions[1:len(words) - 1, :]
    # print(word_predictions)
    replace_word_pred_scores_all = replace_word_pred_scores_all[1:len(words) - 1, :]

    # -------------------INSERT---------------------
    insert_before_masked_texts = []
    insert_after_masked_texts = []
    add_index = [0 for _ in range(len(words))]
    # -------------------MERGE----------------------
    merge_masked_texts = []
    delete_index = [0 for _ in range(len(words))]
    for top_index in list_of_index:
        # top_index[0] 是下标，top_index[1]是score，数据来源是带有特殊标志的句子
        if feature.change > int(0.4 * (len(words))):
            feature.success = 1  # exceed
            return feature
        if attack_name == 'BAE':
            insert_masked_text = insert_mask_based_index(top_index[0], mode=0, raw_text=words, mask_token=tokenizer.mask_token)
            insert_before_masked_texts.append(insert_masked_text)
        if attack_name in ['BAE', 'CLARE']:
            insert_masked_text = insert_mask_based_index(top_index[0], mode=1, raw_text=words, mask_token=tokenizer.mask_token)
            insert_after_masked_texts.append(insert_masked_text)

            merge_masked_text = merge_mask_based_index(top_index[0], words, tokenizer.mask_token)
            merge_masked_texts.append(merge_masked_text)

        tgt_word = words[top_index[0]]
        if tgt_word in chinese_filter_words and attack_name=='BertAttack':
            continue
        if top_index[0] > max_length - 2 or top_index[0] == 0 or top_index[0] == len(words)-1:
            continue

        if attack_name == 'BertAttack' or (attack_name == 'BAE' and (attack_mode == 'R' or attack_mode == 'R+I') ):
            substitutes = replace_word_predictions[top_index[0]-1:top_index[0]]  # L, k
            word_pred_scores = replace_word_pred_scores_all[top_index[0]-1:top_index[0]]

            substitutes = get_substitues(substitutes, tokenizer, mlm_model, use_bpe, word_pred_scores, threshold_pred_score)
            most_gap = 0.0
            candidate = None
            print(tgt_word, substitutes)
            for substitute_ in substitutes:
                substitute = substitute_

                if substitute == tgt_word:
                    continue  # filter out original word
                if '##' in substitute:
                    continue  # filter out sub-word
                if substitute in chinese_filter_words:
                    continue
                if substitute in w2i and tgt_word in w2i:
                    if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4: #计算cos_sim作为条件之一
                        continue
                temp_replace = copy.deepcopy(final_words)
                temp_replace[top_index[0]+add_index[top_index[0]]] = substitute
                temp_text = ''.join(temp_replace)
                inputs = tokenizer(temp_text, None, max_length=max_length, truncation=True, 
                    padding='max_length')
                input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to('cuda')
                temp_prob = tgt_model(input_ids)[0].squeeze()
                feature.query += 1
                temp_prob = torch.softmax(temp_prob, -1)
                temp_label = torch.argmax(temp_prob)

                if temp_label != orig_label:
                    feature.change += 1
                    final_words[top_index[0]+add_index[top_index[0]]] = substitute
                    feature.changes.append([top_index[0]+add_index[top_index[0]], substitute, tgt_word])
                    feature.final_adverse = temp_text
                    feature.success = 4
                    return feature
                else:
                    label_prob = temp_prob[orig_label]
                    gap = current_prob - label_prob
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute
            if most_gap > 0:
                feature.change += 1
                feature.changes.append([top_index[0]+add_index[top_index[0]], candidate, tgt_word])
                current_prob = current_prob - most_gap
                final_words[top_index[0]+add_index[top_index[0]]] = candidate
            if attack_mode == 'R+I' and attack_name=='BAE':
                insert_index = top_index[0] + add_index[top_index[0]]
                insert_text = insert_mask_based_index(insert_index, mode=1, raw_text=final_words, mask_token=tokenizer.mask_token)
                final_words = insert_text
                insert_input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(insert_text)])
                insert_word_predictions = mlm_model(insert_input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab
                insert_word_pred_scores_all, insert_word_predictions = torch.topk(insert_word_predictions, k, -1)  # seq-len k

                substitutes = insert_word_predictions[insert_index+1:insert_index+2]  # L, k
                word_pred_scores = insert_word_pred_scores_all[insert_index+1:insert_index+2]
                substitutes = get_substitues(substitutes, tokenizer, mlm_model, use_bpe, word_pred_scores, threshold_pred_score)
                
                most_gap = 0.0
                candidate = None
                print(insert_text[insert_index], substitutes)
                for substitute_ in substitutes:
                    substitute = substitute_
                    if substitute in chinese_filter_words:
                        continue
                    if '##' in substitute:
                        continue  # filter out sub-word
                    temp_insert = copy.deepcopy(final_words)
                    temp_insert[insert_index+1] = substitute
                    # input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(temp_insert)).unsqueeze(0).to('cuda')
                    temp_text = ''.join(temp_insert)
                    inputs = tokenizer(temp_text, None, max_length=max_length, truncation=True, 
                    padding='max_length')
                    input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to('cuda')
                    temp_prob = tgt_model(input_ids)[0].squeeze()
                    temp_prob = torch.softmax(temp_prob, -1)
                    temp_label = torch.argmax(temp_prob)
                    if temp_label != orig_label:
                        add_index = [add_index[i]for i in range(top_index[0]+1)] + [add_index[i] + 1 for i in range(top_index[0]+1, len(words))]
                        feature.change += 1
                        final_words[insert_index+1] = substitute
                        feature.changes.append([insert_index+1, substitute, tokenizer.mask_token])
                        feature.final_adverse = ''.join(final_words)
                        feature.success = 4
                        return feature
                    else:
                        label_prob = temp_prob[orig_label]
                        gap = current_prob - label_prob
                        if gap > most_gap:
                            most_gap = gap
                            candidate = substitute

                if most_gap > 0:
                    feature.change += 1
                    feature.changes.append([insert_index+1, candidate, tokenizer.mask_token])
                    current_prob = current_prob - most_gap
                    final_words[insert_index+1] = candidate
                    add_index = [add_index[i]for i in range(top_index[0]+1)] + [add_index[i] + 1 for i in range(top_index[0]+1, len(words))]
                else:
                    final_words = final_words[:insert_index+1] + final_words[insert_index+2:]
    # if not (attack_name == 'TextFooler' or attack_name == 'BertAttack' or (attack_name == 'BAE' and attack_mode == 'R')):
    #     i = 0
    #     while i < len(insert_before_masked_texts):
    #         insert_before_inputs = tokenizer(insert_before_masked_texts[i:i+batch_size], 
    #                                         None, max_length=max_length, padding='max_length', truncation=True)
            
    #         i += batch_size
    feature.final_adverse = ''.join(final_words)
    feature.success = 2
    return feature


def evaluate(features):
    do_use = 0
    use = None
    sim_thres = 0
    # evaluate with USE
    acc = 0
    origin_success = 0
    total = 0
    total_q = 0
    total_change = 0
    total_word = 0
    for feat in features:
        if feat.success > 2:

            if do_use == 1:
                sim = float(use.semantic_sim([feat.seq], [feat.final_adverse]))
                if sim < sim_thres:
                    continue
            
            acc += 1
            total_q += feat.query
            total_change += feat.change
            total_word += len(feat.seq.split(' '))

            if feat.success == 3:
                origin_success += 1

        total += 1

    suc = float(acc / total)

    query = float(total_q / acc)
    change_rate = float(total_change / total_word)

    origin_acc = 1 - origin_success / total
    after_atk = 1 - suc

    print('acc/aft-atk-acc {:.6f}/ {:.6f}, query-num {:.4f}, change-rate {:.4f}'.format(origin_acc, after_atk, query, change_rate))


def dump_features(features, output):
    outputs = []

    for feature in features:
        outputs.append({'label': feature.label,
                        'success': feature.success,
                        'change': feature.change,
                        'num_word': len(feature.seq.split(' ')),
                        'query': feature.query,
                        'changes': feature.changes,
                        'seq_a': feature.seq,
                        'adv': feature.final_adverse,
                        })
    output_json = output
    json.dump(outputs, open(output_json, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print('finished dump')


def run_attack():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default = './data/eval.txt', help="./data/xxx")
    parser.add_argument("--mlm_path", type=str, default = './bert-base-chinese', help="xxx mlm")
    parser.add_argument("--tgt_path", type=str, default = './finetuned-model', help="xxx classifier")

    parser.add_argument("--output_dir", type=str, default = './data/output.json', help="train file")
    parser.add_argument("--use_sim_mat", type=int, help='whether use cosine_similarity to filter out atonyms')
    parser.add_argument("--start", type=int, help="start step, for multi-thread process")
    parser.add_argument("--end", type=int, help="end step, for multi-thread process")
    parser.add_argument("--num_label", type=int, default=17)
    parser.add_argument("--use_bpe", type=int, )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--threshold_pred_score", type=float, default=0.3)

    parser.add_argument("--attack_name", type=str, default='BertAttack')
    parser.add_argument("--attack_mode", type=str, choices=['R', 'I', 'R/I', 'R+I'], default=None)
    parser.add_argument("--use_threshold", type=float, default=0.7)
    
    attack_list = ['BertAttack', 'TextFooler', 'BAE', 'CLARE']
    
    args = parser.parse_args()
    assert(args.attack_name in attack_list)
    data_path = str(args.data_path)
    mlm_path = str(args.mlm_path)
    tgt_path = str(args.tgt_path)
    output_dir = str(args.output_dir)
    num_label = args.num_label
    use_bpe = args.use_bpe
    k = args.k
    start = args.start
    end = args.end
    threshold_pred_score = args.threshold_pred_score

    print('start process')

    tokenizer_mlm = BertTokenizer.from_pretrained(mlm_path, do_lower_case=True)
    tokenizer_tgt = BertTokenizer.from_pretrained(tgt_path, do_lower_case=True)

    config_atk = BertConfig.from_pretrained(mlm_path)
    mlm_model = BertForMaskedLM.from_pretrained(mlm_path, config=config_atk)
    mlm_model.to('cuda')

    config_tgt = BertConfig.from_pretrained(tgt_path, num_labels=num_label)
    tgt_model = BertForSequenceClassification.from_pretrained(tgt_path, config=config_tgt)
    tgt_model.to('cuda')
    features = get_data_cls(data_path)
    print('loading sim-embed')
    
    if args.use_sim_mat == 1:
        cos_mat, w2i, i2w = get_sim_embed('data_defense/counter-fitted-vectors.txt', 'data_defense/cos_sim_counter_fitting.npy')
    else:        
        cos_mat, w2i, i2w = None, {}, {}

    print('finish get-sim-embed')
    features_output = []

    with torch.no_grad():
        for index, feature in enumerate(features[start:end]):
            seq_a, label = feature
            feat = Feature(seq_a, label)
            print('\r number {:d} '.format(index) + tgt_path, end='')
            # print(feat.seq[:100], feat.label)
            feat = attack(feat, tgt_model, mlm_model, tokenizer_tgt, k, batch_size=16, max_length=64, 
                          attack_name = args.attack_name, attack_mode=args.attack_mode,
                          cos_mat=cos_mat, w2i=w2i, i2w=i2w, use_bpe=use_bpe,threshold_pred_score=threshold_pred_score)

            # print(feat.changes, feat.change, feat.query, feat.success)
            if feat.success > 2:
                print('success', end='')
            else:
                print('failed', end='')
            features_output.append(feat)

    evaluate(features_output)

    dump_features(features_output, output_dir)


if __name__ == '__main__':
    run_attack()