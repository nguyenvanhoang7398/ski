import pickle
import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import (OpenAIGPTTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer, AutoTokenizer)
from typing import Optional
import os
import numpy
from utils.io import *


class MultiGPUSparseAdjDataBatchGenerator(object):
    def __init__(self, args, mode, device0, device1, batch_size, indexes, qids, labels,
                 tensors0=[], lists0=[], tensors1=[], lists1=[], adj_data=None):
        self.args = args
        self.mode = mode
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.labels = labels
        self.tensors0 = tensors0
        self.lists0 = lists0
        self.tensors1 = tensors1
        self.lists1 = lists1
        # self.adj_empty = adj_empty.to(self.device1)
        self.adj_data = adj_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)
        if self.mode=='train' and self.args.drop_partial_batch:
            print ('dropping partial batch')
            n = (n//bs) *bs
        elif self.mode=='train' and self.args.fill_partial_batch:
            print ('filling partial batch')
            remain = n % bs
            if remain > 0:
                extra = np.random.choice(self.indexes[:-remain], size=(bs-remain), replace=False)
                self.indexes = torch.cat([self.indexes, torch.tensor(extra)])
                n = self.indexes.size(0)
                assert n % bs == 0

        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_labels = self._to_device(self.labels[batch_indexes], self.device0)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device0) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]
            batch_lists0 = [self._to_device([x[i] for i in batch_indexes], self.device0) for x in self.lists0]
            batch_lists1 = [self._to_device([x[i] for i in batch_indexes], self.device1) for x in self.lists1]

            edge_index_all, edge_type_all = self.adj_data
            #edge_index_all: nested list of shape (n_samples, num_choice), where each entry is tensor[2, E]
            #edge_type_all:  nested list of shape (n_samples, num_choice), where each entry is tensor[E, ]
            edge_index = self._to_device([edge_index_all[i] for i in batch_indexes], self.device1)
            edge_type  = self._to_device([edge_type_all[i] for i in batch_indexes], self.device1)

            # batch_tensors0 = all_input_ids, all_attention_mask, all_token_type_ids, all_ent_mask,
            #                  all_ent_ner, all_ent_pos, all_ent_distance, all_structure_mask, all_label_mask
            # batch_tensors1 = concept_ids, node_type_ids, node_scores, adj_lengths

            yield tuple([batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1,
                         edge_index, edge_type])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)


class SkiExample(object):

    def __init__(self, guid, title, vertex_set, sents, labels=None):
        self.guid = guid
        self.title = title
        self.vertex_set = vertex_set
        self.sents = sents
        self.labels = labels


class SkiInputFeatures(object):
    def __init__(self, example_id, docred_input_features):
        self.example_id = example_id
        self.docred_input_features = docred_input_features


class DocREDInputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, ent_mask, ent_ner, ent_pos, ent_distance,
                 structure_mask, label=None, label_mask=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.ent_mask = ent_mask
        self.ent_ner = ent_ner
        self.ent_pos = ent_pos
        self.ent_distance = ent_distance
        self.structure_mask = structure_mask
        self.label = label
        self.label_mask = label_mask


def norm_mask(input_mask):
    output_mask = numpy.zeros(input_mask.shape)
    for i in range(len(input_mask)):
        if not numpy.all(input_mask[i] == 0):
            output_mask[i] = input_mask[i] / sum(input_mask[i])
    return output_mask


def load_bert_xlnet_roberta_input_tensors(statement_jsonl_path, label_map_path, kb_entity_path, model_type,
                                          model_name, max_seq_length, max_ent_cnt, set_type):

    def read_examples(input_file):
        examples = []

        with open(input_file) as f:
            file_content = json.load(f)

        for (i, ins) in enumerate(file_content):
            guid = "%s-%s" % (set_type, i)
            examples.append(SkiExample(guid=guid,
                                       title=ins['title'],
                                       vertex_set=ins['vertexSet'],
                                       sents=ins['sents'],
                                       labels=ins['labels'] if set_type != "test" else None))

        return examples

    def ski_convert_examples_to_features(examples, model_type, tokenizer, kb_ent_idx_map, max_length=512, max_ent_cnt=42,
                                         label_map=None, pad_token=0):
        """
        Loads a data file into a list of ``InputFeatures``

        Args:
            examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length
            task: GLUE task
            label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
            output_mode: String indicating the output mode. Either ``regression`` or ``classification``
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token: Padding token
            pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)

        Returns:
            If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
            containing the task-specific features. If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.

        """

        features, all_tok_kb_ent_edges = [], []

        ner_map = {'PAD': 0, 'ORG': 1, 'LOC': 2, 'NUM': 3, 'TIME': 4, 'MISC': 5, 'PER': 6}
        distance_buckets = numpy.zeros((512), dtype='int64')
        distance_buckets[1] = 1
        distance_buckets[2:] = 2
        distance_buckets[4:] = 3
        distance_buckets[8:] = 4
        distance_buckets[16:] = 5
        distance_buckets[32:] = 6
        distance_buckets[64:] = 7
        distance_buckets[128:] = 8
        distance_buckets[256:] = 9

        for (ex_index, example) in enumerate(tqdm(examples, desc="Converting examples to features")):

            len_examples = len(examples)

            input_tokens = []
            tok_to_sent = []
            tok_to_word = []
            for sent_idx, sent in enumerate(example.sents):
                for word_idx, word in enumerate(sent):
                    tokens_tmp = tokenizer.tokenize(word, add_prefix_space=True)
                    input_tokens += tokens_tmp
                    tok_to_sent += [sent_idx] * len(tokens_tmp)
                    tok_to_word += [word_idx] * len(tokens_tmp)

            if len(input_tokens) <= max_length - 2:
                if model_type == 'roberta':
                    input_tokens = [tokenizer.bos_token] + input_tokens + [tokenizer.eos_token]
                else:
                    input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
                tok_to_sent = [None] + tok_to_sent + [None]
                tok_to_word = [None] + tok_to_word + [None]
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                attention_mask = [1] * len(input_ids)
                token_type_ids = [0] * len(input_ids)
                # padding
                padding = [None] * (max_length - len(input_ids))
                tok_to_sent += padding
                tok_to_word += padding
                padding = [0] * (max_length - len(input_ids))
                attention_mask += padding
                token_type_ids += padding
                padding = [pad_token] * (max_length - len(input_ids))
                input_ids += padding
            else:
                input_tokens = input_tokens[:max_length - 2]
                tok_to_sent = tok_to_sent[:max_length - 2]
                tok_to_word = tok_to_word[:max_length - 2]
                if model_type == 'roberta':
                    input_tokens = [tokenizer.bos_token] + input_tokens + [tokenizer.eos_token]
                else:
                    input_tokens = [tokenizer.cls_token] + input_tokens + [tokenizer.sep_token]
                tok_to_sent = [None] + tok_to_sent + [None]
                tok_to_word = [None] + tok_to_word + [None]
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                attention_mask = [1] * len(input_ids)
                token_type_ids = [pad_token] * len(input_ids)

            # ent_mask & ner / coreference feature
            ent_mask = numpy.zeros((max_ent_cnt, max_length), dtype='int64')
            ent_ner = [0] * max_length
            ent_pos = [0] * max_length
            tok_to_ent = [-1] * max_length
            ents = example.vertex_set

            tok_kb_ent_edges = []

            for ent_idx, ent in enumerate(ents):
                for mention in ent:
                    for tok_idx in range(len(input_ids)):
                        if tok_to_sent[tok_idx] == mention['sent_id'] \
                                and mention['pos'][0] <= tok_to_word[tok_idx] < mention['pos'][1]:
                            ent_mask[ent_idx][tok_idx] = 1
                            ent_ner[tok_idx] = ner_map[ent[0]['type']]
                            ent_pos[tok_idx] = ent_idx + 1
                            tok_to_ent[tok_idx] = ent_idx
                            if "ent_alias" in mention:
                                kb_ent_alias = mention["ent_alias"]
                                tok_kb_ent_edges.append((tok_idx, kb_ent_idx_map[kb_ent_alias]))
            all_tok_kb_ent_edges.append(tok_kb_ent_edges)

            # distance feature
            ent_first_appearance = [0] * max_ent_cnt
            ent_distance = numpy.zeros((max_ent_cnt, max_ent_cnt), dtype='int8')  # padding id is 10
            for i in range(len(ents)):
                if numpy.all(ent_mask[i] == 0):
                    continue
                else:
                    ent_first_appearance[i] = numpy.where(ent_mask[i] == 1)[0][0]
            for i in range(len(ents)):
                for j in range(len(ents)):
                    if ent_first_appearance[i] != 0 and ent_first_appearance[j] != 0:
                        if ent_first_appearance[i] >= ent_first_appearance[j]:
                            ent_distance[i][j] = distance_buckets[ent_first_appearance[i] - ent_first_appearance[j]]
                        else:
                            ent_distance[i][j] = - distance_buckets[- ent_first_appearance[i] + ent_first_appearance[j]]
            ent_distance += 10  # norm from [-9, 9] to [1, 19]

            structure_mask = numpy.zeros((5, max_length, max_length), dtype='float')
            for i in range(max_length):
                if attention_mask[i] == 0:
                    break
                else:
                    if tok_to_ent[i] != -1:
                        for j in range(max_length):
                            if tok_to_sent[j] is None:
                                continue
                            # intra
                            if tok_to_sent[j] == tok_to_sent[i]:
                                # intra-coref
                                if tok_to_ent[j] == tok_to_ent[i]:
                                    structure_mask[0][i][j] = 1
                                # intra-relate
                                elif tok_to_ent[j] != -1:
                                    structure_mask[1][i][j] = 1
                                # intra-NA
                                else:
                                    structure_mask[2][i][j] = 1
                            # inter
                            else:
                                # inter-coref
                                if tok_to_ent[j] == tok_to_ent[i]:
                                    structure_mask[3][i][j] = 1
                                # inter-relate
                                elif tok_to_ent[j] != -1:
                                    structure_mask[4][i][j] = 1

            # label
            label_ids = numpy.zeros((max_ent_cnt, max_ent_cnt, len(label_map.keys())), dtype='bool')
            # test file does not have "labels"
            if example.labels is not None:
                labels = example.labels
                for label in labels:
                    label_ids[label['h']][label['t']][label_map[label['r']]] = 1
            for h in range(len(ents)):
                for t in range(len(ents)):
                    if numpy.all(label_ids[h][t] == 0):
                        label_ids[h][t][0] = 1

            label_mask = numpy.zeros((max_ent_cnt, max_ent_cnt), dtype='bool')
            label_mask[:len(ents), :len(ents)] = 1
            for ent in range(len(ents)):
                label_mask[ent][ent] = 0
            for ent in range(len(ents)):
                if numpy.all(ent_mask[ent] == 0):
                    label_mask[ent, :] = 0
                    label_mask[:, ent] = 0

            ent_mask = norm_mask(ent_mask)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            assert ent_mask.shape == (max_ent_cnt, max_length)
            assert label_ids.shape == (max_ent_cnt, max_ent_cnt, len(label_map.keys()))
            assert label_mask.shape == (max_ent_cnt, max_ent_cnt)
            assert len(ent_ner) == max_length
            assert len(ent_pos) == max_length
            assert ent_distance.shape == (max_ent_cnt, max_ent_cnt)
            assert structure_mask.shape == (5, max_length, max_length)

            docred_input_features = DocREDInputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                ent_mask=ent_mask,
                ent_ner=ent_ner,
                ent_pos=ent_pos,
                ent_distance=ent_distance,
                structure_mask=structure_mask,
                label=label_ids,
                label_mask=label_mask,
            )
            features.append(
                SkiInputFeatures(example_id=example.guid, docred_input_features=docred_input_features)
            )

        return features, all_tok_kb_ent_edges

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def ski_convert_features_to_tensors(features):
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.docred_input_features.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.docred_input_features.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.docred_input_features.token_type_ids for f in features], dtype=torch.long)
        all_ent_mask = torch.tensor([f.docred_input_features.ent_mask for f in features], dtype=torch.float)
        all_ent_ner = torch.tensor([f.docred_input_features.ent_ner for f in features], dtype=torch.long)
        all_ent_pos = torch.tensor([f.docred_input_features.ent_pos for f in features], dtype=torch.long)
        all_ent_distance = torch.tensor([f.docred_input_features.ent_distance for f in features], dtype=torch.long)
        all_structure_mask = torch.tensor([f.docred_input_features.structure_mask for f in features], dtype=torch.bool)
        all_label = torch.tensor([f.docred_input_features.label for f in features], dtype=torch.bool)
        all_label_mask = torch.tensor([f.docred_input_features.label_mask for f in features], dtype=torch.bool)
        return all_input_ids, all_attention_mask, all_token_type_ids, all_ent_mask, all_ent_ner, \
            all_ent_pos, all_ent_distance, all_structure_mask, all_label_mask, all_label

    with open(label_map_path) as f:
        label_map = json.load(f)

    kb_entities = load_text_as_list(kb_entity_path)
    kb_ent_idx_map = {k: i for i, k in enumerate(kb_entities)}

    tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name)
    # examples = qagnn_read_example(statement_jsonl_path)
    # features = qagnn_convert_examples_to_features(examples, list(range(len(examples[0].endings))), max_seq_length, tokenizer,
    #                                         cls_token_at_end=bool(model_type in ['xlnet']),  # xlnet has a cls token at the end
    #                                         cls_token=tokenizer.cls_token,
    #                                         sep_token=tokenizer.sep_token,
    #                                         sep_token_extra=bool(model_type in ['roberta', 'albert']),
    #                                         cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
    #                                         pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
    #                                         pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
    #                                         sequence_b_segment_id=0 if model_type in ['roberta', 'albert'] else 1)
    examples = read_examples(statement_jsonl_path)
    features, data_tok_kb_ent_edges = ski_convert_examples_to_features(examples, model_type, tokenizer,
                                                                       max_length=max_seq_length,
                                                                       max_ent_cnt=max_ent_cnt,
                                                                       label_map=label_map,
                                                                       kb_ent_idx_map=kb_ent_idx_map
                                                                       )
    example_ids = [f.example_id for f in features]
    *data_tensors, all_label = ski_convert_features_to_tensors(features)
    return (example_ids, all_label, data_tok_kb_ent_edges, label_map, examples, *data_tensors)


def load_input_tensors(input_jsonl_path, label_map_path, kb_entity_path, model_type, model_name, max_seq_length, max_ent_cnt, set_type):
    return load_bert_xlnet_roberta_input_tensors(input_jsonl_path, label_map_path, kb_entity_path,
                                                 model_type, model_name, max_seq_length, max_ent_cnt, set_type)


def load_sparse_adj_data_with_contextnode(adj_pk_path, all_tok_kb_ent_edges, max_node_num, max_mention_node_num, args):
    cache_path = adj_pk_path + '.loaded_cache'
    use_cache = False # TODO: change this back
    half_n_rel = None

    if use_cache and not os.path.exists(cache_path):
        use_cache = False
    actual_max_mention_node_num = 0

    if use_cache:
        with open(cache_path, 'rb') as f:
            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel, \
                all_mention_toks, valid_node_mask = pickle.load(f)
    else:
        with open(adj_pk_path, 'rb') as fin:
            adj_concept_pairs = pickle.load(fin)

        n_samples = len(adj_concept_pairs)  # this is actually n_questions x n_choices
        edge_index, edge_type = [], []
        adj_lengths = torch.zeros((n_samples,), dtype=torch.long)
        concept_ids = torch.full((n_samples, max_node_num-max_mention_node_num), 1, dtype=torch.long)
        node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long)  # default 2: "other node" or extra nodes
        node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)
        all_mention_toks = torch.zeros((n_samples, max_mention_node_num), dtype=torch.long)
        valid_node_mask = torch.zeros((n_samples, max_node_num), dtype=torch.long)

        all_rel_ids = []

        adj_lengths_ori = adj_lengths.clone()
        for idx, (_data, tok_kb_ent_edges) in tqdm(enumerate(zip(adj_concept_pairs, all_tok_kb_ent_edges)), total=n_samples, desc='loading adj matrices'):
            adj, concepts, ent_mask, cid2score = _data['adj'], _data['concepts'], _data['ent_mask'], _data['ent2score']
            # DEBUG: Change this
            if ent_mask is None:
            # half_n_rel = 52
            # if True:
                edge_index.append(torch.zeros((2, 1)))
                edge_type.append(torch.zeros(1))
                continue

            num_concept = len(concepts)
            # remove duplicate edges
            unique_tok_kb_ent_edges = list(set(tok_kb_ent_edges))
            # concepts = entities + extra_nodes
            mention_toks = [x[0] for x in unique_tok_kb_ent_edges]
            mention_toks = sorted(list(set(mention_toks)))
            n_mention_toks = len(mention_toks)
            actual_max_mention_node_num = max(n_mention_toks, actual_max_mention_node_num)
            assert num_concept + max_mention_node_num < max_node_num, \
                "Please increase max node number or risk missing entities in documents"
            assert max_mention_node_num >= n_mention_toks, "Found example with {} mentioned tokens > {} max_mention_node"

            # assert len(mention_toks) == len(set(mention_toks)), \
            #     "Oops, our assumption that each mention token only refer to 1 KB entity might not be correct"
            mention_toks2idxs = {k: i for i, k in enumerate(mention_toks)}
            linked_entities = [x[1] for x in unique_tok_kb_ent_edges]
            # sanity check all linked entities are in the concepts
            assert set(linked_entities).issubset(set(concepts))

            all_mention_toks[idx, :n_mention_toks] = torch.tensor(mention_toks)+1   # IMPORTANT: save 1 for the padding
            # create a valid node mask to mark the context mention node, kb entity node and other kb nodes
            valid_node_mask[idx, :n_mention_toks] = torch.ones(n_mention_toks)
            valid_node_mask[idx, max_mention_node_num: max_mention_node_num+num_concept] = torch.ones(num_concept)

            assert ent_mask[0]
            F_start = False
            for TF in ent_mask:
                if TF == False:
                    F_start = True
                else:
                    assert not F_start

            # this is the final number of nodes including # mentioned toks, entities and extra nodes
            extended_num_valid_nodes = max_mention_node_num + num_concept
            num_valid_nodes = min(extended_num_valid_nodes, max_node_num)
            adj_lengths_ori[idx] = num_concept
            adj_lengths[idx] = num_valid_nodes

            # Prepare nodes
            concept_ids[idx, :num_concept] = torch.tensor(concepts)
            concepts2idx = {c: i for i, c in enumerate(concepts)}

            # Prepare node scores
            if (cid2score is not None):
                # for mention token node, the node score = 1 by default
                for _j_ in range(0, n_mention_toks):
                    node_scores[idx, _j_, 0] = torch.tensor(1)

                for _j_ in range(num_concept):
                    _cid = int(concept_ids[idx, _j_])
                    assert _cid in cid2score
                    node_scores[idx, max_mention_node_num+_j_, 0] = torch.tensor(cid2score[_cid])

            # 2 masks, 1 for entity node in the context, and 1 in the kb
            kb_ent_mask = np.concatenate([np.zeros(max_mention_node_num, dtype=bool), ent_mask])
            context_ent_mask = np.concatenate([np.ones(n_mention_toks), np.zeros(max_mention_node_num-n_mention_toks),
                                               np.zeros(len(concepts), dtype=bool)])
            assert len(context_ent_mask) == len(kb_ent_mask) == max_mention_node_num+num_concept
            # Prepare node types
            # 0: context (mentioned token) node
            # 1: kb node
            # 2: other nodes (extra + padding nodes)
            node_type_ids[idx, :max_mention_node_num+num_concept][
                torch.tensor(kb_ent_mask, dtype=torch.bool)[:max_mention_node_num+num_concept]] = 1  # for entities in kb
            node_type_ids[idx, :max_mention_node_num+num_concept][
                torch.tensor(context_ent_mask, dtype=torch.bool)[:max_mention_node_num+num_concept]] = 0 # for entities in context

            # Load adj
            ij = torch.tensor(adj.row, dtype=torch.int64)  # (num_matrix_entries, ), where each entry is coordinate
            k = torch.tensor(adj.col, dtype=torch.int64)  # (num_matrix_entries, ), where each entry is coordinate
            n_node = adj.shape[1]
            half_n_rel = adj.shape[0] // n_node
            i, j = ij // n_node, ij % n_node

            ex_rel_ids = adj.row // n_node
            all_rel_ids.extend(ex_rel_ids)

            # i: rel_id, j: coord 1, k: coord 2
            # we add 1 more relation between the KB entity node and context entity node
            # we also add the context entity node into the adj graph
            # prepare edges
            i += 1
            j += max_mention_node_num
            k += max_mention_node_num  # **** increment coordinate by 1, rel_id by 2 ****
            extra_i, extra_j, extra_k = [], [], []
            for tok, kb_ent_idx in unique_tok_kb_ent_edges:
                extra_i.append(0)  # rel from contextnode to question concept
                extra_j.append(mention_toks2idxs[tok])  # mention_node index
                kb_node_idx = concepts2idx[kb_ent_idx] + max_mention_node_num
                extra_k.append(kb_node_idx)  # question concept coordinate

            half_n_rel += 1  # should be 19 now
            if len(extra_i) > 0:
                i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                k = torch.cat([k, torch.tensor(extra_k)], dim=0)

            mask = (j < max_node_num) & (k < max_node_num)
            i, j, k = i[mask], j[mask], k[mask]
            i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j),
                                                                                         0)  # add inverse relations
            edge_index.append(torch.stack([j, k], dim=0))  # each entry is [2, E]
            edge_type.append(i)  # each entry is [E, ]

        assert half_n_rel is not None
        with open(cache_path, 'wb') as f:
            pickle.dump(
                [adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type,
                 half_n_rel, all_mention_toks, valid_node_mask], f)

        # ori_adj_mean = adj_lengths_ori.float().mean().item()
        # ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean) ** 2).mean().item())
        # print('| ori_adj_len: mu {:.2f} sigma {:.2f} | adj_len: {:.2f} |'.format(ori_adj_mean, ori_adj_sigma,
        #                                                                          adj_lengths.float().mean().item()) +
        #       ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
        #       ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
        #                                                   (node_type_ids == 1).float().sum(1).mean().item()))

        # concept_ids: (n_questions, num_choice, max_node_num)
        # node_type_ids: (n_questions, num_choice, max_node_num)
        # node_scores: (n_questions, num_choice, max_node_num)
        # adj_lengths: (n_questions,　num_choice)

    edge_index = list(map(list, zip(*(iter(edge_index),) * 1)))
    edge_type = list(map(list, zip(*(iter(edge_type),) * 1)))

    print("Actual max mention node num = {}".format(actual_max_mention_node_num))

    return concept_ids, node_type_ids, node_scores, all_mention_toks, valid_node_mask, \
           adj_lengths, (edge_index, edge_type), half_n_rel  # , half_n_rel * 2 + 1
