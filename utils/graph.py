import pandas as pd
import torch
import networkx as nx
import itertools
import json
from tqdm import tqdm
import numpy as np
from scipy import sparse
from scipy.spatial import distance
import pickle
from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool
from collections import OrderedDict

from utils.maths import *
from utils.io import *
from ogb.lsc import WikiKG90Mv2Dataset

MAX_NODES = 200
ENTITY_DIM = 768
RELEVANCE_SCORE_THRESHOLD = 0.35
WikiKG90Mv2_ROOT = "data/WikiKG90Mv2Dataset"

__all__ = ['generate_graph']

concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None

wiki_dataset = None
title2large_idx = None


def load_resources(wiki_entity_path, wiki_relation_path):
    global concept2id, id2concept, relation2id, id2relation

    id2concept = load_text_as_list(wiki_entity_path)
    concept2id = {w: i for i, w in enumerate(id2concept)}
    id2relation = load_text_as_list(wiki_relation_path)
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_large_wiki_resources():
    global wiki_dataset, title2large_idx
    wiki_dataset = WikiKG90Mv2Dataset(root=WikiKG90Mv2_ROOT)
    wiki_entity_path = os.path.join(WikiKG90Mv2_ROOT, "wikikg90m-v2", "mapping", "entity.csv")
    title2large_idx = {}
    chunksize = 100000
    concept_sets = set(id2concept)
    for chunk in tqdm(pd.read_csv(wiki_entity_path, chunksize=chunksize), desc="Load wiki in chunks"):
        entity_idxs, entity_titles = chunk["idx"], chunk["title"]
        for eid, entity_title in zip(entity_idxs, entity_titles):
            if entity_title in concept_sets:
                title2large_idx[entity_title] = eid


def compute_relevance_score(cid, extra_id):
    concept_title_id = title2large_idx[id2concept[cid]]
    concept_feature = wiki_dataset.entity_feat[int(concept_title_id)]

    extra_title_id =  title2large_idx[id2concept[extra_id]]
    extra_feature = wiki_dataset.entity_feat[int(extra_title_id)]
    return distance.cosine(concept_feature, extra_feature)


def batch_compute_relevance_score(v1_list, v2):
    v2_feature = wiki_dataset.entity_feat[int(title2large_idx[id2concept[v2]])]
    v1_feature_list = np.array([wiki_dataset.entity_feat[int(title2large_idx[id2concept[v1]])]
                                for v1 in v1_list])
    v1_norms = np.linalg.norm(v1_feature_list, axis=1)
    v2_norm = np.linalg.norm(v2_feature)
    norm_products = v1_norms * v2_norm
    return np.matmul(v1_feature_list, v2_feature) / norm_products


def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def relational_graph_generation(qcs, acs, paths, rels):
    raise NotImplementedError()  # TODO


# plain graph generation
def plain_graph_generation(qcs, acs, paths, rels):
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple

    graph = nx.Graph()
    for p in paths:
        for c_index in range(len(p) - 1):
            h = p[c_index]
            t = p[c_index + 1]
            # TODO: the weight can computed by concept embeddings and relation embeddings of TransE
            graph.add_edge(h, t, weight=1.0)

    for qc1, qc2 in list(itertools.combinations(qcs, 2)):
        if cpnet_simple.has_edge(qc1, qc2):
            graph.add_edge(qc1, qc2, weight=1.0)

    for ac1, ac2 in list(itertools.combinations(acs, 2)):
        if cpnet_simple.has_edge(ac1, ac2):
            graph.add_edge(ac1, ac2, weight=1.0)

    if len(qcs) == 0:
        qcs.append(-1)

    if len(acs) == 0:
        acs.append(-1)

    if len(paths) == 0:
        for qc in qcs:
            for ac in acs:
                graph.add_edge(qc, ac, rel=-1, weight=0.1)

    g = nx.convert_node_labels_to_integers(graph, label_attribute='cid')  # re-index
    return nx.node_link_data(g)


def generate_adj_matrix_per_inst(nxg_str):
    global id2relation
    n_rel = len(id2relation)

    nxg = nx.node_link_graph(json.loads(nxg_str))
    n_node = len(nxg.nodes)
    cids = np.zeros(n_node, dtype=np.int32)
    for node_id, node_attr in nxg.nodes(data=True):
        cids[node_id] = node_attr['cid']

    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet_all.has_edge(s_c, t_c):
                for e_attr in cpnet_all[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    cids += 1
    adj = coo_matrix(adj.reshape(-1, n_node))
    return (adj, cids)


def concepts2adj(node_ids):
    global id2relation
    cids = np.array(node_ids, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet.has_edge(s_c, t_c):
                for e_attr in cpnet[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    # cids += 1  # note!!! index 0 is reserved for padding
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cids


def concepts_to_adj_matrices_1hop_neighbours(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for u in set(qc_ids) | set(ac_ids):
        if u in cpnet.nodes:
            extra_nodes |= set(cpnet[u])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_1hop_neighbours_without_relatedto(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for u in set(qc_ids) | set(ac_ids):
        if u in cpnet.nodes:
            for v in cpnet[u]:
                for data in cpnet[u][v].values():
                    if data['rel'] not in (15, 32):
                        extra_nodes.add(v)
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_2hop_qa_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_2hop_all_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_2step_relax_all_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    intermediate_ids = extra_nodes - qa_nodes
    for qid in intermediate_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    for qid in qc_ids:
        for aid in intermediate_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_3hop_qa_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                for u in cpnet_simple[qid]:
                    for v in cpnet_simple[aid]:
                        if cpnet_simple.has_edge(u, v):  # ac is a 3-hop neighbour of qc
                            extra_nodes.add(u)
                            extra_nodes.add(v)
                        if u == v:  # ac is a 2-hop neighbour of qc
                            extra_nodes.add(u)
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask



######################################################################
from transformers import RobertaTokenizer, RobertaForMaskedLM

class RobertaForMaskedLMwithLoss(RobertaForMaskedLM):
    #
    def __init__(self, config):
        super().__init__(config)
    #
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
        #
        assert attention_mask is not None
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0] #hidden_states of final layer (batch_size, sequence_length, hidden_size)
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
            # (masked_lm_loss), prediction_scores, sequence_output, (hidden_states), (attentions)
        return outputs

print ('loading pre-trained LM...')
TOKENIZER = RobertaTokenizer.from_pretrained('roberta-large')
LM_MODEL = RobertaForMaskedLMwithLoss.from_pretrained('roberta-large')
LM_MODEL.cuda(); LM_MODEL.eval()
print ('loading done')

def get_LM_score(cids, question):
    cids = cids[:]
    cids.insert(0, -1) #QAcontext node
    sents, scores = [], []
    for cid in cids:
        if cid==-1:
            sent = question.lower()
        else:
            sent = '{} {}.'.format(question.lower(), ' '.join(id2concept[cid].split('_')))
        sent = TOKENIZER.encode(sent, add_special_tokens=True)
        sents.append(sent)
    n_cids = len(cids)
    cur_idx = 0
    batch_size = 50
    while cur_idx < n_cids:
        #Prepare batch
        input_ids = sents[cur_idx: cur_idx+batch_size]
        max_len = max([len(seq) for seq in input_ids])
        for j, seq in enumerate(input_ids):
            seq += [TOKENIZER.pad_token_id] * (max_len-len(seq))
            input_ids[j] = seq
        input_ids = torch.tensor(input_ids).cuda() #[B, seqlen]
        mask = (input_ids!=1).long() #[B, seq_len]
        #Get LM score
        with torch.no_grad():
            outputs = LM_MODEL(input_ids, attention_mask=mask, masked_lm_labels=input_ids)
            loss = outputs[0] #[B, ]
            _scores = list(-loss.detach().cpu().numpy()) #list of float
        scores += _scores
        cur_idx += batch_size
    assert len(sents) == len(scores) == len(cids)
    cid2score = OrderedDict(sorted(list(zip(cids, scores)), key=lambda x: -x[1])) #score: from high to low
    return cid2score

def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1(data):
    qc_ids, ac_ids, question = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    return (sorted(qc_ids), sorted(ac_ids), question, sorted(extra_nodes))

def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(data):
    qc_ids, ac_ids, question, extra_nodes = data
    cid2score = get_LM_score(qc_ids+ac_ids+extra_nodes, question)
    return (qc_ids, ac_ids, question, extra_nodes, cid2score)

def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part3(data):
    qc_ids, ac_ids, question, extra_nodes, cid2score = data
    schema_graph = qc_ids + ac_ids + sorted(extra_nodes, key=lambda x: -cid2score[x]) #score: from high to low
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return {'adj': adj, 'concepts': concepts, 'qmask': qmask, 'amask': amask, 'cid2score': cid2score}

################################################################################



#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################


def generate_graph(grounded_path, pruned_paths_path, cpnet_vocab_path, cpnet_graph_path, output_path):
    print(f'generating schema graphs for {grounded_path} and {pruned_paths_path}...')

    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    global cpnet, cpnet_simple
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    nrow = sum(1 for _ in open(grounded_path, 'r'))
    with open(grounded_path, 'r') as fin_gr, \
            open(pruned_paths_path, 'r') as fin_pf, \
            open(output_path, 'w') as fout:
        for line_gr, line_pf in tqdm(zip(fin_gr, fin_pf), total=nrow):
            mcp = json.loads(line_gr)
            qa_pairs = json.loads(line_pf)

            statement_paths = []
            statement_rel_list = []
            for qas in qa_pairs:
                if qas["pf_res"] is None:
                    cur_paths = []
                    cur_rels = []
                else:
                    cur_paths = [item["path"] for item in qas["pf_res"]]
                    cur_rels = [item["rel"] for item in qas["pf_res"]]
                statement_paths.extend(cur_paths)
                statement_rel_list.extend(cur_rels)

            qcs = [concept2id[c] for c in mcp["qc"]]
            acs = [concept2id[c] for c in mcp["ac"]]

            gobj = plain_graph_generation(qcs=qcs, acs=acs,
                                          paths=statement_paths,
                                          rels=statement_rel_list)
            fout.write(json.dumps(gobj) + '\n')

    print(f'schema graphs saved to {output_path}')
    print()


def generate_adj_matrices(ori_schema_graph_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes, num_rels=34, debug=False):
    print(f'generating adjacency matrices for {ori_schema_graph_path} and {cpnet_graph_path}...')

    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    global cpnet_all
    if cpnet_all is None:
        cpnet_all = nx.read_gpickle(cpnet_graph_path)

    with open(ori_schema_graph_path, 'r') as fin:
        nxg_strs = [line for line in fin]

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(generate_adj_matrix_per_inst, nxg_strs), total=len(nxg_strs)))

    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adjacency matrices saved to {output_path}')
    print()


def generate_adj_data_from_grounded_concepts(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            a_ids = set(concept2id[c] for c in dic['ac'])
            q_ids = q_ids - a_ids
            qa_data.append((q_ids, a_ids))

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair, qa_data), total=len(qa_data)))

    # res is a list of tuples, each tuple consists of four elements (adj, concepts, qmask, amask)
    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adj data saved to {output_path}')
    print()


def ski_generate_adj_data_from_grounded_concepts__use_LM(grounded_path, cpnet_graph_path,
                                                         cpnet_vocab_path, cpnet_relation_path, output_path, num_processes):
    print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path, cpnet_relation_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)
    if any(x is None for x in [wiki_dataset, title2large_idx]):
        load_large_wiki_resources()

    grounded_data = load_json(grounded_path)
    re_data = []
    for example in tqdm(grounded_data, desc="Preparing"):
        ent_aliases = [ent[0]["ent_alias"] for ent in example["vertexSet"] if "ent_alias" in ent[0]]
        ent_ids = set([concept2id[ent] for ent in ent_aliases])

        doc_sents = [" ".join(s) for s in example["sents"]]
        doc_text = " ".join(doc_sents)
        re_data.append((ent_ids, doc_text))

    if num_processes > 1:
        print("Run multi-processing")
        with Pool(num_processes) as p:
            res1 = list(tqdm(p.imap(ski_concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1, re_data), total=len(re_data),
                             desc="Running part 1"))

        with Pool(num_processes) as p:
            res2 = list(tqdm(p.imap(ski_concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2, res1), total=len(res1),
                             desc="Running part 2"))

        with Pool(num_processes) as p:
            res3 = list(tqdm(p.imap(ski_concepts_to_adj_matrices_2hop_all_pair__use_LM__Part3, res2), total=len(res2),
                             desc="Running part 3"))
    else:
        print("Run single processing")
        res1 = []
        for data in tqdm(re_data, desc="Running part 1"):
            res1.append(ski_concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1(data))
        res2 = []
        for data in tqdm(res1, desc="Running part 2"):
            res2.append(ski_concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(data))
        res3 = []
        for i, data in enumerate(tqdm(res2, desc="Running part 3")):
            res3.append(ski_concepts_to_adj_matrices_2hop_all_pair__use_LM__Part3(data))


    save_to_pickle(res3, output_path)

    print(f'adj data saved to {output_path}')
    print()


def ski_concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1(data):
    ent_ids, doc_text = data

    if len(ent_ids) == 0:
        return ([], doc_text, [])

    extra_nodes = set()
    avg_neighbor_size = int(MAX_NODES / len(ent_ids))
    for ent_id in tqdm(ent_ids, desc="Get 2hop pairs"):
        if ent_id in cpnet_simple.nodes:
            neighbors = list(set(cpnet_simple[ent_id]))
            neighbors = np.random.choice(neighbors, size=avg_neighbor_size, replace=False) \
                if len(neighbors) > avg_neighbor_size else neighbors
            distances = batch_compute_relevance_score(neighbors, ent_id)
            selected_indices = list(np.argwhere(distances > RELEVANCE_SCORE_THRESHOLD).T[0])
            extra_nodes |= set([neighbors[i] for i in selected_indices])
    # for ent_id in ent_ids:
    #     if ent_id in cpnet_simple.nodes and ent_id in cpnet_simple.nodes:
    #         extra_nodes |= set(cpnet_simple[ent_id])
    extra_nodes = extra_nodes - ent_ids
    return (sorted(ent_ids), doc_text, sorted(extra_nodes))


def ski_concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(data):
    ent_ids, doc_text, extra_nodes = data
    if len(ent_ids) + len(extra_nodes) > MAX_NODES:
        eidscore = ski_get_LM_score(ent_ids, extra_nodes)
    else:
        eidscore = OrderedDict([(e, 1) for e in ent_ids + extra_nodes])
    return (ent_ids, doc_text, extra_nodes, eidscore)


def ski_get_LM_score(ent_ids, extra_nodes):
    cid2score = {ent_id: 1 for ent_id in ent_ids}
    for extra_id in tqdm(extra_nodes, desc="Get LM score"):
        all_distances = batch_compute_relevance_score(ent_ids, extra_id)
        avg_dist = np.mean(all_distances)
        cid2score[extra_id] = avg_dist
    return cid2score


def ski_concepts_to_adj_matrices_2hop_all_pair__use_LM__Part3(data):
    ent_ids, doc_text, extra_nodes, eidscore = data
    schema_graph = ent_ids + sorted(extra_nodes, key=lambda x: -eidscore[x])  # score: from high to low
    if len(schema_graph) > 0:
        arange = np.arange(len(schema_graph))
        ent_mask = arange < len(ent_ids)
        adj, concepts = concepts2adj(schema_graph)
    else:
        adj, concepts, ent_mask = None, [], None
    return {'adj': adj, 'concepts': concepts, 'ent_mask': ent_mask, 'ent2score': eidscore}


def generate_adj_data_from_grounded_concepts__use_LM(grounded_path, cpnet_graph_path,
                                                     cpnet_vocab_path, cpnet_relation_path, output_path, num_processes):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
        (5) cid2score that maps a concept id to its relevance score given the QA context
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path, cpnet_relation_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    qa_data = []
    statement_path = grounded_path.replace('grounded', 'statement')
    with open(grounded_path, 'r', encoding='utf-8') as fin_ground, open(statement_path, 'r', encoding='utf-8') as fin_state:
        lines_ground = fin_ground.readlines()
        lines_state  = fin_state.readlines()
        assert len(lines_ground) % len(lines_state) == 0
        n_choices = len(lines_ground) // len(lines_state)
        for j, line in enumerate(lines_ground):
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            a_ids = set(concept2id[c] for c in dic['ac'])
            q_ids = q_ids - a_ids
            statement_obj = json.loads(lines_state[j//n_choices])
            QAcontext = "{} {}.".format(statement_obj['question']['stem'], dic['ans'])
            qa_data.append((q_ids, a_ids, QAcontext))

    with Pool(num_processes) as p:
        res1 = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1, qa_data), total=len(qa_data)))

    res2 = []
    for j, _data in enumerate(res1):
        if j % 100 == 0: print (j)
        res2.append(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(_data))

    with Pool(num_processes) as p:
        res3 = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part3, res2), total=len(res2)))

    # res is a list of responses
    with open(output_path, 'wb') as fout:
        pickle.dump(res3, fout)

    print(f'adj data saved to {output_path}')
    print()



#################### adj to sparse ####################

def coo_to_normalized_per_inst(data):
    adj, concepts, qm, am, max_node_num = data
    ori_adj_len = len(concepts)
    concepts = torch.tensor(concepts[:min(len(concepts), max_node_num)])
    adj_len = len(concepts)
    qm = torch.tensor(qm[:adj_len], dtype=torch.uint8)
    am = torch.tensor(am[:adj_len], dtype=torch.uint8)
    ij = adj.row
    k = adj.col
    n_node = adj.shape[1]
    n_rel = 2 * adj.shape[0] // n_node
    i, j = ij // n_node, ij % n_node
    mask = (j < max_node_num) & (k < max_node_num)
    i, j, k = i[mask], j[mask], k[mask]
    i, j, k = np.concatenate((i, i + n_rel // 2), 0), np.concatenate((j, k), 0), np.concatenate((k, j), 0)  # add inverse relations
    adj_list = []
    for r in range(n_rel):
        mask = i == r
        ones = np.ones(mask.sum(), dtype=np.float32)
        A = sparse.csr_matrix((ones, (k[mask], j[mask])), shape=(max_node_num, max_node_num))  # A is transposed by exchanging the order of j and k
        adj_list.append(normalize_sparse_adj(A, 'coo'))
    adj_list.append(sparse.identity(max_node_num, dtype=np.float32, format='coo'))
    return ori_adj_len, adj_len, concepts, adj_list, qm, am


def coo_to_normalized(adj_path, output_path, max_node_num, num_processes):
    print(f'converting {adj_path} to normalized adj')

    with open(adj_path, 'rb') as fin:
        adj_data = pickle.load(fin)
    data = [(adj, concepts, qmask, amask, max_node_num) for adj, concepts, qmask, amask in adj_data]

    ori_adj_lengths = torch.zeros((len(data),), dtype=torch.int64)
    adj_lengths = torch.zeros((len(data),), dtype=torch.int64)
    concepts_ids = torch.zeros((len(data), max_node_num), dtype=torch.int64)
    qmask = torch.zeros((len(data), max_node_num), dtype=torch.uint8)
    amask = torch.zeros((len(data), max_node_num), dtype=torch.uint8)

    adj_data = []
    with Pool(num_processes) as p:
        for i, (ori_adj_len, adj_len, concepts, adj_list, qm, am) in tqdm(enumerate(p.imap(coo_to_normalized_per_inst, data)), total=len(data)):
            ori_adj_lengths[i] = ori_adj_len
            adj_lengths[i] = adj_len
            concepts_ids[i][:adj_len] = concepts
            qmask[i][:adj_len] = qm
            amask[i][:adj_len] = am
            adj_list = [(torch.LongTensor(np.stack((adj.row, adj.col), 0)),
                         torch.FloatTensor(adj.data)) for adj in adj_list]
            adj_data.append(adj_list)

    torch.save((ori_adj_lengths, adj_lengths, concepts_ids, adj_data), output_path)

    print(f'normalized adj saved to {output_path}')
    print()


def generate_entity_features(entity_list_path, relation_list_path, entity_feature_path):
    # TODO: test this hyper-parameter
    dim = 128
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(entity_list_path, relation_list_path)
    if any(x is None for x in [wiki_dataset, title2large_idx]):
        load_large_wiki_resources()

    entities = load_text_as_list(entity_list_path)
    entity_ids = np.array([title2large_idx[e] for e in entities])

    batch_start, batch_size = 0, 128
    entity_features = []

    while batch_start < len(entities):
        if batch_start % 20 == 0:
            print(batch_start)
        batch_ids = entity_ids[batch_start: batch_start+batch_size]
        batch_features = wiki_dataset.entity_feat[batch_ids][:, :dim]
        entity_features.extend(batch_features)
        batch_start += batch_size
    entity_features = np.array(entity_features)

    with open(entity_feature_path, 'wb') as f:
        np.save(f, entity_features)

# if __name__ == '__main__':
#     generate_adj_matrices_from_grounded_concepts('./data/csqa/grounded/train.grounded.jsonl',
#                                                  './data/cpnet/conceptnet.en.pruned.graph',
#                                                  './data/cpnet/concept.txt',
#                                                  '/tmp/asdf', 40)
