import networkx as nx
import nltk
import pandas as pd
from ogb.lsc import WikiKG90Mv2Dataset
from utils.io import *
from tqdm import tqdm
from ogb.linkproppred import LinkPropPredDataset, Evaluator


RELATION_MAP = "data/WikiKG90Mv2Dataset/wikikg90m-v2/mapping/relation.csv"
ENTITY_MAP = "data/WikiKG90Mv2Dataset/wikikg90m-v2/mapping/entity.csv"
EDGE_PATH = "data/ogbl_wikikg2/raw/edge.csv"
EDGE_REL_PATH = "data/ogbl_wikikg2/raw/edge_reltype.csv"
NODE_IDX_2_ENTITY_IDX = "data/ogbl_wikikg2/mapping/nodeidx2entityid.csv"
RELTYPE_2_RELID = "data/ogbl_wikikg2/mapping/reltype2relid.csv"


REDUCED_RELATIONS = [92,
                     317,
                     118,
                     307,
                     323,
                     238,
                     408,
                     123,
                     356,
                     51,
                     159,
                     166,
                     116,
                     143,
                     152,
                     454,
                     386,
                     450,
                     59,
                     384,
                     42,
                     41,
                     257,
                     481,
                     460,
                     237,
                     518,
                     31,
                     275,
                     229,
                     325,
                     370,
                     251,
                     497,
                     500,
                     453,
                     137,
                     369,
                     443,
                     225,
                     168,
                     204,
                     133,
                     230,
                     283,
                     163,
                     207,
                     2,
                     394]

REDUCED_RELATION_SET = set(REDUCED_RELATIONS)


def load_merge_relation():
    # think of an effective way to map the relation if need to reduce the number of dimensions
    df = pd.read_csv(RELATION_MAP)
    relations = df["title"].tolist()
    relation_mapping = {x: x for x in relations}
    return relation_mapping, relations


def load_entities():
    df = pd.read_csv(ENTITY_MAP)
    relations = df["entity"].tolist()
    return relations


def load_maps(map_path, idx_map_path, mtype):
    idx2type_df = pd.read_csv(idx_map_path)
    idx2type = idx2type_df["rel id" if mtype == "relation" else "entity id"]
    idx2type_set = set(idx2type)
    type_df = pd.read_csv(map_path)
    type_df = type_df.loc[type_df[mtype].isin(idx2type_set)]
    type_name, type_title = type_df[mtype].tolist(), type_df["title"].tolist()
    type_map = {x: t for x, t in zip(type_name, type_title)}
    idx2type = [type_map[x] if x in type_map else None for x in idx2type]
    return type_map, idx2type


def extract_english(wiki_root, output_csv_path, output_vocab_path, output_relation_path):
    print('Extracting English entities and relations from ConceptNet...')
    relation_map, idx2rel = load_maps(RELATION_MAP, RELTYPE_2_RELID, "relation")
    entity_map, idx2ent = load_maps(ENTITY_MAP, NODE_IDX_2_ENTITY_IDX, "entity")

    all_heads, all_relations, all_tails, all_weights = [], [], [], []

    edge_df = pd.read_csv(EDGE_PATH)
    edge_rel_df = pd.read_csv(EDGE_REL_PATH)
    heads, tails, relations = edge_df.iloc[:, 0].tolist(), edge_df.iloc[:, 1].tolist(), edge_rel_df.iloc[:, 0].tolist()
    unique_entities, unique_relations = set(), set()

    for h, r, t in tqdm(zip(heads, relations, tails), desc="Processing"):
        h_title, t_title = idx2ent[int(h)], idx2ent[int(t)]
        r_title = idx2rel[int(r)]
        if (h_title is not None and not pd.isna(h_title) and len(h_title.strip()) > 0) and \
           (t_title is not None and not pd.isna(t_title) and len(h_title.strip()) > 0) and \
           (r_title is not None and not pd.isna(r_title) and len(r_title.strip()) > 0):
            all_heads.append(h_title)
            all_relations.append(r_title)
            all_tails.append(t_title)
            all_weights.append(1)
            unique_entities.add(h_title)
            unique_entities.add(t_title)
            unique_relations.add(r_title)

    output_df = pd.DataFrame()
    output_df["relations"] = all_relations
    output_df["heads"] = all_heads
    output_df["tails"] = all_tails
    output_df["weights"] = all_weights

    output_df.to_csv(output_csv_path, header=False, sep="\t", index=False)

    save_list_as_text(list(unique_entities), output_vocab_path)
    save_list_as_text(list(unique_relations), output_relation_path)

    print(f'extracted Wiki csv file saved to {output_csv_path}')
    print(f'extracted entities saved to {output_vocab_path}')
    print()


def construct_graph(cpnet_csv_path, wiki_entity_path, wiki_relation_path, output_path, prune=True, reduced=False,
                    reduced_wiki_relation_path=None):
    print('generating ConceptNet graph file...')

    nltk.download('stopwords', quiet=True)
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    nltk_stopwords += ["like", "gone", "did", "going", "would", "could",
                       "get", "in", "up", "may", "wanter"]  # issue: mismatch with the stop words in grouding.py

    # blacklist = set(["uk", "us", "take", "make", "object", "person", "people"])  # issue: mismatch with the blacklist in grouding.py
    blacklist = []

    id2concept = load_text_as_list(wiki_entity_path)
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = load_text_as_list(wiki_relation_path)
    relation2id = {r: i for i, r in enumerate(id2relation)}

    reduced_id2relation = []
    reduced_relation2id = {}
    if reduced:
        reduced_id2relation = ["COMBINED"] + [id2relation[r] for r in REDUCED_RELATIONS]
        reduced_relation2id = {r: i for i, r in enumerate(reduced_id2relation)}

    graph = nx.MultiDiGraph()
    nrow = sum(1 for _ in open(cpnet_csv_path, 'r', encoding='utf-8'))
    with open(cpnet_csv_path, "r", encoding="utf8") as fin:

        def not_save(cpt):
            if cpt in blacklist:
                return True
            '''originally phrases like "branch out" would not be kept in the graph'''
            # for t in cpt.split("_"):
            #     if t in nltk_stopwords:
            #         return True
            return False

        attrs = set()

        for line in tqdm(fin, total=nrow):
            ls = line.strip().split('\t')
            if reduced:
                if relation2id[ls[0]] not in REDUCED_RELATION_SET:
                    rel = 0
                else:
                    rel = reduced_relation2id[ls[0]]
            else:
                rel = relation2id[ls[0]]

            for j in [1, 2]:
                if ls[j] not in concept2id:
                    if ls[j].startswith('"') and ls[j].endswith('"'):
                        ls[j] = ls[j][1:-1]
                        ls[j] = ls[j].replace('""', '"')
            subj = concept2id[ls[1]]
            obj = concept2id[ls[2]]
            weight = float(ls[3])
            if prune and (not_save(ls[1]) or not_save(ls[2]) or id2relation[rel] == "hascontext"):
                continue
            # if id2relation[rel] == "relatedto" or id2relation[rel] == "antonym":
            # weight -= 0.3
            # continue
            if subj == obj:  # delete loops
                continue
            # weight = 1 + float(math.exp(1 - weight))  # issue: ???

            if (subj, obj, rel) not in attrs:
                graph.add_edge(subj, obj, rel=rel, weight=weight)
                attrs.add((subj, obj, rel))
                graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                attrs.add((obj, subj, rel + len(relation2id)))

    nx.write_gpickle(graph, output_path)
    print(f"graph file saved to {output_path}")

    if reduced:
        save_list_as_text(reduced_id2relation, reduced_wiki_relation_path)
