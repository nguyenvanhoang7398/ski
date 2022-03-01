import os
import argparse
from multiprocessing import cpu_count
from utils.wiki import extract_english, construct_graph
from utils.grounding import create_matcher_patterns, ground, ski_ground
from utils.graph import generate_adj_data_from_grounded_concepts__use_LM, \
    ski_generate_adj_data_from_grounded_concepts__use_LM, \
    generate_entity_features


input_paths = {
    'wiki': {
        'csv': 'data/WikiKG90Mv2Dataset',
    },
    'docred': {
        'annotated-train': "./data/DocRED/train_annotated.json",
        'distant-train': "./data/DocRED/train_distant.json",
        'dev': './data/DocRED/dev.json',
        'test': './data/DocRED/test.json'
    }
}


output_paths = {
    'wiki': {
        'csv': './data/wiki/wiki.en.tsv',
        'vocab': './data/wiki/entities.txt',
        'relation': './data/wiki/relations.txt',
        'reduced-relation': './data/wiki/reduced_relations.txt',
        'patterns': './data/wiki/matcher_patterns.json',
        'unpruned-graph': './data/wiki/wiki.en.unpruned.graph',
        'pruned-graph': './data/wiki/wiki.en.pruned.graph',
        'reduced-unpruned-graph': './data/wiki/wiki.en.reduced.unpruned.graph',
        'reduced-pruned-graph': './data/wiki/wiki.en.reduced.pruned.graph',
        'entity-feature': './data/wiki/entity.npy'
    },
    'docred': {
        'grounded': {
            'annotated-train': './data/wiki/train_annotated.grounded.jsonl',
            'dev': './data/wiki/dev.grounded.jsonl',
            'test': './data/wiki/test.grounded.jsonl',
        },
        'graph': {
            'adj-annotated-train': './data/wiki/graph/train_annotated.graph.adj.pk',
            'adj-dev': './data/wiki/graph/dev.graph.adj.pk',
            'adj-test': './data/wiki/graph/test.graph.adj.pk',
            'reduced-adj-annotated-train': './data/wiki/graph/reduced_train_annotated.graph.adj.pk',
            'reduced-adj-dev': './data/wiki/graph/reduced_dev.graph.adj.pk',
            'reduced-adj-test': './data/wiki/graph/reduced_test.graph.adj.pk',
        },
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['wiki'], choices=['wiki'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'wiki': [
            # {'func': extract_english, 'args': (input_paths["wiki"]["csv"],
            #                                    output_paths['wiki']['csv'], output_paths['wiki']['vocab'],
            #                                    output_paths['wiki']['relation'])},
            # {'func': construct_graph, 'args': (output_paths['wiki']['csv'], output_paths['wiki']['vocab'],
            #                                    output_paths['wiki']['relation'],
            #                                    output_paths['wiki']['unpruned-graph'], False, False, None)},
            # {'func': construct_graph, 'args': (output_paths['wiki']['csv'], output_paths['wiki']['vocab'],
            #                                    output_paths['wiki']['relation'],
            #                                    output_paths['wiki']['pruned-graph'], True, False, None)},
            # {'func': construct_graph, 'args': (output_paths['wiki']['csv'], output_paths['wiki']['vocab'],
            #                                    output_paths['wiki']['relation'],
            #                                    output_paths['wiki']['reduced-unpruned-graph'], False, True,
            #                                    output_paths['wiki']['reduced-relation'])},
            # {'func': construct_graph, 'args': (output_paths['wiki']['csv'], output_paths['wiki']['vocab'],
            #                                    output_paths['wiki']['relation'],
            #                                    output_paths['wiki']['reduced-pruned-graph'], True, True,
            #                                    output_paths['wiki']['reduced-relation'])},
            # {'func': create_matcher_patterns,
            #  'args': (output_paths['wiki']['vocab'], output_paths['wiki']['patterns'])},
            # {'func': ski_ground(input_paths["docred"]["annotated-train"], output_paths['wiki']['vocab'],
            #                     output_paths['docred']['grounded']['annotated-train'], args.nprocs)},
            # {'func': ski_ground(input_paths["docred"]["dev"], output_paths['wiki']['vocab'],
            #                     output_paths['docred']['grounded']['dev'], args.nprocs)},
            # {'func': ski_ground(input_paths["docred"]["test"], output_paths['wiki']['vocab'],
            #                     output_paths['docred']['grounded']['test'], args.nprocs)},
            # {'func': ski_generate_adj_data_from_grounded_concepts__use_LM, 'args': (
            #     output_paths['docred']['grounded']['annotated-train'], output_paths['wiki']['pruned-graph'],
            #     output_paths['wiki']['vocab'], output_paths['wiki']['relation'],
            #     output_paths['docred']['graph']['adj-annotated-train'], args.nprocs)},
            # {'func': ski_generate_adj_data_from_grounded_concepts__use_LM, 'args': (
            #     output_paths['docred']['grounded']['dev'], output_paths['wiki']['pruned-graph'],
            #     output_paths['wiki']['vocab'], output_paths['wiki']['relation'],
            #     output_paths['docred']['graph']['adj-dev'], args.nprocs)},
            # {'func': ski_generate_adj_data_from_grounded_concepts__use_LM, 'args': (
            #     output_paths['docred']['grounded']['test'], output_paths['wiki']['pruned-graph'],
            #     output_paths['wiki']['vocab'], output_paths['wiki']['relation'],
            #     output_paths['docred']['graph']['adj-test'], args.nprocs)}
            # {'func': ski_generate_adj_data_from_grounded_concepts__use_LM, 'args': (
            #     output_paths['docred']['grounded']['annotated-train'], output_paths['wiki']['reduced-pruned-graph'],
            #     output_paths['wiki']['vocab'], output_paths['wiki']['reduced-relation'],
            #     output_paths['docred']['graph']['reduced-adj-annotated-train'], args.nprocs)},
            # {'func': ski_generate_adj_data_from_grounded_concepts__use_LM, 'args': (
            #     output_paths['docred']['grounded']['dev'], output_paths['wiki']['reduced-pruned-graph'],
            #     output_paths['wiki']['vocab'], output_paths['wiki']['reduced-relation'],
            #     output_paths['docred']['graph']['reduced-adj-dev'], args.nprocs)},
            # {'func': ski_generate_adj_data_from_grounded_concepts__use_LM, 'args': (
            #     output_paths['docred']['grounded']['test'], output_paths['wiki']['reduced-pruned-graph'],
            #     output_paths['wiki']['vocab'], output_paths['wiki']['reduced-relation'],
            #     output_paths['docred']['graph']['reduced-adj-test'], args.nprocs)},
            {'func': generate_entity_features, 'args': (output_paths["wiki"]['vocab'],
                                                        output_paths["wiki"]['relation'],
                                                        output_paths["wiki"]['entity-feature'])}
        ]
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
    # pass
