import argparse
from datetime import datetime
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup
)

from dataset import docred_convert_examples_to_features as convert_examples_to_features
from dataset import DocREDProcessor

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, tokenizer, evaluate=False, predict=False):
    processor = DocREDProcessor()
    # Load data
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_map = processor.get_label_map(args.data_dir)

    if evaluate:
        examples = processor.get_dev_examples(args.data_dir)
    elif predict:
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_train_examples(args.data_dir)

    if evaluate or predict:
        cache_file = "evaluate_{}.cache".format(args.model_type)

        if os.path.exists(cache_file):
            print("Load evaluate features from", cache_file)
            features = torch.load(cache_file)
        else:
            features = convert_examples_to_features(
                examples,
                args.model_type,
                tokenizer,
                max_length=args.max_seq_length,
                max_ent_cnt=args.max_ent_cnt,
                label_map=label_map
            )
            torch.save(features, cache_file)
    else:
        features = convert_examples_to_features(
            examples,
            args.model_type,
            tokenizer,
            max_length=args.max_seq_length,
            max_ent_cnt=args.max_ent_cnt,
            label_map=label_map
        )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_ent_mask = torch.tensor([f.ent_mask for f in features], dtype=torch.float)
    all_ent_ner = torch.tensor([f.ent_ner for f in features], dtype=torch.long)
    all_ent_pos = torch.tensor([f.ent_pos for f in features], dtype=torch.long)
    all_ent_distance = torch.tensor([f.ent_distance for f in features], dtype=torch.long)
    all_structure_mask = torch.tensor([f.structure_mask for f in features], dtype=torch.bool)
    all_label = torch.tensor([f.label for f in features], dtype=torch.bool)
    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.bool)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                            all_ent_mask, all_ent_ner, all_ent_pos, all_ent_distance,
                            all_structure_mask, all_label, all_label_mask)

    return dataset



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_ent_cnt",
        default=42,
        type=int,
        help="The maximum entities considered.",
    )
    parser.add_argument("--no_naive_feature", action="store_true",
                        help="do not exploit naive features for DocRED, include ner tag, entity id, and entity pair distance")
    parser.add_argument("--entity_structure", default='biaffine', type=str, choices=['none', 'decomp', 'biaffine'],
                        help="whether and how do we incorporate entity structure in Transformer models.")
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run pred on the pred set.")
    parser.add_argument("--predict_thresh", default=0.51700735, type=float, help="pred thresh")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=30, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup ratio, overwriting warmup_steps.")
    parser.add_argument("--lr_schedule", default='linear', type=str, choices=['linear', 'constant'],
                        help="Linear warmup ratio, overwriting warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    args.exp_name = "{}-{}".format(args.model_type, datetime.now().strftime("%D-%H-%M-%S").replace("/", "_"))
    args.log_dir = "logs"
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    return args


def train(args, train_dataset, model, tokenizer):
    pass


def main():
    args = parse_args()

    # test with 1 gpu first
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.n_gpu = 1

    args.device = device
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    processor = DocREDProcessor()
    label_map = processor.get_label_map(args.data_dir)
    num_labels = len(label_map.keys())

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.do_train:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        pass

if __name__ == "__main__":
    main()