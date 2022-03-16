from datetime import datetime
import torch
import random
from torch import nn
try:
    from transformers import (AutoConfig, ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup
from utils.parser_utils import *
from utils.optimization_utils import OPTIMIZER_CLASSES
from modeling.modeling_ski import SkiDataLoader, Ski
import numpy as np
from tqdm import tqdm
from utils.io import *
from torch.utils.tensorboard import SummaryWriter

from modeling.modeling_parallel import SkiDataParallel



DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    'wiki': 1e-3,
    'obqa': 3e-4,
    'medqa_usmle': 1e-3,
}


def get_parsed_args():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    parser.add_argument('--do-train', action='store_true', help="run training")
    parser.add_argument('--do-eval', action='store_true', help="run evaluation")
    parser.add_argument('--do-test', action='store_true', help="run test")
    parser.add_argument('--eval-during-train', action='store_true', help="whether to evaluate during train")
    parser.add_argument('--parallel', action='store_true', help="whether to run parallel training")
    parser.add_argument('--baseline', action='store_true', help="whether to run the SSAN baseline")

    parser.add_argument('--save_dir', default=f'./saved_models/qagnn/', help='model output directory')
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.add_argument('--checkpoint_dir', default=None)

    # model
    parser.add_argument('--max_seq_length', default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--with_naive_feature', action='store_true')
    parser.add_argument('--predict_thresh', type=float, help="prediction threshold")

    # resources path
    parser.add_argument('--ent_emb_path', default=f'./data/wiki/entity_features.npy', help='path to entity embedding')
    parser.add_argument('--kb_entity_path', default=f'./data/wiki/entities.txt', help='path to entity list')

    parser.add_argument('--encoder_model_name_or_path', default='./pretrained_lm/roberta_base/', type=str,
                        help="Path to pre-trained model or shortcut name",)

    # data
    parser.add_argument('--num_relation', default=102, type=int, help='number of relations')
    parser.add_argument('--max_ent_cnt', default=42, type=int, help='number of max entity count')
    parser.add_argument('--train_adj', default=f'data/{args.dataset}/small_graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'data/{args.dataset}/small_graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'data/{args.dataset}/small_graph/test.graph.adj.pk')
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True,
                        help='use cached data to accelerate data loading')
    parser.add_argument('--label_map_path', default=f'data/{args.dataset}/label_map.json')

    # model architecture
    parser.add_argument('-k', '--k', default=5, type=int, help='perform k-layer message passing')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--gnn_dim', default=128, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True,
                        help='freeze entity embedding layer')

    parser.add_argument('--max_node_num', default=400, type=int)
    parser.add_argument('--simple', default=False, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('--init_range', default=0.02, type=float,
                        help='stddev when initializing with normal distribution')

    # regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float,
                        help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--fp16', default=False, type=bool_flag, help='use fp16 training. this requires torch>=1.6.0')
    parser.add_argument('--drop_partial_batch', default=False, type=bool_flag, help='')
    parser.add_argument('--fill_partial_batch', default=False, type=bool_flag, help='')

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='show this help message and exit')

    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(k=1)
    args = parser.parse_args()
    args.fp16 = args.fp16 and (torch.__version__ >= '1.6.0')
    return args


def train(args, dataloader, dev_dataloaer, model):
    args.seed = 42
    writer = SummaryWriter(os.path.join(args.exp_name))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,dev_acc,test_acc\n')

    # Load optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    if args.parallel:
        encoder_params = model.module.encoder.named_parameters()
        decoder_params = model.module.decoder.named_parameters()
    else:
        encoder_params = model.encoder.named_parameters()
        decoder_params = model.decoder.named_parameters()

    grouped_parameters = [
        {'params': [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in decoder_params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in decoder_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataloader.data_size() / args.batch_size))
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                        num_training_steps=max_steps)

    # Preparing parameters
    print('parameters:')
    for name, param in decoder_params:
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))
    num_params = sum(p.numel() for p in decoder_params if p.requires_grad)
    print('\ttotal:', num_params)

    best_model_path = f"{model_path}.best"
    print()
    print('-' * 71)
    if args.fp16:
        print('Using fp16 training')
        scaler = torch.cuda.amp.GradScaler()

    main_metric = "f1"
    global_step, best_dev_epoch = 0, 0
    best_dev_metric, final_test_acc, total_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    model.train()
    # freeze_net(model.encoder) # TODO: experiment with freezing encoder

    for epoch_id in range(args.n_epochs):
        # if epoch_id == args.unfreeze_epoch:
        #     unfreeze_net(model.encoder)
        # if epoch_id == args.refreeze_epoch:
        #     freeze_net(model.encoder)
        model.train()
        for qids, labels, *input_data in tqdm(dataloader.generator(random=True), desc="Training"): # TODO: change this back to True
            # input_data = *batch_tensors0, *batch_lists0, *batch_tensors1, *batch_lists1, edge_index, edge_type
            # batch_tensors0 = all_input_ids, all_attention_mask, all_token_type_ids, all_ent_mask,
            #                  all_ent_ner, all_ent_pos, all_ent_distance, all_structure_mask, all_label_mask
            optimizer.zero_grad()
            bs = labels.size(0)
            for a in range(0, bs, args.mini_batch_size):
                # divide the batch further into mini batches
                b = min(a + args.mini_batch_size, bs)
                if args.fp16:
                    with torch.cuda.amp.autocast():
                        logits, loss = model(*[x[a:b] for x in input_data],
                                             layer_id=args.encoder_layer, labels=labels[a:b])
                else:
                    logits, loss = model(*[x[a:b] for x in input_data],
                                         layer_id=args.encoder_layer, labels=labels[a:b])
                loss = loss * (b - a) / bs
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                total_loss += loss.item()
            if args.max_grad_norm > 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scheduler.step()
            if args.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if (global_step + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                current_lr = scheduler.get_lr()[0]
                print('| step {:5} |  lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step,
                                                                                              current_lr,
                                                                                              total_loss, ms_per_batch))
                train_results = {
                    "lr": current_lr,
                    "loss": total_loss
                }
                writer.add_scalars("train", train_results, global_step=global_step)
                total_loss = 0
                start_time = time.time()
            global_step += 1

        print("-" * 20)
        print("Epoch {}:".format(epoch_id))

        if args.eval_during_train:
            # train_results = evaluate(args, train_dataloader, model)
            # writer.add_scalars("train_eval", train_results, global_step=global_step)
            # print("-" * 20)
            # print("Train results")
            # print(train_results)
            dev_results = evaluate(args, dev_dataloaer, model)
            writer.add_scalars("dev_eval", dev_results, global_step=global_step)
            print("-" * 20)
            print("Dev results")
            print(dev_results)

            if dev_results[main_metric] > best_dev_metric:
                print("Update best eval model at epoch {} with {} = {}"
                      .format(epoch_id, main_metric, dev_results[main_metric]))
                best_dev_metric = dev_results[main_metric]
                if args.save_model:
                    torch.save([model.state_dict(), args], best_model_path)

    if best_dev_metric > 0.0:
        print("-" * 20)
        print("Finished training. best model at epoch {} with {} = {}")
        best_state_dict, best_args = torch.load(best_model_path)
        model.load_state_dict(best_state_dict)
    return model


def evaluate(args, dataloader, model):
    model.eval()

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None  # (n_examples, n_entities, n_entities, n_relations)
    ent_masks = None
    out_label_ids = None

    for qids, labels, *input_data in tqdm(dataloader.generator(random=False), desc="Evaluating"):
        if args.fp16:
            with torch.cuda.amp.autocast():
                logits, tmp_eval_loss = model(*input_data, layer_id=args.encoder_layer, labels=labels)
        else:
            logits, tmp_eval_loss = model(*input_data, layer_id=args.encoder_layer, labels=labels)

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        batch_ent_mask = input_data[3]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            ent_masks = batch_ent_mask.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            ent_masks = np.append(ent_masks, batch_ent_mask.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    dev_examples = dataloader.examples
    label_map = dataloader.label_map
    predicate_map = {}
    for predicate in label_map.keys():
        predicate_map[label_map[predicate]] = predicate

    total_labels = 0
    output_preds = []
    for (i, (example, pred, ent_mask)) in tqdm(
            enumerate(zip(dev_examples, preds, ent_masks)), desc="Post-process eval", total=len(dev_examples)):

        spo_gt_tmp = []
        for spo_gt in example.labels:
            spo_gt_tmp.append((spo_gt['h'], spo_gt['t'], spo_gt['r']))
        total_labels += len(spo_gt_tmp)  # total labels in an example

        for h in range(len(example.vertex_set)):
            for t in range(len(example.vertex_set)):
                if h == t:
                    continue
                if np.all(ent_mask[h] == 0) or np.all(ent_mask[t] == 0):
                    continue
                for predicate_id, logit in enumerate(pred[h][t]):
                    if predicate_id == 0:  # first predicate is "None"
                        continue
                    if (h, t, predicate_map[predicate_id]) in spo_gt_tmp:
                        flag = True
                    else:
                        flag = False
                    output_preds.append((flag, logit, example.title, h, t, predicate_map[predicate_id]))
    output_preds.sort(key=lambda x: x[1], reverse=True)
    pr_x = []
    pr_y = []
    correct = 0
    for i, pred in enumerate(
            output_preds):  # determine the best threshold to include the prediction of relation for best f1
        correct += pred[0]
        pr_y.append(float(correct) / (i + 1))
        pr_x.append(float(correct) / total_labels)

    pr_x = np.asarray(pr_x, dtype='float32')
    pr_y = np.asarray(pr_y, dtype='float32')
    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    thresh = output_preds[f1_pos][1]

    output_preds_thresh = []
    for i in range(f1_pos + 1):
        output_preds_thresh.append({"title": output_preds[i][2],
                                    "h_idx": output_preds[i][3],
                                    "t_idx": output_preds[i][4],
                                    "r": output_preds[i][5],
                                    "evidence": []
                                    })

    result = {"loss": eval_loss, "precision": pr_y[f1_pos],
              "recall": pr_x[f1_pos], "f1": f1, "thresh": thresh}
    print(result)

    return result


def load_model(args):
    cp_emb = np.load(args.ent_emb_path)
    cp_emb = torch.tensor(cp_emb, dtype=torch.float)
    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)

    if torch.cuda.device_count() >= 2 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
    elif torch.cuda.device_count() == 1 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")

    config = AutoConfig.from_pretrained(
        args.encoder_model_name_or_path,
        cache_dir=None,
    )

    encoder_configs = {
        "with_naive_feature": True,
        "entity_structure": "biaffine"
    }

    n_ntype = 3 # 0: context mention node, 1: kb node, 2: other node
    model = Ski(args, config, model_name=args.encoder, k=args.k, n_ntype=n_ntype, n_etype=args.num_relation,
                n_labels=len(pargs.label_map),
                n_concept=concept_num, concept_dim=args.gnn_dim, concept_in_dim=concept_dim,
                n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
                p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
                pretrained_concept_emb=cp_emb, freeze_ent_emb=args.freeze_ent_emb,
                init_range=args.init_range,
                encoder_config=encoder_configs, with_naive_feature=args.with_naive_feature)

    if args.checkpoint_dir:
        loaded_model_path = os.path.join(args.checkpoint_dir, "model.pt.best")
        print(f'loading and initializing model from {loaded_model_path}')
        model_state_dict, old_args = torch.load(loaded_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_state_dict)

    model.encoder.to(device0)
    model.distance_emb.to(device0)
    model.bili.to(device0)
    model.decoder.to(device1)

    return model


def predict(args, dataloader: SkiDataLoader, model):
    model.eval()

    nb_eval_steps = 0
    preds = None
    ent_masks = None
    out_label_ids = None
    all_attentions = []

    for qids, labels, *input_data in tqdm(dataloader.generator(random=False), desc="Evaluating"):
        if args.fp16:
            with torch.cuda.amp.autocast():
                logits, _ = model(*input_data, layer_id=args.encoder_layer, labels=labels)
        else:
            logits, _ = model(*input_data, layer_id=args.encoder_layer, labels=labels)

        nb_eval_steps += 1

        batch_ent_mask = input_data[3]

        if preds is None:
            preds = logits.detach().cpu().numpy()
            ent_masks = batch_ent_mask.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            ent_masks = np.append(ent_masks, batch_ent_mask.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    pred_examples = dataloader.examples
    label_map = dataloader.label_map
    predicate_map = {}
    for predicate in label_map.keys():
        predicate_map[label_map[predicate]] = predicate

    output_preds = []
    for (i, (example, pred, ent_mask)) in enumerate(zip(pred_examples, preds, ent_masks)):
        for h in range(len(example.vertex_set)):
            for t in range(len(example.vertex_set)):
                if h == t:
                    continue
                if np.all(ent_mask[h] == 0) or np.all(ent_mask[t] == 0):
                    continue
                for predicate_id, logit in enumerate(pred[h][t]):
                    if predicate_id == 0:
                        continue
                    else:
                        output_preds.append((logit, example.title, h, t, predicate_map[predicate_id]))
    output_preds.sort(key=lambda x: x[0], reverse=True)
    output_preds_thresh = []
    for i in range(len(output_preds)):
        if output_preds[i][0] < args.predict_thresh:
            break
        output_preds_thresh.append({"title": output_preds[i][1],
                                    "h_idx": output_preds[i][2],
                                    "t_idx": output_preds[i][3],
                                    "r": output_preds[i][4],
                                    "evidence": []
                                    })
    # write pred file
    if not os.path.exists('./data/DocRED/') and args.local_rank in [-1, 0]:
        os.makedirs('./data/DocRED')
    output_eval_file = os.path.join(args.checkpoint_dir, "result.json")
    with open(output_eval_file, 'w') as f:
        json.dump(output_preds_thresh, f)
    output_attention_file = os.path.join(args.checkpoint_dir, "attention.pickle")
    save_to_pickle(all_attentions, output_attention_file)


def get_data_loaders(args):
    if torch.cuda.device_count() >= 2 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
    elif torch.cuda.device_count() == 1 and args.cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")

    train_dl = SkiDataLoader(args=args, statement_path=args.train_statements, label_map_path=args.label_map_path,
                             kb_entity_path=args.kb_entity_path,
                             adj_path=args.train_adj, batch_size=args.batch_size, device=(device0, device1),
                             model_name=args.encoder,
                             max_node_num=args.max_node_num, max_seq_length=args.max_seq_length, set_type="train")
    dev_dl = SkiDataLoader(args=args, statement_path=args.dev_statements, label_map_path=args.label_map_path,
                           kb_entity_path=args.kb_entity_path,
                           adj_path=args.dev_adj, batch_size=args.eval_batch_size, device=(device0, device1),
                           model_name=args.encoder,
                           max_node_num=args.max_node_num, max_seq_length=args.max_seq_length, set_type="dev")
    test_dl = SkiDataLoader(args=args, statement_path=args.test_statements, label_map_path=args.label_map_path,
                            kb_entity_path=args.kb_entity_path,
                            adj_path=args.test_adj, batch_size=args.eval_batch_size, device=(device0, device1),
                            model_name=args.encoder,
                            max_node_num=args.max_node_num, max_seq_length=args.max_seq_length, set_type="test")

    args.label_map = train_dl.label_map

    return train_dl, dev_dl, test_dl


if __name__ == "__main__":
    pargs = get_parsed_args()
    if pargs.baseline:
        model_name = "ssan"
    else:
        model_name = "ski"
    pargs.exp_name = "{}-{}".format(model_name, datetime.now().strftime("%D-%H-%M-%S").replace("/", "_"))

    train_dataloader, validate_dataloader, test_dataloader = get_data_loaders(pargs)
    ski_model = load_model(pargs)

    if pargs.do_train:
        if pargs.parallel:
            ski_model = SkiDataParallel(ski_model, device_ids=list(range(8)))
        ski_model = train(pargs, train_dataloader, validate_dataloader, ski_model)
    else:
        loaded_state_dict, loaded_args = torch.load(os.path.join(pargs.checkpoint_dir, "model.pt.best"))
        ski_model.load_state_dict(loaded_state_dict)
    if pargs.do_eval:
        print("-" * 20)
        print("Evaluate best model:")
        eval_results = evaluate(pargs, validate_dataloader, ski_model)
    if pargs.do_test:
        print("-" * 20)
        print("Test best model:")
        predict(pargs, test_dataloader, ski_model)
