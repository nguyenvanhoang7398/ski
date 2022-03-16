from modeling.modeling_encoder import MODEL_NAME_TO_CLASS, TextEncoder
from modeling.modeling_gnn import GNN
from utils.data_utils import load_input_tensors, load_sparse_adj_data_with_contextnode, MultiGPUSparseAdjDataBatchGenerator
import torch.nn as nn
import torch
from torch.nn import BCEWithLogitsLoss


class SkiDataLoader(object):
    def __init__(self, args, statement_path, label_map_path, kb_entity_path, adj_path,
                 batch_size, device, model_name, set_type, subsample=1.0, max_node_num=400, max_seq_length=512, max_ent_cnt=42,
                 max_mention_node_num=200):
        self.args = args
        self.batch_size = batch_size
        self.device0, self.device1 = device

        model_type = MODEL_NAME_TO_CLASS[model_name]
        # encoder_data = all_input_ids, all_attention_mask, all_token_type_ids,
        #                all_ent_mask, all_ent_ner, all_ent_pos, all_ent_distance, all_structure_mask, all_label_mask
        self.qids, self.labels, tok_kb_ent_edges, self.label_map, self.examples, *self.encoder_data = load_input_tensors(
            statement_path, label_map_path, kb_entity_path, model_type, model_name, max_seq_length, max_ent_cnt, set_type)
        # decoder_data = concept_ids, node_type_ids, node_scores, adj_lengths
        # adj_data (edge_index, edge_type)
        *self.decoder_data, self.adj_data, self.half_n_rel = load_sparse_adj_data_with_contextnode(
            adj_path, tok_kb_ent_edges, max_node_num, max_mention_node_num, args)

        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.data_size() * subsample)
            assert n_train > 0
            self.qids = self.qids[:n_train]
            self.labels = self.labels[:n_train]
            self.encoder_data = [x[:n_train] for x in self.encoder_data]
            self.decoder_data = [x[:n_train] for x in self.decoder_data]
            self.adj_data = self.adj_data[:n_train]
            assert all(len(self.qids) == len(self.adj_data[0]) == x.size(0) for x in
                       [self.labels] + self.encoder_data + self.decoder_data)

            assert self.data_size() == n_train

    def data_size(self):
        return len(self.qids)

    def generator(self, random=False):
        if random:
            indexes = torch.randperm(len(self.qids))
        else:
            indexes = torch.arange(len(self.qids))
        tag = "train" if random else "dev"
        return MultiGPUSparseAdjDataBatchGenerator(self.args, tag, self.device0, self.device1, self.batch_size,
                                                   indexes, self.qids, self.labels,
                                                   tensors0=self.encoder_data, tensors1=self.decoder_data,
                                                   adj_data=self.adj_data)


class Ski(nn.Module):
    def __init__(self, args, config, model_name, k, n_ntype, n_etype, n_labels,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.0, encoder_config={}, with_naive_feature=False):
        super().__init__()
        self.encoder = TextEncoder.from_pretrained(args.encoder_model_name_or_path,
                                                   from_tf=bool(".ckpt" in args.encoder_model_name_or_path),
                                                   cache_dir=None,
                                                   config=config,
                                                   model_name=model_name, **encoder_config)
        self.decoder = GNN(args, k, n_ntype, n_etype, config.hidden_size,
                           n_concept, concept_dim, concept_in_dim, n_attention_head,
                           fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                           pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                           init_range=init_range)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = n_labels
        self.max_ent_cnt = args.max_ent_cnt
        self.with_naive_feature = with_naive_feature
        self.feature_size = concept_dim
        if self.with_naive_feature:
            self.feature_size += 20
            self.distance_emb = nn.Embedding(20, 20, padding_idx=10)
        self.bili = torch.nn.Bilinear(self.feature_size, self.feature_size, self.num_labels)

    def batched_index_select(self, t, dim, inds):
        dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
        out = t.gather(dim, dummy)  # b x e x f
        return out

    def forward(self, *inputs, layer_id=-1, cache_output=False, detail=False, labels=None):
        # batch_tensors0 = encoder_data = all_input_ids, all_attention_mask, all_token_type_ids,
        #                  all_ent_mask, all_ent_ner, all_ent_pos, all_ent_distance, all_structure_mask, all_label_mask
        # batch_tesnors1 = decoder_data
        # inputs = batch_qids, batch_labels, *batch_tensors0, *batch_lists0, *batch_tensors1,
        #          *batch_lists1, edge_index, edge_type, batch_mention_toks
        bs, doc_len = inputs[0].size(0), inputs[0].size(1)

        # Here, merge the batch dimension and the num_choice dimension
        # edge_index_orig = (batch_size, 1, 2, n_edges)
        # edge_type_orig = (batch_size, 1, n_edges)
        edge_index_orig, edge_type_orig = inputs[-3:-1]
        _inputs = [x.view(x.size(0), *x.size()[1:]) for x in inputs[:-2]] + [sum(x, []) for x in inputs[-2:]]

        *lm_inputs, concept_ids, node_type_ids, node_scores, mention_toks, valid_node_mask, adj_lengths, edge_index, edge_type,  = _inputs
        # lm_inputs = input_ids, attention_mask, token_type_ids, ent_mask, position_ids, head_mask, inputs_embeds,
        #             ent_ner, ent_pos, ent_distance, structure_mask, label, label_mask, output_attentions
        batch_input_ids, batch_attn_mask, batch_token_type_ids, batch_ent_mask, batch_ent_ner, batch_ent_pos, \
            batch_ent_distance, batch_structure_mask, batch_label_mask = lm_inputs

        encoder_outputs = self.encoder(input_ids=batch_input_ids, attention_mask=batch_attn_mask,
                                       token_type_ids=batch_token_type_ids, ent_mask=batch_ent_mask,
                                       ent_ner=batch_ent_ner, ent_pos=batch_ent_pos, ent_distance=batch_ent_distance,
                                       structure_mask=batch_structure_mask, label_mask=batch_label_mask)

        encoder_device = batch_ent_mask.device
        decoder_device = node_type_ids.device
        # encoder_outputs = encoder_outputs.to(decoder_device)
        context_dim = encoder_outputs.shape[2]
        # IMPORTANT: add fake context state for padding
        zero_pad_state = torch.zeros((bs, 1, context_dim), dtype=encoder_outputs.dtype, device=encoder_device)
        pre_padded_encoder_outputs = torch.cat([zero_pad_state, encoder_outputs], dim=1)
        pre_padded_encoder_outputs = pre_padded_encoder_outputs.to(decoder_device)
        mention_context_states = self.batched_index_select(pre_padded_encoder_outputs, 1, mention_toks)

        # a: with kb, b: without kb
        a_indices, b_indices = [], []
        for i in range(bs):
            if torch.sum(edge_index[i]) == 0 and torch.sum(edge_type[i]) == 0:
                b_indices.append(i)
            else:
                a_indices.append(i)

        # edge_index = (batch_size, 2, n_edges)
        # edge_type = (batch_size, n_edges)
        # this means no kb information is provided
        # context_filled_output = torch.zeros(
        #         (bs, doc_len, self.decoder.concept_dim), dtype=encoder_outputs.dtype, device=encoder_device)

        # implement skip connection here as encoder(x) + decoder(encoder(x))

        # if len(b_indices) > 0:
        #     b_encoder_outputs = encoder_outputs[b_indices].to(encoder_device)
        #     b_encoder_outputs = self.encoder.dim_reduction(b_encoder_outputs).to(context_filled_output.dtype)
        #     b_encoder_outputs = torch.relu(b_encoder_outputs)
        #     context_filled_output[b_indices] = b_encoder_outputs

        encoder_x = self.encoder.dim_reduction(encoder_outputs).to(encoder_outputs.dtype)
        context_filled_output = torch.relu(encoder_x)

        if len(a_indices) > 0:
            a_edge_index, a_edge_type = [edge_index[i] for i in a_indices], [edge_type[i] for i in a_indices]
            a_concept_ids = concept_ids[a_indices]
            a_edge_index, a_edge_type = self.batch_graph(a_edge_index, a_edge_type, a_concept_ids.size(1))
            # a_edge_index = (2, edge numbers)
            a_node_type_ids = node_type_ids[a_indices]
            # edge_index: [2, total_E]   edge_type: [total_E, ]
            a_adj = (a_edge_index.to(a_node_type_ids.device), a_edge_type.to(a_node_type_ids.device))
            a_mention_context_states = mention_context_states[a_indices]
            a_valid_node_mask = valid_node_mask[a_indices]
            a_node_scores = node_scores[a_indices]
            a_adj_lengths = adj_lengths[a_indices]

            # use mention encoder as graph input
            gnn_output = self.decoder(a_mention_context_states, a_valid_node_mask, a_concept_ids,
                                      a_node_type_ids, a_node_scores, a_adj_lengths, a_adj,
                                      emb_data=None, cache_output=cache_output).to(encoder_device)
            a_context_filled_output = torch.zeros(
                (len(a_indices), doc_len, self.decoder.concept_dim), dtype=encoder_outputs.dtype, device=encoder_device)
            max_mention_toks = mention_toks.shape[1]
            context_gnn_output = gnn_output[:, :max_mention_toks, :]
            a_mention_toks = mention_toks[a_indices]
            for i in range(len(a_indices)):
                a_context_filled_output[i][a_mention_toks[i]] = context_gnn_output[i]
            context_filled_output[a_indices] = 0.5 * (context_filled_output[a_indices] + a_context_filled_output)

        ent_rep = torch.matmul(batch_ent_mask, context_filled_output)

        # prepare entity rep
        ent_rep_h = ent_rep.unsqueeze(2).repeat(1, 1, self.max_ent_cnt, 1)
        ent_rep_t = ent_rep.unsqueeze(1).repeat(1, self.max_ent_cnt, 1, 1)

        # concate distance feature
        if self.with_naive_feature:
            ent_rep_h = torch.cat([ent_rep_h, self.encoder.distance_emb(batch_ent_distance)], dim=-1)
            ent_rep_t = torch.cat([ent_rep_t, self.encoder.distance_emb(20 - batch_ent_distance)], dim=-1)

        ent_rep_h = self.dropout(ent_rep_h)
        ent_rep_t = self.dropout(ent_rep_t)
        logits = self.encoder.bili(ent_rep_h, ent_rep_t)

        loss_fct = BCEWithLogitsLoss(reduction='none')

        loss_all_ent_pair = loss_fct(logits.view(-1, self.num_labels), labels.float().view(-1, self.num_labels))
        # loss_all_ent_pair: [bs, max_ent_cnt, max_ent_cnt]
        # label_mask: [bs, max_ent_cnt, max_ent_cnt]
        loss_all_ent_pair = loss_all_ent_pair.view(-1, self.max_ent_cnt, self.max_ent_cnt, self.num_labels)
        loss_all_ent_pair = torch.mean(loss_all_ent_pair, dim=-1)
        loss_per_example = torch.sum(loss_all_ent_pair * batch_label_mask, dim=[1, 2]) / \
                           torch.sum(batch_label_mask, dim=[1, 2])
        loss = torch.mean(loss_per_example)

        logits = torch.sigmoid(logits)

        return logits, loss

    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]
        return edge_index, edge_type