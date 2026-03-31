import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Union, Annotated
from collections import OrderedDict


class Attn(nn.Module):
    def __init__(self, dim: int, num_head: int, dropout: float = 0):
        super(Attn, self).__init__()

        self.dim = dim
        self.num_head = num_head
        self.dim_per_head = dim // num_head
        self.drop = nn.Dropout(p=dropout)

        self.P = nn.Linear(dim, dim, bias=True)
        self.V = nn.Linear(dim, dim, bias=True)
        self.Q = nn.Linear(dim, dim, bias=True)
        self.K = nn.Linear(dim, dim, bias=True)
        self.a = nn.Parameter(torch.zeros((1, num_head, self.dim_per_head)))
        self.act = nn.PReLU()
        self.emb_ln = nn.LayerNorm(dim)

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.P.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.V.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.Q.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.K.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.a, gain=nn.init.calculate_gain('relu'))

        nn.init.zeros_(self.P.bias)
        nn.init.zeros_(self.V.bias)
        nn.init.zeros_(self.Q.bias)
        nn.init.zeros_(self.K.bias)

    def forward(self, query: torch.FloatTensor, keys: List[torch.FloatTensor], values: List[torch.FloatTensor],
                query_len: Union[int, torch.LongTensor], self_attn: bool = False,
                attn_idxs: Optional[List[torch.LongTensor]] = None,
                query_idxs: Optional[List[torch.LongTensor]] = None,
                key_idxs: Optional[List[torch.LongTensor]] = None,
                value_idxs: Optional[List[torch.LongTensor]] = None,
                pair_weights: Optional[List[torch.FloatTensor]] = None) -> torch.FloatTensor:
        if query_idxs is None:
            query_idxs = [None for _ in range(len(keys))]
        if key_idxs is None:
            key_idxs = [None for _ in range(len(keys))]
        if value_idxs is None:
            value_idxs = [None for _ in range(len(values))]
        if pair_weights is None:
            pair_weights = [None for _ in range(len(keys))]
        if attn_idxs is None:
            attn_idxs = [None for _ in range(len(keys))]
        if self_attn:
            attn_query_per_query = (self.a * self.act(self.Q(query) + self.K(query)).reshape(-1, self.num_head,
                                                                                             self.dim_per_head)).sum(
                dim=-1, keepdim=True)

        attn_query_per_key = []
        for key, query_idx, key_idx, attn_idx in zip(keys, query_idxs, key_idxs, attn_idxs):
            if (query_idx is not None and len(query) > 1) and key_idx is not None:
                if len(query) < len(query_idx) and len(key) < len(key_idx):
                    attn_query_per_key.append((self.a * self.act(
                        torch.index_select(self.Q(query), 0, query_idx) + torch.index_select(self.K(key), 0,
                                                                                             key_idx)).reshape(-1,
                                                                                                               self.num_head,
                                                                                                               self.dim_per_head)).sum(
                        dim=-1, keepdim=True))
                elif len(query) < len(query_idx) and len(key) >= len(key_idx):
                    attn_query_per_key.append((self.a * self.act(
                        torch.index_select(self.Q(query), 0, query_idx) + self.K(
                            torch.index_select(key, 0, key_idx))).reshape(-1, self.num_head, self.dim_per_head)).sum(
                        dim=-1, keepdim=True))
                elif len(query) >= len(query_idx) and len(key) < len(key_idx):
                    attn_query_per_key.append((self.a * self.act(
                        self.Q(torch.index_select(query, 0, query_idx)) + torch.index_select(self.K(key), 0,
                                                                                             key_idx)).reshape(-1,
                                                                                                               self.num_head,
                                                                                                               self.dim_per_head)).sum(
                        dim=-1, keepdim=True))
                else:
                    attn_query_per_key.append((self.a * self.act(
                        self.Q(torch.index_select(query, 0, query_idx)) + self.K(
                            torch.index_select(key, 0, key_idx))).reshape(-1, self.num_head, self.dim_per_head)).sum(
                        dim=-1, keepdim=True))

            elif (query_idx is None or len(query) == 1) and key_idx is not None:
                if len(key) < len(key_idx):
                    attn_query_per_key.append((self.a * self.act(
                        self.Q(query) + torch.index_select(self.K(key), 0, key_idx)).reshape(-1, self.num_head,
                                                                                             self.dim_per_head)).sum(
                        dim=-1, keepdim=True))
                else:
                    attn_query_per_key.append((self.a * self.act(
                        self.Q(query) + self.K(torch.index_select(key, 0, key_idx))).reshape(-1, self.num_head,
                                                                                             self.dim_per_head)).sum(
                        dim=-1, keepdim=True))

            elif (query_idx is not None and len(query) > 1) and key_idx is None:
                if len(query) < len(query_idx):
                    attn_query_per_key.append((self.a * self.act(
                        torch.index_select(self.Q(query), 0, query_idx) + self.K(key)).reshape(-1, self.num_head,
                                                                                               self.dim_per_head)).sum(
                        dim=-1, keepdim=True))
                else:
                    attn_query_per_key.append((self.a * self.act(
                        self.Q(torch.index_select(query, 0, query_idx)) + self.K(key)).reshape(-1, self.num_head,
                                                                                               self.dim_per_head)).sum(
                        dim=-1, keepdim=True))

            else:
                attn_query_per_key.append(
                    (self.a * self.act(self.Q(query) + self.K(key)).reshape(-1, self.num_head, self.dim_per_head)).sum(
                        dim=-1, keepdim=True))
            if attn_idx is not None:
                attn_query_per_key[-1] = attn_query_per_key[-1][attn_idx]

        if self_attn:
            max_attn_per_query = attn_query_per_query
        else:
            min_attn = min([torch.min(attn) for attn in attn_query_per_key if len(attn) > 0])
            max_attn_per_query = min_attn * torch.ones((query_len, self.num_head, 1)).cuda()
        for attn, query_idx in zip(attn_query_per_key, query_idxs):
            if query_idx is not None:
                max_attn_per_query = max_attn_per_query.index_reduce(dim=0, index=query_idx, source=attn, reduce='amax',
                                                                     include_self=True).detach()
            else:
                max_attn_per_query = torch.maximum(max_attn_per_query, attn).detach()

        if self_attn:
            exp_attn_query_per_query = torch.exp(attn_query_per_query - max_attn_per_query)
            exp_attn_per_query = exp_attn_query_per_query
        else:
            exp_attn_per_query = torch.zeros((query_len, self.num_head, 1)).cuda()
        exp_attn_query_per_key = []
        for attn, query_idx, pair_weight in zip(attn_query_per_key, query_idxs, pair_weights):
            if pair_weight is None:
                pair_weight = torch.ones_like(attn)
            if query_idx is not None:
                exp_attn_query_per_key.append(torch.exp(attn - max_attn_per_query[query_idx]))
                exp_attn_per_query = exp_attn_per_query.index_add(dim=0, index=query_idx,
                                                                  source=pair_weight * exp_attn_query_per_key[-1])
            else:
                exp_attn_query_per_key.append(torch.exp(attn - max_attn_per_query))
                exp_attn_per_query = exp_attn_per_query + pair_weight * exp_attn_query_per_key[-1]
        exp_attn_per_query = torch.where(exp_attn_per_query == 0, 1, exp_attn_per_query)

        if self_attn:
            msg_to_query = exp_attn_query_per_query * self.V(query).reshape(-1, self.num_head, self.dim_per_head)
        else:
            msg_to_query = torch.zeros((query_len, self.num_head, self.dim_per_head)).cuda()
        for exp_attn, query_idx, value, value_idx, pair_weight in zip(exp_attn_query_per_key, query_idxs, values,
                                                                      value_idxs, pair_weights):
            if pair_weight is None:
                pair_weight = torch.ones_like(exp_attn)
            if query_idx is not None and value_idx is not None:
                if len(value) < len(value_idx):
                    msg_to_query = msg_to_query.index_add(dim=0, index=query_idx,
                                                          source=pair_weight * exp_attn * torch.index_select(
                                                              self.V(value), 0, value_idx).reshape(-1, self.num_head,
                                                                                                   self.dim_per_head))
                else:
                    msg_to_query = msg_to_query.index_add(dim=0, index=query_idx,
                                                          source=pair_weight * exp_attn * self.V(
                                                              torch.index_select(value, 0, value_idx)).reshape(-1,
                                                                                                               self.num_head,
                                                                                                               self.dim_per_head))
            elif query_idx is None and value_idx is not None:
                if len(value) < len(value_idx):
                    msg_to_query = msg_to_query + pair_weight * exp_attn * torch.index_select(self.V(value), 0,
                                                                                              value_idx).reshape(-1,
                                                                                                                 self.num_head,
                                                                                                                 self.dim_per_head)
                else:
                    msg_to_query = msg_to_query + pair_weight * exp_attn * self.V(
                        torch.index_select(value, 0, value_idx)).reshape(-1, self.num_head, self.dim_per_head)
            elif query_idx is not None and value_idx is None:
                msg_to_query = msg_to_query.index_add(dim=0, index=query_idx,
                                                      source=pair_weight * exp_attn * self.V(value).reshape(-1,
                                                                                                            self.num_head,
                                                                                                            self.dim_per_head))
            else:
                msg_to_query = msg_to_query + pair_weight * exp_attn * self.V(value).reshape(-1, self.num_head,
                                                                                             self.dim_per_head)

        new_query = self.emb_ln(query + self.drop(self.P((msg_to_query / exp_attn_per_query).flatten(start_dim=1))))

        return new_query


class Init_Layer(nn.Module):
    def __init__(self, dim, logger, dropout=0.1):
        super(Init_Layer, self).__init__()

        self.dim = dim

        self.drop = nn.Dropout(p=dropout)

        self.logger = logger

        self.proj_he2e = nn.Linear(dim, dim, bias=True)
        self.proj_te2e = nn.Linear(dim, dim, bias=True)
        self.proj_qe2e = nn.Linear(dim, dim, bias=True)

        self.proj_he2pr = nn.Linear(dim, dim, bias=True)
        self.proj_te2pr = nn.Linear(dim, dim, bias=True)
        self.proj_qe2qr = nn.Linear(dim, dim, bias=True)

        self.proj_pr2he = nn.Linear(dim, dim, bias=True)
        self.proj_pr2te = nn.Linear(dim, dim, bias=True)
        self.proj_qr2qe = nn.Linear(dim, dim, bias=True)

        self.proj_pr2r = nn.Linear(dim, dim, bias=True)
        self.proj_qr2r = nn.Linear(dim, dim, bias=True)

        self.proj_fe2he = nn.Linear(dim, dim, bias=True)
        self.proj_fe2te = nn.Linear(dim, dim, bias=True)
        self.proj_fe2qe = nn.Linear(dim, dim, bias=True)

        self.proj_fr2pr = nn.Linear(dim, dim, bias=True)
        self.proj_fr2qr = nn.Linear(dim, dim, bias=True)

        self.emb_ent_ln = nn.LayerNorm(dim)
        self.emb_rel_ln = nn.LayerNorm(dim)

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.proj_he2e.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_te2e.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qe2e.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_he2pr.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_te2pr.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qe2qr.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_pr2he.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_pr2te.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qr2qe.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_pr2r.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qr2r.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_fe2he.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_fe2te.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_fe2qe.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_fr2pr.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_fr2qr.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.zeros_(self.proj_he2e.bias)
        nn.init.zeros_(self.proj_te2e.bias)
        nn.init.zeros_(self.proj_qe2e.bias)

        nn.init.zeros_(self.proj_he2pr.bias)
        nn.init.zeros_(self.proj_te2pr.bias)
        nn.init.zeros_(self.proj_qe2qr.bias)

        nn.init.zeros_(self.proj_pr2he.bias)
        nn.init.zeros_(self.proj_pr2te.bias)
        nn.init.zeros_(self.proj_qr2qe.bias)

        nn.init.zeros_(self.proj_pr2r.bias)
        nn.init.zeros_(self.proj_qr2r.bias)

        nn.init.zeros_(self.proj_fe2he.bias)
        nn.init.zeros_(self.proj_fe2te.bias)
        nn.init.zeros_(self.proj_fe2qe.bias)

        nn.init.zeros_(self.proj_fr2pr.bias)
        nn.init.zeros_(self.proj_fr2qr.bias)

    def forward(self, emb_ent, emb_rel, pri, qual2fact, qual):
        heads = pri[:, 0]
        rels = pri[:, 1]
        tails = pri[:, 2]
        if len(qual) > 0:
            qual_rels = qual[:, 0]
            qual_ents = qual[:, 1]
        else:
            qual_rels = torch.tensor([]).long().cuda()
            qual_ents = torch.tensor([]).long().cuda()

        msg_he2e = torch.index_select(self.proj_he2e(emb_ent), 0, heads)
        msg_te2e = torch.index_select(self.proj_te2e(emb_ent), 0, tails)
        msg_qe2e = torch.index_select(self.proj_qe2e(emb_ent), 0, qual_ents)
        msg_fe2e = (msg_he2e + msg_te2e).index_add(dim=0, index=qual2fact, source=msg_qe2e)
        fe_cnt = torch.bincount(qual2fact, minlength=len(pri)).float() + 2

        msg_he2pr = torch.index_select(self.proj_he2pr(emb_ent), 0, heads)
        msg_te2pr = torch.index_select(self.proj_te2pr(emb_ent), 0, tails)
        msg_qe2qr = torch.index_select(self.proj_qe2qr(emb_ent), 0, qual_ents)

        msg_pr2he = torch.index_select(self.proj_pr2he(emb_rel), 0, rels)
        msg_pr2te = torch.index_select(self.proj_pr2te(emb_rel), 0, rels)
        msg_qr2qe = torch.index_select(self.proj_qr2qe(emb_rel), 0, qual_rels)

        msg_pr2r = torch.index_select(self.proj_pr2r(emb_rel), 0, rels)
        msg_qr2r = torch.index_select(self.proj_qr2r(emb_rel), 0, qual_rels)
        msg_fr2r = msg_pr2r.index_add(dim=0, index=qual2fact, source=msg_qr2r)
        fr_cnt = fe_cnt - 1

        zero4ent = torch.zeros_like(emb_ent)

        idx4e = torch.cat([heads, tails], dim=0)
        src4e2e = torch.cat([self.proj_fe2he(msg_fe2e - msg_he2e), self.proj_fe2te(msg_fe2e - msg_te2e)], dim=0)
        src4e2ecnt = torch.cat([fe_cnt - 1, fe_cnt - 1], dim=0)  # exclude self
        src4r2e = torch.cat([msg_pr2he, msg_pr2te], dim=0)

        if len(qual2fact) > 0:
            idx4e = torch.cat([idx4e, qual_ents], dim=0)
            src4e2e = torch.cat([src4e2e, self.proj_fe2qe(torch.index_select(msg_fe2e, 0, qual2fact) - msg_qe2e)],
                                dim=0)
            src4e2ecnt = torch.cat([src4e2ecnt, torch.index_select(fe_cnt - 1, 0, qual2fact)], dim=0)
            src4r2e = torch.cat([src4r2e, msg_qr2qe], dim=0)

        e2e = zero4ent.index_add(dim=0, index=idx4e, source=src4e2e)
        e2e_cnt = torch.zeros(len(emb_ent)).cuda().index_add(dim=0, index=idx4e, source=src4e2ecnt).unsqueeze(dim=1)
        e2e_cnt = torch.where(e2e_cnt == 0, 1, e2e_cnt)

        r2e = zero4ent.index_reduce(dim=0, index=idx4e, source=src4r2e, reduce='mean', include_self=False)

        new_emb_ent = self.emb_ent_ln(emb_ent + self.drop(e2e / e2e_cnt) + self.drop(r2e))

        zero4rel = torch.zeros_like(emb_rel)

        if len(qual2fact) > 0:
            r2r = zero4rel.index_add(dim=0, index=torch.cat([rels, qual_rels], dim=0), \
                                     source=torch.cat([self.proj_fr2pr(msg_fr2r - msg_pr2r), self.proj_fr2qr(
                                         torch.index_select(msg_fr2r, 0, qual2fact) - msg_qr2r)], dim=0))

            r2r_cnt = torch.zeros(len(emb_rel)).cuda().index_add(dim=0, index=torch.cat([rels, qual_rels], dim=0), \
                                                                 source=torch.cat([fr_cnt - 1,
                                                                                   torch.index_select(fr_cnt - 1, 0,
                                                                                                      qual2fact)],
                                                                                  dim=0)).unsqueeze(
                dim=1)  # exclude self
            r2r_cnt = torch.where(r2r_cnt == 0, 1, r2r_cnt)

            e2r = zero4rel.index_reduce(dim=0, index=torch.cat([rels, rels, qual_rels], dim=0),
                                        source=torch.cat([msg_he2pr, msg_te2pr, msg_qe2qr], dim=0), reduce='mean',
                                        include_self=False)

            new_emb_rel = self.emb_rel_ln(emb_rel + self.drop(r2r / r2r_cnt) + self.drop(e2r))

        else:
            e2r = zero4rel.index_reduce(dim=0, index=torch.cat([rels, rels], dim=0),
                                        source=torch.cat([msg_he2pr, msg_te2pr], dim=0), reduce='mean',
                                        include_self=False)
            new_emb_rel = self.emb_rel_ln(emb_rel + self.drop(e2r))

        return new_emb_ent, new_emb_rel


class ANMP_Layer(nn.Module):
    def __init__(self, dim, num_head, logger, dropout=0.2):
        super(ANMP_Layer, self).__init__()

        self.dim = dim
        self.dim_per_head = dim // num_head
        assert self.dim_per_head * num_head == dim
        self.num_head = num_head

        self.drop = nn.Dropout(p=dropout)

        self.init_emb_fact = nn.Parameter(torch.zeros(1, dim))
        self.init_emb_fact_ln = nn.LayerNorm(dim)

        self.fact_ln = nn.LayerNorm(dim)

        ## Parameters for hyper-relational fact representation

        self.proj_head_ent = nn.Linear(dim, dim, bias=True)
        self.proj_tail_ent = nn.Linear(dim, dim, bias=True)
        self.proj_qual_ent = nn.Linear(dim, dim, bias=True)

        self.proj_head_rel = nn.Linear(dim, dim, bias=True)
        self.proj_tail_rel = nn.Linear(dim, dim, bias=True)
        self.proj_qual_rel = nn.Linear(dim, dim, bias=True)

        self.proj_hpair_to_fact = nn.Linear(dim, dim, bias=True)
        self.proj_tpair_to_fact = nn.Linear(dim, dim, bias=True)
        self.proj_qpair_to_fact = nn.Linear(dim, dim, bias=True)

        self.proj_head_rel2 = nn.Linear(dim, dim, bias=True)
        self.proj_tail_rel2 = nn.Linear(dim, dim, bias=True)
        self.proj_qual_rel2 = nn.Linear(dim, dim, bias=True)
        self.proj_head_ent2 = nn.Linear(dim, dim, bias=True)
        self.proj_tail_ent2 = nn.Linear(dim, dim, bias=True)
        self.proj_qual_ent2 = nn.Linear(dim, dim, bias=True)

        self.proj_fact_to_head_ent = nn.Linear(dim, dim, bias=True)
        self.proj_fact_to_tail_ent = nn.Linear(dim, dim, bias=True)
        self.proj_fact_to_pri_rel = nn.Linear(dim, dim, bias=True)
        self.proj_fact_to_qual_ent = nn.Linear(dim, dim, bias=True)
        self.proj_fact_to_qual_rel = nn.Linear(dim, dim, bias=True)

        self.ent_attn = Attn(dim, num_head, dropout=dropout)
        self.rel_attn = Attn(dim, num_head, dropout=dropout)

        self.to_fact_ln = nn.LayerNorm(dim)
        self.to_ent_ln = nn.LayerNorm(dim)
        self.to_rel_ln = nn.LayerNorm(dim)

        self.logger = logger

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.init_emb_fact, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_head_ent.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_tail_ent.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qual_ent.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_head_rel.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_tail_rel.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qual_rel.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_head_ent2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_tail_ent2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qual_ent2.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_head_rel2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_tail_rel2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qual_rel2.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_hpair_to_fact.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_tpair_to_fact.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_qpair_to_fact.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_normal_(self.proj_fact_to_head_ent.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_fact_to_tail_ent.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_fact_to_pri_rel.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_fact_to_qual_ent.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.proj_fact_to_qual_rel.weight, gain=nn.init.calculate_gain('relu'))

        nn.init.zeros_(self.proj_head_ent.bias)
        nn.init.zeros_(self.proj_tail_ent.bias)
        nn.init.zeros_(self.proj_qual_ent.bias)

        nn.init.zeros_(self.proj_head_rel.bias)
        nn.init.zeros_(self.proj_tail_rel.bias)
        nn.init.zeros_(self.proj_qual_rel.bias)

        nn.init.zeros_(self.proj_head_ent2.bias)
        nn.init.zeros_(self.proj_tail_ent2.bias)
        nn.init.zeros_(self.proj_qual_ent2.bias)

        nn.init.zeros_(self.proj_head_rel2.bias)
        nn.init.zeros_(self.proj_tail_rel2.bias)
        nn.init.zeros_(self.proj_qual_rel2.bias)

        nn.init.zeros_(self.proj_hpair_to_fact.bias)
        nn.init.zeros_(self.proj_tpair_to_fact.bias)
        nn.init.zeros_(self.proj_qpair_to_fact.bias)

        nn.init.zeros_(self.proj_fact_to_head_ent.bias)
        nn.init.zeros_(self.proj_fact_to_tail_ent.bias)
        nn.init.zeros_(self.proj_fact_to_pri_rel.bias)
        nn.init.zeros_(self.proj_fact_to_qual_ent.bias)
        nn.init.zeros_(self.proj_fact_to_qual_rel.bias)

    def forward(self, emb_ent, emb_rel, \
                pri, qual2fact, qual, \
                hpair_ent, hpair_rel, hpair_freq, fact2hpair, \
                tpair_ent, tpair_rel, tpair_freq, fact2tpair, \
                qpair_ent, qpair_rel, qpair_freq, qual2qpair):
        heads = pri[:, 0]
        rels = pri[:, 1]
        tails = pri[:, 2]
        if len(qual) > 0:
            qual_rels = qual[:, 0]
            qual_ents = qual[:, 1]
        else:
            qual_rels = torch.tensor([]).long().cuda()
            qual_ents = torch.tensor([]).long().cuda()

        emb_fact = self.drop(self.init_emb_fact_ln(self.init_emb_fact))

        emb_ent_head = self.proj_head_ent(emb_ent)
        emb_ent_tail = self.proj_tail_ent(emb_ent)
        if len(qual2fact) > 0:
            emb_ent_qual = self.proj_qual_ent(emb_ent)
        else:
            emb_ent_qual = torch.empty((0, self.dim)).cuda()

        emb_rel_head = self.proj_head_rel(emb_rel)
        emb_rel_tail = self.proj_tail_rel(emb_rel)
        if len(qual2fact) > 0:
            emb_rel_qual = self.proj_qual_rel(emb_rel)
        else:
            emb_rel_qual = torch.empty((0, self.dim)).cuda()

        emb_hpair_to_fact = self.to_fact_ln(self.proj_hpair_to_fact(
            torch.index_select(emb_rel_head, 0, hpair_rel) * torch.index_select(emb_ent_head, 0, hpair_ent)))
        emb_tpair_to_fact = self.to_fact_ln(self.proj_tpair_to_fact(
            torch.index_select(emb_rel_tail, 0, tpair_rel) * torch.index_select(emb_ent_tail, 0, tpair_ent)))
        emb_qpair_to_fact = self.to_fact_ln(self.proj_qpair_to_fact(
            torch.index_select(emb_rel_qual, 0, qpair_rel) * torch.index_select(emb_ent_qual, 0, qpair_ent)))

        new_emb_fact = torch.index_select(emb_hpair_to_fact, 0, fact2hpair) + torch.index_select(emb_tpair_to_fact, 0,
                                                                                                 fact2tpair)
        new_emb_fact = new_emb_fact.index_add(dim=0, index=qual2fact,
                                              source=torch.index_select(emb_qpair_to_fact, 0, qual2qpair))
        new_emb_fact = new_emb_fact / (torch.bincount(qual2fact, minlength=len(pri)) + 2).unsqueeze(dim=1)

        new_emb_fact = self.fact_ln(emb_fact + self.drop(new_emb_fact))

        emb_fact_to_head = self.to_ent_ln(
            self.proj_fact_to_head_ent(torch.index_select(self.proj_head_rel2(emb_rel), 0, rels) * new_emb_fact))
        emb_fact_to_tail = self.to_ent_ln(
            self.proj_fact_to_tail_ent(torch.index_select(self.proj_tail_rel2(emb_rel), 0, rels) * new_emb_fact))
        emb_fact_to_qual = self.to_ent_ln(self.proj_fact_to_qual_ent(
            torch.index_select(self.proj_qual_rel2(emb_rel), 0, qual_rels) * new_emb_fact[qual2fact]))

        new_emb_ent = self.ent_attn(query=emb_ent, keys=[emb_fact_to_head, emb_fact_to_tail, emb_fact_to_qual], \
                                    values=[emb_fact_to_head, emb_fact_to_tail, emb_fact_to_qual],
                                    query_len=len(emb_ent), self_attn=True, \
                                    query_idxs=[heads, tails, qual_ents])

        emb_fact_to_pri_rel = self.to_rel_ln(self.proj_fact_to_pri_rel((torch.index_select(self.proj_head_ent2(emb_ent),
                                                                                           0,
                                                                                           heads) + torch.index_select(
            self.proj_tail_ent2(emb_ent), 0, tails)) * new_emb_fact))
        emb_fact_to_qual_rel = self.to_rel_ln(self.proj_fact_to_qual_rel(
            torch.index_select(self.proj_qual_ent2(emb_ent), 0, qual_ents) * torch.index_select(new_emb_fact, 0,
                                                                                                qual2fact)))

        new_emb_rel = self.rel_attn(query=emb_rel, keys=[emb_fact_to_pri_rel, emb_fact_to_qual_rel], \
                                    values=[emb_fact_to_pri_rel, emb_fact_to_qual_rel], query_len=len(emb_rel),
                                    self_attn=True, \
                                    query_idxs=[rels, qual_rels])

        return new_emb_ent, new_emb_rel

    def pred(self, num_ent, num_rel, emb_ent, emb_rel, pri, qual2fact, qual, \
             hpair_ent, hpair_rel, hpair_freq, fact2hpair, \
             tpair_ent, tpair_rel, tpair_freq, fact2tpair, \
             qpair_ent, qpair_rel, qpair_freq, qual2qpair):
        heads = pri[:, 0]
        rels = pri[:, 1]
        tails = pri[:, 2]
        if len(qual) > 0:
            qual_rels = qual[:, 0]
            qual_ents = qual[:, 1]
        else:
            qual_rels = torch.tensor([]).long().cuda()
            qual_ents = torch.tensor([]).long().cuda()

        pred_heads = (heads >= num_ent).nonzero(as_tuple=True)[0]
        pred_tails = (tails >= num_ent).nonzero(as_tuple=True)[0]
        if len(qual) > 0:
            pred_qual_ents = (qual_ents >= num_ent).nonzero(as_tuple=True)[0]
        else:
            qual = qual.long()
            qual2fact = qual2fact.long()
            pred_qual_ents = torch.tensor([]).long().cuda()

        pred_ent = torch.zeros_like(heads)
        pred_ent[pred_heads] = 1
        pred_ent[pred_tails] = 1
        pred_ent[qual2fact[pred_qual_ents]] = 1
        pred_ent = pred_ent.nonzero(as_tuple=True)[0]

        emb_fact = self.drop(self.init_emb_fact_ln(self.init_emb_fact))

        emb_ent_head = self.proj_head_ent(emb_ent)
        emb_ent_tail = self.proj_tail_ent(emb_ent)
        if len(qual2fact) > 0:
            emb_ent_qual = self.proj_qual_ent(emb_ent)
        else:
            emb_ent_qual = torch.empty((0, self.dim)).cuda()

        emb_rel_head = self.proj_head_rel(emb_rel)
        emb_rel_tail = self.proj_tail_rel(emb_rel)
        if len(qual2fact) > 0:
            emb_rel_qual = self.proj_qual_rel(emb_rel)
        else:
            emb_rel_qual = torch.empty((0, self.dim)).cuda()

        emb_hpair_to_fact = self.to_fact_ln(self.proj_hpair_to_fact(
            torch.index_select(emb_rel_head, 0, hpair_rel) * torch.index_select(emb_ent_head, 0, hpair_ent)))
        emb_tpair_to_fact = self.to_fact_ln(self.proj_tpair_to_fact(
            torch.index_select(emb_rel_tail, 0, tpair_rel) * torch.index_select(emb_ent_tail, 0, tpair_ent)))
        emb_qpair_to_fact = self.to_fact_ln(self.proj_qpair_to_fact(
            torch.index_select(emb_rel_qual, 0, qpair_rel) * torch.index_select(emb_ent_qual, 0, qpair_ent)))

        new_emb_fact = torch.index_select(emb_hpair_to_fact, 0, fact2hpair) + torch.index_select(emb_tpair_to_fact, 0,
                                                                                                 fact2tpair)
        new_emb_fact = new_emb_fact.index_add(dim=0, index=qual2fact,
                                              source=torch.index_select(emb_qpair_to_fact, 0, qual2qpair))
        new_emb_fact = new_emb_fact / (torch.bincount(qual2fact, minlength=len(pri)) + 2).unsqueeze(dim=1)
        new_emb_fact = self.fact_ln(emb_fact + self.drop(new_emb_fact))

        emb_fact_to_head = self.to_ent_ln(self.proj_fact_to_head_ent(
            torch.index_select(self.proj_head_rel2(emb_rel), 0, rels[pred_heads]) * torch.index_select(new_emb_fact, 0,
                                                                                                       pred_heads)))
        emb_fact_to_tail = self.to_ent_ln(self.proj_fact_to_tail_ent(
            torch.index_select(self.proj_tail_rel2(emb_rel), 0, rels[pred_tails]) * torch.index_select(new_emb_fact, 0,
                                                                                                       pred_tails)))
        emb_fact_to_qual = self.to_ent_ln(self.proj_fact_to_qual_ent(
            torch.index_select(self.proj_qual_rel2(emb_rel), 0, qual_rels[pred_qual_ents]) * torch.index_select(
                new_emb_fact, 0, qual2fact[pred_qual_ents])))

        pred_ent_idxs = torch.arange(num_ent, len(emb_ent)).cuda()

        new_emb_ent = self.ent_attn(query=emb_ent, keys=[emb_ent, emb_fact_to_head, emb_fact_to_tail, emb_fact_to_qual], \
                                    values=[emb_ent, emb_fact_to_head, emb_fact_to_tail, emb_fact_to_qual],
                                    query_len=len(emb_ent), self_attn=False, \
                                    query_idxs=[pred_ent_idxs, heads[pred_heads], tails[pred_tails],
                                                qual_ents[pred_qual_ents]], \
                                    key_idxs=[pred_ent_idxs, None, None, None],
                                    value_idxs=[pred_ent_idxs, None, None, None])
        return new_emb_ent


class MAYPL(nn.Module):
    def __init__(self, dim, num_init_layer, num_layer, num_head, logger, model_dropout=0.2):
        super(MAYPL, self).__init__()
        init_layers = []
        layers = []

        for _ in range(num_init_layer):
            init_layers.append(Init_Layer(dim, logger, dropout=model_dropout))

        for _ in range(num_layer):
            layers.append(ANMP_Layer(dim, num_head, logger, dropout=model_dropout))

        self.init_layers = nn.ModuleList(init_layers)
        self.layers = nn.ModuleList(layers)

        self.dim = dim
        self.drop = nn.Dropout(p=model_dropout)

        self.init_emb_ent = nn.Parameter(torch.zeros(1, dim))
        self.init_emb_rel = nn.Parameter(torch.zeros(1, dim))

        self.logger = logger

        self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.init_emb_ent, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.init_emb_rel, gain=nn.init.calculate_gain('relu'))

    def forward(self, pri, qual, qual2fact, num_ent, num_rel, \
                hpair, hpair_freq, fact2hpair, \
                tpair, tpair_freq, fact2tpair, \
                qpair, qpair_freq, qual2qpair):

        hpair_ent = hpair[:, 0]
        hpair_rel = hpair[:, 1]
        hpair_freq = hpair_freq.unsqueeze(dim=1).unsqueeze(dim=2)

        tpair_ent = tpair[:, 0]
        tpair_rel = tpair[:, 1]
        tpair_freq = tpair_freq.unsqueeze(dim=1).unsqueeze(dim=2)

        if len(qual) > 0:
            qpair_ent = qpair[:, 1]
            qpair_rel = qpair[:, 0]
        else:
            qpair_ent = torch.tensor([]).long().cuda()
            qpair_rel = torch.tensor([]).long().cuda()
        qpair_freq = qpair_freq.unsqueeze(dim=1).unsqueeze(dim=2)

        emb_ent = self.init_emb_ent.repeat(num_ent, 1)
        emb_rel = self.init_emb_rel.repeat(num_rel, 1)

        init_embs_ent = [emb_ent]
        init_embs_rel = [emb_rel]
        for init_layer in self.init_layers:
            emb_ent, emb_rel = init_layer(emb_ent, emb_rel, pri, qual2fact, qual)
            init_embs_ent.append(emb_ent)
            init_embs_rel.append(emb_rel)

        init_emb_ent = emb_ent
        init_emb_rel = emb_rel

        emb_ents = [init_emb_ent]
        emb_rels = [init_emb_rel]

        for layer in self.layers:
            emb_ent, emb_rel = layer(emb_ent, emb_rel, \
                                     pri, qual2fact, qual, \
                                     hpair_ent, hpair_rel, hpair_freq, fact2hpair, \
                                     tpair_ent, tpair_rel, tpair_freq, fact2tpair, \
                                     qpair_ent, qpair_rel, qpair_freq, qual2qpair)
            emb_ents.append(emb_ent)
            emb_rels.append(emb_rel)
        return emb_ents, emb_rels, init_embs_ent, init_embs_rel

    def pred(self, pri, qual, qual2fact, \
             hpair, hpair_freq, fact2hpair, \
             tpair, tpair_freq, fact2tpair, \
             qpair, qpair_freq, qual2qpair, \
             emb_ents, emb_rels, init_embs_ent, init_embs_rel):
        # assumption: former indices are known entities, and latter indices are unknown entities

        hpair_ent = hpair[:, 0]
        hpair_rel = hpair[:, 1]
        hpair_freq = hpair_freq.unsqueeze(dim=1).unsqueeze(dim=2)

        tpair_ent = tpair[:, 0]
        tpair_rel = tpair[:, 1]
        tpair_freq = tpair_freq.unsqueeze(dim=1).unsqueeze(dim=2)

        if len(qual) > 0:
            qpair_ent = qpair[:, 1]
            qpair_rel = qpair[:, 0]
        else:
            qpair_ent = torch.tensor([]).long().cuda()
            qpair_rel = torch.tensor([]).long().cuda()
            qual2qpair = qual2qpair.long()
        qpair_freq = qpair_freq.unsqueeze(dim=1).unsqueeze(dim=2)

        num_ent = len(emb_ents[0])
        num_rel = len(emb_rels[0])

        max_ent_cands = [max(pri[:, 0]), num_ent - 1]
        max_ent_cands.append(max(pri[:, 2]))
        if len(qual) > 0:
            max_ent_cands.append(max(qual[:, 1]))

        num_ent_preds = max(max_ent_cands) - num_ent + 1

        emb_ent = torch.cat([init_embs_ent[0], self.init_emb_ent.repeat(num_ent_preds, 1)], dim=0)

        for l, init_layer in enumerate(self.init_layers):
            emb_ent, _ = init_layer(emb_ent, init_embs_rel[l], pri, qual2fact, qual)
            emb_ent = torch.cat([init_embs_ent[l + 1], emb_ent[num_ent:]], dim=0)

        input_emb_ent = emb_ent
        input_emb_rel = init_embs_rel[-1]
        for emb_ent, emb_rel, layer in zip(emb_ents[1:], emb_rels[1:], self.layers):
            output_emb_ent = layer.pred(num_ent, num_rel,
                                        input_emb_ent, input_emb_rel,
                                        pri, qual2fact, qual,
                                        hpair_ent, hpair_rel, hpair_freq, fact2hpair,
                                        tpair_ent, tpair_rel, tpair_freq, fact2tpair,
                                        qpair_ent, qpair_rel, qpair_freq, qual2qpair)

            input_emb_ent = torch.cat([emb_ent, output_emb_ent[num_ent:]], dim=0)
            input_emb_rel = emb_rel

        ent_preds = torch.inner(input_emb_ent[num_ent:], input_emb_ent[:num_ent])

        return ent_preds
