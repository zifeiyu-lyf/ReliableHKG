import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
import copy
import time

class HKG(Dataset):
    def __init__(self, datasets_dir, dataset_name, logger, setting = 'Transductive', msg_add_tr = False):
        self.dataset_dir = os.path.join(datasets_dir, dataset_name)+"/"
        self.logger = logger
        self.msg_add_tr = msg_add_tr
        if self.msg_add_tr:
            assert setting == 'Inductive'
        
        logger.info("Loading Dataset...")

        self.ent2id_train = {}
        self.id2ent_train = []
        self.num_ent_train = 0

        self.ent2id_inf = {}
        self.id2ent_inf = []
        self.num_ent_inf = 0

        self.rel2id_train = {}
        self.rel_idxs_train = []
        self.id2rel_train = []
        self.num_rel_train = 0

        self.rel2id_inf = {}
        self.rel_idxs_inf = []
        self.id2rel_inf = []
        self.num_rel_inf = 0

        self.train = []
        self.inference = []
        self.valid = []
        self.test = []
        if setting == 'Transductive':
            self.parse_facts()
            self.inference = self.train
            self.ent2id_inf = self.ent2id_train
            self.id2ent_inf = self.id2ent_train
            self.num_ent_inf = self.num_ent_train

            self.rel2id_inf = self.rel2id_train
            self.rel_idxs_inf = self.rel_idxs_train
            self.id2rel_inf = self.id2rel_train
            self.num_rel_inf = self.num_rel_train

        elif setting == 'Inductive':
            self.parse_facts_ind()
        
        else:
            raise NotImplementedError

        logger.info("Dataset Loaded Successfully!")
        self.num_train = len(self.train)
        self.num_inf = len(self.inference)

        logger.info("Generating Dictionaries for Filtering...")
        self.eval_filter_dict = {}
        self.construct_eval_filter_dict()
        logger.info("Successfully Generated Filtering Dictionary!")

        logger.info("Model Input Construction...")
        self.pri_train = []
        self.qual_train = []
        self.qual2fact_train = []
        self.ent2rel_train = []
        self.ent_train = []
        self.ent2fact_train = []
        self.entloc_train = []
        self.fact2len_train = []
        self.fact2nument_train = []
        self.fact2qualstart_train = []
        self.fact2entstart_train = []
        self.fact2entlocstart_train = []

        self.pri_inf = []
        self.qual_inf = []
        self.qual2fact_inf = []
        self.ent2rel_inf = []
        self.ent_inf = []
        self.ent2fact_inf = []
        self.entloc_inf = []
        self.fact2len_inf = []
        self.fact2nument_inf = []
        self.fact2qualstart_inf = []
        self.fact2entstart_inf = []
        self.fact2entlocstart_inf = []

        self.construct_inputs()
        if setting == 'Transductive':
            self.pri_inf = self.pri_train
            self.qual_inf = self.qual_train
            self.qual2fact_inf = self.qual2fact_train
            self.ent2rel_inf = self.ent2rel_train
            self.ent_inf = self.ent_train
            self.ent2fact_inf = self.ent2fact_train
            self.entloc_inf = self.entloc_train
            self.fact2len_inf = self.fact2len_train
            self.fact2nument_inf = self.fact2nument_train
            self.fact2qualstart_inf = self.fact2qualstart_train
            self.fact2entstart_inf = self.fact2entstart_train
            self.fact2entlocstart_inf = self.fact2entlocstart_train
            self.rel_mask_inf = self.rel_mask_train

            self.hpair_inf = self.hpair_train
            self.fact2hpair_inf = self.fact2hpair_train
           
            self.tpair_inf = self.tpair_train
            self.fact2tpair_inf = self.fact2tpair_train

            self.qpair_inf = self.qpair_train
            self.qual2qpair_inf = self.qual2qpair_train
        
        elif setting == 'Inductive':
            self.construct_inputs_inf()

        else:
            raise NotImplementedError
            
        self.hpair_freq_inf = torch.bincount(self.fact2hpair_inf, minlength = len(self.hpair_inf))
        self.tpair_freq_inf = torch.bincount(self.fact2tpair_inf, minlength = len(self.tpair_inf))
        self.qpair_freq_inf = torch.bincount(self.qual2qpair_inf, minlength = len(self.qpair_inf))

        logger.info("Successfully Saved Model Inputs!")

        logger.info("Evaluation Query Generation...")

        self.valid_query = []
        self.valid_answer = []
        self.test_query = []
        self.test_answer = []
        self.construct_query()
        logger.info("Evaluation Query Generated Successfully!")

        
    def parse_facts(self):
        with open(self.dataset_dir + "train.txt") as f:
            for line in f.readlines():
                fact = []
                elements = line.strip().split("\t")
                for idx in range(len(elements)):
                    if idx % 2 == 0:
                        if elements[idx] not in self.ent2id_train:
                            self.id2ent_train.append(elements[idx])
                            self.ent2id_train[elements[idx]] = self.num_ent_train
                            self.num_ent_train += 1
                        fact.append(self.ent2id_train[elements[idx]])
                    else:
                        if elements[idx] not in self.rel2id_train:
                            self.rel_idxs_train.append(self.num_rel_train)
                            self.rel2id_train[elements[idx]] = self.num_rel_train
                            self.id2rel_train.append(elements[idx])
                            self.num_rel_train += 1
                        fact.append(self.rel2id_train[elements[idx]])
                self.train.append(fact)

        with open(self.dataset_dir + "valid.txt") as f:
            for line in f.readlines():
                fact = []
                elements = line.strip().split("\t")
                for idx in range(len(elements)):
                    if idx % 2 == 0:
                        if elements[idx] not in self.ent2id_train:
                            self.id2ent_train.append(elements[idx])
                            self.ent2id_train[elements[idx]] = self.num_ent_train
                            self.num_ent_train += 1
                        fact.append(self.ent2id_train[elements[idx]])
                    else:
                        if elements[idx] not in self.rel2id_train:
                            self.rel_idxs_train.append(self.num_rel_train)
                            self.rel2id_train[elements[idx]] = self.num_rel_train
                            self.id2rel_train.append(elements[idx])
                            self.num_rel_train += 1
                        fact.append(self.rel2id_train[elements[idx]])
                self.valid.append(fact)

        with open(self.dataset_dir + "test.txt") as f:
            for line in f.readlines():
                fact = []
                elements = line.strip().split("\t")
                for idx in range(len(elements)):
                    if idx % 2 == 0:
                        if elements[idx] not in self.ent2id_train:
                            self.id2ent_train.append(elements[idx])
                            self.ent2id_train[elements[idx]] = self.num_ent_train
                            self.num_ent_train += 1
                        fact.append(self.ent2id_train[elements[idx]])
                    else:
                        if elements[idx] not in self.rel2id_train:
                            self.rel_idxs_train.append(self.num_rel_train)
                            self.rel2id_train[elements[idx]] = self.num_rel_train
                            self.id2rel_train.append(elements[idx])
                            self.num_rel_train += 1
                        fact.append(self.rel2id_train[elements[idx]])
                self.test.append(fact)

    def parse_facts_ind(self):
        with open(self.dataset_dir + "train.txt") as f:
            for line in f.readlines():
                fact = []
                elements = line.strip().split("\t")
                for idx in range(len(elements)):
                    if idx % 2 == 0:
                        if elements[idx] not in self.ent2id_train:
                            self.id2ent_train.append(elements[idx])
                            self.ent2id_train[elements[idx]] = self.num_ent_train
                            self.num_ent_train += 1
                        fact.append(self.ent2id_train[elements[idx]])
                    else:
                        if elements[idx] not in self.rel2id_train:
                            self.rel_idxs_train.append(self.num_rel_train)
                            self.rel2id_train[elements[idx]] = self.num_rel_train
                            self.id2rel_train.append(elements[idx])
                            self.num_rel_train += 1
                        fact.append(self.rel2id_train[elements[idx]])
                self.train.append(fact)
        if self.msg_add_tr:
            with open(self.dataset_dir + "train.txt") as f:
                for line in f.readlines():
                    fact = []
                    elements = line.strip().split("\t")
                    for idx in range(len(elements)):
                        if idx % 2 == 0:
                            if elements[idx] not in self.ent2id_inf:
                                self.id2ent_inf.append(elements[idx])
                                self.ent2id_inf[elements[idx]] = self.num_ent_inf
                                self.num_ent_inf += 1
                            fact.append(self.ent2id_inf[elements[idx]])
                        else:
                            if elements[idx] not in self.rel2id_inf:
                                self.rel_idxs_inf.append(self.num_rel_inf)
                                self.rel2id_inf[elements[idx]] = self.num_rel_inf
                                self.id2rel_inf.append(elements[idx])
                                self.num_rel_inf += 1
                            fact.append(self.rel2id_inf[elements[idx]])
                    self.inference.append(fact)

        with open(self.dataset_dir + "msg.txt") as f:
            for line in f.readlines():
                fact = []
                elements = line.strip().split("\t")
                for idx in range(len(elements)):
                    if idx % 2 == 0:
                        if elements[idx] not in self.ent2id_inf:
                            self.id2ent_inf.append(elements[idx])
                            self.ent2id_inf[elements[idx]] = self.num_ent_inf
                            self.num_ent_inf += 1
                        fact.append(self.ent2id_inf[elements[idx]])
                    else:
                        if elements[idx] not in self.rel2id_inf:
                            self.rel_idxs_inf.append(self.num_rel_inf)
                            self.rel2id_inf[elements[idx]] = self.num_rel_inf
                            self.id2rel_inf.append(elements[idx])
                            self.num_rel_inf += 1
                        fact.append(self.rel2id_inf[elements[idx]])
                self.inference.append(fact)
        with open(self.dataset_dir + "valid.txt") as f:
            for line in f.readlines():
                fact = []
                elements = line.strip().split("\t")
                for idx in range(len(elements)):
                    if idx % 2 == 0:
                        fact.append(self.ent2id_inf[elements[idx]])
                    else:
                        fact.append(self.rel2id_inf[elements[idx]])
                self.valid.append(fact)
        
        with open(self.dataset_dir + "test.txt") as f:
            for line in f.readlines():
                fact = []
                elements = line.strip().split("\t")
                for idx in range(len(elements)):
                    if idx % 2 == 0:
                        fact.append(self.ent2id_inf[elements[idx]])
                    else:
                        fact.append(self.rel2id_inf[elements[idx]])
                self.test.append(fact)

    def construct_eval_filter_dict(self):
        for split in [self.inference, self.valid, self.test]:
            for fact in split:
                for idx in range(len(fact)):
                    corrupted_fact = fact[:idx] + [-1] + fact[idx+1:]
                    primary_triplet = tuple(corrupted_fact[:3])
                    qualifiers = []
                    for i in range(len(corrupted_fact[3::2])):
                        qualifiers.append(tuple(corrupted_fact[3+2*i:5+2*i]))
                    filter_key_list = [primary_triplet]
                    if len(qualifiers) != 0:
                        filter_key_list += sorted(qualifiers)
                    filter_key = tuple(filter_key_list)
                    if filter_key not in self.eval_filter_dict:
                        self.eval_filter_dict[filter_key] = []
                    self.eval_filter_dict[filter_key].append(fact[idx])

    def construct_query(self):
        vquery2idx = {}
        for fact_valid in self.valid:
            for idx in range(len(fact_valid)):
                if idx % 2 == 1 or (idx > 0 and not self.rel_mask_inf[fact_valid[idx-1]]):
                    continue
                corrupted_fact = fact_valid[:idx] + [-1] + fact_valid[idx+1:]
                primary_triplet = tuple(corrupted_fact[:3])
                qualifiers = []
                for i in range(len(corrupted_fact[3::2])):
                    qualifiers.append(tuple(corrupted_fact[3+2*i:5+2*i]))
                filter_key_list = [primary_triplet]
                if len(qualifiers) != 0:
                    filter_key_list += sorted(qualifiers)
                filter_key = tuple(filter_key_list)
                if filter_key not in vquery2idx:
                    vquery2idx[filter_key] = len(self.valid_query)
                    self.valid_query.append(filter_key)
                    self.valid_answer.append([fact_valid[idx]])
                else:
                    loc = vquery2idx[filter_key]
                    self.valid_answer[loc].append(fact_valid[idx])
        
        self.valid_answer = [a for _, a in sorted(zip(self.valid_query, self.valid_answer))]
        self.valid_query = sorted(self.valid_query)
        tquery2idx = {}
        for fact_test in self.test:
            for idx in range(len(fact_test)):
                if idx % 2 == 1 or (idx > 0 and not self.rel_mask_inf[fact_test[idx-1]]):
                    continue
                corrupted_fact = fact_test[:idx] + [-1] + fact_test[idx+1:]
                primary_triplet = tuple(corrupted_fact[:3])
                qualifiers = []
                for i in range(len(corrupted_fact[3::2])):
                    qualifiers.append(tuple(corrupted_fact[3+2*i:5+2*i]))
                filter_key_list = [primary_triplet]
                if len(qualifiers) != 0:
                    filter_key_list += sorted(qualifiers)
                filter_key = tuple(filter_key_list)
                if filter_key not in tquery2idx:
                    tquery2idx[filter_key] = len(self.test_query)
                    self.test_query.append(filter_key)
                    self.test_answer.append([fact_test[idx]])
                else:
                    loc = tquery2idx[filter_key]
                    self.test_answer[loc].append(fact_test[idx])
        self.test_answer = [a for _, a in sorted(zip(self.test_query, self.test_answer))]
        self.test_query = sorted(self.test_query)

    def construct_inputs(self):
        self.rel_mask_train = torch.zeros((self.num_rel_train,)).bool().cuda()
        self.rel_mask_train[self.rel_idxs_train] = 1
        hpair2idx = {}
        tpair2idx = {}
        qpair2idx = {}
        self.hpair_train = []
        self.fact2hpair_train = []
        self.tpair_train = []
        self.fact2tpair_train = []
        self.qpair_train = []
        self.qual2qpair_train = []
        for idx in range(self.num_train):
            fact = self.train[idx]
            self.pri_train.append([fact[0], fact[1], fact[2]])
            self.fact2len_train.append(len(fact))
            self.fact2nument_train.append(1)
            self.fact2entlocstart_train.append(len(self.entloc_train))
            self.fact2qualstart_train.append(len(self.qual_train))
            self.fact2entstart_train.append(len(self.ent_train))
            hpair = (fact[0], fact[1])
            tpair = (fact[2], fact[1])
            if hpair not in hpair2idx:
                hpair2idx[hpair] = len(self.hpair_train)
                self.hpair_train.append(list(hpair))
            if tpair not in tpair2idx:
                tpair2idx[tpair] = len(self.tpair_train)
                self.tpair_train.append(list(tpair))
            self.fact2hpair_train.append(hpair2idx[hpair])
            self.fact2tpair_train.append(tpair2idx[tpair])
            for i, ent in enumerate(fact[::2]):
                if i > 1:
                    qpair = (fact[i*2-1], fact[i*2])
                    self.qual2fact_train.append(idx)
                    self.qual_train.append(list(qpair))
                    if qpair not in qpair2idx:
                        qpair2idx[qpair] = len(self.qpair_train)
                        self.qpair_train.append(list(qpair))
                    self.qual2qpair_train.append(qpair2idx[qpair])
                if i > 0:
                    self.ent2rel_train.append(fact[i*2-1])
                    if self.rel_mask_train[fact[i*2-1]]:
                        self.fact2nument_train[-1] += 1
                        self.entloc_train.append(i*2)
                else:
                    self.ent2rel_train.append(fact[1])
                    self.entloc_train.append(i*2)
                self.ent_train.append(fact[i*2])
                self.ent2fact_train.append(idx)
        self.pri_train = torch.tensor(self.pri_train).cuda()
        self.qual2fact_train = torch.tensor(self.qual2fact_train).long().cuda()
        self.qual_train = torch.tensor(self.qual_train).cuda()
        self.ent2rel_train = torch.tensor(self.ent2rel_train).cuda()
        self.ent_train = torch.tensor(self.ent_train).cuda()
        self.ent2fact_train = torch.tensor(self.ent2fact_train).cuda()
        self.entloc_train = torch.tensor(self.entloc_train).cuda()
        self.fact2len_train = torch.tensor(self.fact2len_train).cuda()
        self.fact2nument_train = torch.tensor(self.fact2nument_train).cuda()
        self.fact2entstart_train = torch.tensor(self.fact2entstart_train).cuda()
        self.fact2qualstart_train = torch.tensor(self.fact2qualstart_train).cuda()
        self.fact2entlocstart_train = torch.tensor(self.fact2entlocstart_train).cuda()

        self.hpair_train = torch.tensor(self.hpair_train).cuda()
        self.fact2hpair_train = torch.tensor(self.fact2hpair_train).cuda()
        self.tpair_train = torch.tensor(self.tpair_train).cuda()
        self.fact2tpair_train = torch.tensor(self.fact2tpair_train).cuda()
        self.qpair_train = torch.tensor(self.qpair_train).cuda()
        self.qual2qpair_train = torch.tensor(self.qual2qpair_train).long().cuda()

    def construct_inputs_inf(self):
        self.rel_mask_inf = torch.zeros((self.num_rel_inf,)).bool().cuda()
        self.rel_mask_inf[self.rel_idxs_inf] = 1
        hpair2idx = {}
        tpair2idx = {}
        qpair2idx = {}
        self.hpair_inf = []
        self.fact2hpair_inf = []
        self.tpair_inf = []
        self.fact2tpair_inf = []
        self.qpair_inf = []
        self.qual2qpair_inf = []
        for idx in range(self.num_inf):
            fact = self.inference[idx]
            self.pri_inf.append([fact[0], fact[1], fact[2]])
            self.fact2len_inf.append(len(fact))
            self.fact2nument_inf.append(1)
            self.fact2entlocstart_inf.append(len(self.entloc_inf))
            self.fact2qualstart_inf.append(len(self.qual_inf))
            self.fact2entstart_inf.append(len(self.ent_inf))
            hpair = (fact[0], fact[1])
            tpair = (fact[2], fact[1])
            if hpair not in hpair2idx:
                hpair2idx[hpair] = len(self.hpair_inf)
                self.hpair_inf.append(list(hpair))
            if tpair not in tpair2idx:
                tpair2idx[tpair] = len(self.tpair_inf)
                self.tpair_inf.append(list(tpair))
            self.fact2hpair_inf.append(hpair2idx[hpair])
            self.fact2tpair_inf.append(tpair2idx[tpair])
            for i, ent in enumerate(fact[::2]):
                if i > 1:
                    qpair = (fact[i*2-1], fact[i*2])
                    self.qual2fact_inf.append(idx)
                    self.qual_inf.append(list(qpair))
                    if qpair not in qpair2idx:
                        qpair2idx[qpair] = len(self.qpair_inf)
                        self.qpair_inf.append(list(qpair))
                    self.qual2qpair_inf.append(qpair2idx[qpair])
                if i > 0:
                    self.ent2rel_inf.append(fact[i*2-1])
                    if self.rel_mask_inf[fact[i*2-1]]:
                        self.fact2nument_inf[-1] += 1
                        self.entloc_inf.append(i*2)
                else:
                    self.ent2rel_inf.append(fact[1])
                    self.entloc_inf.append(i*2)
                self.ent_inf.append(fact[i*2])
                self.ent2fact_inf.append(idx)
        self.pri_inf = torch.tensor(self.pri_inf).cuda()
        self.qual2fact_inf = torch.tensor(self.qual2fact_inf).long().cuda()
        self.qual_inf = torch.tensor(self.qual_inf).cuda()
        self.ent2rel_inf = torch.tensor(self.ent2rel_inf).cuda()
        self.ent_inf = torch.tensor(self.ent_inf).cuda()
        self.ent2fact_inf = torch.tensor(self.ent2fact_inf).cuda()
        self.entloc_inf = torch.tensor(self.entloc_inf).cuda()
        self.fact2len_inf = torch.tensor(self.fact2len_inf).cuda()
        self.fact2nument_inf = torch.tensor(self.fact2nument_inf).cuda()
        self.fact2entstart_inf = torch.tensor(self.fact2entstart_inf).cuda()
        self.fact2qualstart_inf = torch.tensor(self.fact2qualstart_inf).cuda()
        self.fact2entlocstart_inf = torch.tensor(self.fact2entlocstart_inf).cuda()

        self.hpair_inf = torch.tensor(self.hpair_inf).cuda()
        self.fact2hpair_inf = torch.tensor(self.fact2hpair_inf).cuda()
        self.tpair_inf = torch.tensor(self.tpair_inf).cuda()
        self.fact2tpair_inf = torch.tensor(self.fact2tpair_inf).cuda()
        self.qpair_inf = torch.tensor(self.qpair_inf).cuda()
        self.qual2qpair_inf = torch.tensor(self.qual2qpair_inf).long().cuda()

    def train_split(self, base_idxs):
        base_idxs, _ = base_idxs.sort()
        seen_ent = torch.zeros(self.num_ent_train).bool().cuda()
        seen_rel = torch.zeros(self.num_rel_train).bool().cuda()
        base_mask = torch.zeros(len(self.pri_train)).bool().cuda()
        base_mask[base_idxs] = True

        seen_ent[self.pri_train[base_idxs, 0]] = True
        seen_ent[self.pri_train[base_idxs, 2]] = True
        if len(self.qual_train) > 0:
            seen_ent[self.qual_train[base_mask[self.qual2fact_train], 1]] = True

        seen_rel[self.pri_train[base_idxs, 1]] = True
        if len(self.qual_train) > 0:
            seen_rel[self.qual_train[base_mask[self.qual2fact_train], 0]] = True

        query_mask = torch.ones(len(self.pri_train)).bool().cuda()
        query_mask[base_idxs] = False

        query_mask[torch.logical_not(seen_ent[self.pri_train[:, 0]])] = False       
        query_mask[torch.logical_not(seen_ent[self.pri_train[:, 2]])] = False
        if len(self.qual_train) > 0:
            query_mask[self.qual2fact_train[torch.logical_not(seen_ent[self.qual_train[:, 1]])]] = False

        query_mask[torch.logical_not(seen_rel[self.pri_train[:, 1]])] = False
        if len(self.qual_train) > 0:
            query_mask[self.qual2fact_train[torch.logical_not(seen_rel[self.qual_train[:, 0]])]] = False
        query_idxs = query_mask.nonzero(as_tuple = True)[0]

        pri = self.pri_train.clone().detach()
        qual = self.qual_train.clone().detach()

        hpair = self.hpair_train.clone().detach()
        fact2hpair = self.fact2hpair_train.clone().detach()
        tpair = self.tpair_train.clone().detach()
        fact2tpair = self.fact2tpair_train.clone().detach()
        qpair = self.qpair_train.clone().detach()
        qual2qpair = self.qual2qpair_train.clone().detach()

        idx2idx = 0 * torch.arange(self.num_train).cuda()
        idx2idx[base_idxs] = torch.arange(1, len(base_idxs) + 1).cuda()

        conv_ent = -1 * torch.ones(self.num_ent_train).long().cuda()
        conv_ent[seen_ent] = torch.arange(seen_ent.sum()).cuda()
        conv_rel = -1 * torch.ones(self.num_rel_train).long().cuda()
        conv_rel[seen_rel] = torch.arange(seen_rel.sum()).cuda()

        conv_pri = pri
        conv_pri[:, 0] = conv_ent[conv_pri[:, 0]]
        conv_pri[:, 1] = conv_rel[conv_pri[:, 1]]
        conv_pri[:, 2] = conv_ent[conv_pri[:, 2]]
        conv_qual2fact = idx2idx[self.qual2fact_train]
        conv_qual = qual
        if len(conv_qual) > 0:
            conv_qual[:, 0] = conv_rel[conv_qual[:, 0]]
            conv_qual[:, 1] = conv_ent[conv_qual[:, 1]]
        
        base_qual2fact_mask = (conv_qual2fact > 0)
        
        base_pri = conv_pri[idx2idx > 0]
        assert (base_pri[:, 0] == -1).sum() == 0
        base_qual2fact = conv_qual2fact[base_qual2fact_mask] - 1
        base_qual = conv_qual[base_qual2fact_mask]
        
        base_hpair_freq = torch.bincount(fact2hpair[idx2idx > 0], minlength = len(hpair))
        base_hpair = hpair[base_hpair_freq > 0]
        base_hpair_idx2idx = -1 * torch.ones(len(hpair)).long().cuda()
        base_hpair_idx2idx[base_hpair_freq > 0] = torch.arange(len(base_hpair)).cuda()
        base_hpair_freq = base_hpair_freq[base_hpair_freq > 0]
        base_fact2hpair = base_hpair_idx2idx[fact2hpair[idx2idx > 0]]
        base_hpair[:, 0] = conv_ent[base_hpair[:, 0]]
        base_hpair[:, 1] = conv_rel[base_hpair[:, 1]]

        base_tpair_freq = torch.bincount(fact2tpair[idx2idx > 0], minlength = len(tpair))
        base_tpair = tpair[base_tpair_freq > 0]
        base_tpair_idx2idx = -1 * torch.ones(len(tpair)).long().cuda()
        base_tpair_idx2idx[base_tpair_freq > 0] = torch.arange(len(base_tpair)).cuda()
        base_tpair_freq = base_tpair_freq[base_tpair_freq > 0]
        base_fact2tpair = base_tpair_idx2idx[fact2tpair[idx2idx > 0]]
        base_tpair[:, 0] = conv_ent[base_tpair[:, 0]]
        base_tpair[:, 1] = conv_rel[base_tpair[:, 1]]

        base_qpair_freq = torch.bincount(qual2qpair[conv_qual2fact > 0], minlength = len(qpair))
        base_qpair = qpair[base_qpair_freq > 0]
        base_qpair_idx2idx = -1 * torch.ones(len(qpair)).long().cuda()
        base_qpair_idx2idx[base_qpair_freq > 0] = torch.arange(len(base_qpair)).cuda()
        base_qpair_freq = base_qpair_freq[base_qpair_freq > 0]
        base_qual2qpair = base_qpair_idx2idx[qual2qpair[conv_qual2fact > 0]]
        if len(base_qpair) > 0:
            base_qpair[:, 0] = conv_rel[base_qpair[:, 0]]
            base_qpair[:, 1] = conv_ent[base_qpair[:, 1]]  

        num_base_ents = seen_ent.sum()
        num_base_rels = seen_rel.sum()

        return base_pri, base_qual, base_qual2fact, num_base_ents, num_base_rels, \
               base_hpair, base_hpair_freq, base_fact2hpair, \
               base_tpair, base_tpair_freq, base_fact2tpair, \
               base_qpair, base_qpair_freq, base_qual2qpair, \
               conv_ent, conv_rel, query_idxs

    def train_preds(self, conv_ent, conv_rel, query_idxs):
        query_idxs, _ = query_idxs.sort()

        pred_locs = (self.fact2nument_train[query_idxs] * torch.rand(len(query_idxs)).cuda()).type(torch.LongTensor).cuda()

        pred_locs = self.entloc_train[self.fact2entlocstart_train[query_idxs] + pred_locs]
        pred_idxs = query_idxs

        pri = self.pri_train.clone().detach()
        qual = self.qual_train.clone().detach()
        qual2fact = self.qual2fact_train.clone().detach()

        hpair = self.hpair_train.clone().detach()
        fact2hpair = self.fact2hpair_train.clone().detach()
        tpair = self.tpair_train.clone().detach()
        fact2tpair = self.fact2tpair_train.clone().detach()
        qpair = self.qpair_train.clone().detach()
        qual2qpair = self.qual2qpair_train.clone().detach()

        pred_ids = torch.arange(len(pred_locs)).cuda() + self.num_ent_train

        pri[pred_idxs[pred_locs == 0], 0] = pred_ids[pred_locs == 0]
        fact2hpair[pred_idxs[pred_locs == 0]] = len(hpair) + torch.arange((pred_locs == 0).sum()).cuda()
        hpair = torch.cat([hpair, pri[pred_idxs[pred_locs == 0], :2]], dim = 0)

        pri[pred_idxs[pred_locs == 2], 2] = pred_ids[pred_locs == 2]
        fact2tpair[pred_idxs[pred_locs == 2]] = len(tpair) + torch.arange((pred_locs == 2).sum()).cuda()
        tpair = torch.cat([tpair, pri[pred_idxs[pred_locs == 2], 1:].flip(dims = (-1, ))], dim = 0)

        qual_pred_locs = self.fact2qualstart_train[pred_idxs][pred_locs > 2] + pred_locs[pred_locs > 2]//2 - 2

        if len(qual) > 0:
            qual[qual_pred_locs, 1] = pred_ids[pred_locs > 2]
            qual2qpair[qual_pred_locs] = len(qpair) + torch.arange((pred_locs > 2).sum()).cuda()
            qpair = torch.cat([qpair, qual[qual_pred_locs]], dim = 0)

        answers = conv_ent[self.ent_train[self.fact2entstart_train[pred_idxs] + pred_locs//2]]

        idx2idx = 0 * torch.arange(self.num_train).cuda()
        idx2idx[query_idxs] = torch.arange(1, len(query_idxs) + 1).cuda()

        query_mask = torch.zeros(len(self.pri_train)).bool().cuda()
        query_mask[query_idxs] = True
        
        conv_ent = torch.cat([conv_ent, pred_ids - self.num_ent_train + (conv_ent != -1).sum()], dim = 0)

        conv_pri = pri
        conv_pri[:, 0] = conv_ent[conv_pri[:, 0]]
        conv_pri[:, 1] = conv_rel[conv_pri[:, 1]]
        conv_pri[:, 2] = conv_ent[conv_pri[:, 2]]
        conv_qual2fact = idx2idx[self.qual2fact_train]
        conv_qual = qual
        if len(conv_qual) > 0:
            conv_qual[:, 0] = conv_rel[conv_qual[:, 0]]
            conv_qual[:, 1] = conv_ent[conv_qual[:, 1]]

        query_qual2fact_mask = (conv_qual2fact > 0)

        query_pri = conv_pri[idx2idx > 0]
        query_qual2fact = conv_qual2fact[query_qual2fact_mask] - 1 
        query_qual = conv_qual[query_qual2fact_mask]

        query_hpair_freq = torch.bincount(fact2hpair[idx2idx > 0], minlength = len(hpair))
        query_hpair = hpair[query_hpair_freq > 0]
        query_hpair_idx2idx = -1 * torch.ones(len(hpair)).long().cuda()
        query_hpair_idx2idx[query_hpair_freq > 0] = torch.arange(len(query_hpair)).cuda()
        query_hpair_freq = query_hpair_freq[query_hpair_freq > 0]
        query_fact2hpair = query_hpair_idx2idx[fact2hpair[idx2idx > 0]]
        query_hpair[:, 0] = conv_ent[query_hpair[:, 0]]
        query_hpair[:, 1] = conv_rel[query_hpair[:, 1]]

        query_tpair_freq = torch.bincount(fact2tpair[idx2idx > 0], minlength = len(tpair))
        query_tpair = tpair[query_tpair_freq > 0]
        query_tpair_idx2idx = -1 * torch.ones(len(tpair)).long().cuda()
        query_tpair_idx2idx[query_tpair_freq > 0] = torch.arange(len(query_tpair)).cuda()
        query_tpair_freq = query_tpair_freq[query_tpair_freq > 0]
        query_fact2tpair = query_tpair_idx2idx[fact2tpair[idx2idx > 0]]
        query_tpair[:, 0] = conv_ent[query_tpair[:, 0]]
        query_tpair[:, 1] = conv_rel[query_tpair[:, 1]]

        query_qpair_freq = torch.bincount(qual2qpair[conv_qual2fact > 0], minlength = len(qpair))
        query_qpair = qpair[query_qpair_freq > 0]
        query_qpair_idx2idx = -1 * torch.ones(len(qpair)).long().cuda()
        query_qpair_idx2idx[query_qpair_freq > 0] = torch.arange(len(query_qpair)).cuda()
        query_qpair_freq = query_qpair_freq[query_qpair_freq > 0]
        query_qual2qpair = query_qpair_idx2idx[qual2qpair[conv_qual2fact > 0]]
        if len(query_qpair) > 0:
            query_qpair[:, 0] = conv_rel[query_qpair[:, 0]]
            query_qpair[:, 1] = conv_ent[query_qpair[:, 1]]

        return query_pri, query_qual, query_qual2fact, \
               query_hpair, query_hpair_freq, query_fact2hpair, \
               query_tpair, query_tpair_freq, query_fact2tpair, \
               query_qpair, query_qpair_freq, query_qual2qpair, answers


    def valid_inputs(self, idxs):
        fact_ids = torch.arange(len(idxs)).cuda()
        pris = []
        qual2fact = []
        quals = []

        hpairs = []
        hpair2idx = {}
        hpair_freqs = []
        fact2hpairs = []

        tpairs = []
        tpair2idx = {}
        tpair_freqs = []
        fact2tpairs = []

        qpairs = []
        qpair2idx = {}
        qpair_freqs = []
        qual2qpairs = []

        all_answers = []
        locs = []

        ent_idx = self.num_ent_inf
        for i, idx in enumerate(idxs):
            query = [list(comp) for comp in self.valid_query[idx]]
            pred_loc = sum(query, []).index(-1)
            pri = query[0]
            if pred_loc == 0:
                pri[0] = ent_idx
                ent_idx += 1
            elif pred_loc == 2:
                pri[2] = ent_idx
                ent_idx += 1
            elif pred_loc % 2 == 0:
                tmp_qual = query[pred_loc//2 - 1]
                tmp_qual[1] = ent_idx
                ent_idx += 1
                query[pred_loc//2 - 1] = tmp_qual
            else:
                raise NotImplementedError
            locs.append(pred_loc)
            pris.append(pri)

            hpair = (pri[0], pri[1])
            tpair = (pri[2], pri[1])
            if hpair not in hpair2idx:
                hpair2idx[hpair] = len(hpairs)
                hpairs.append(list(hpair))
                hpair_freqs.append(0)
            fact2hpairs.append(hpair2idx[hpair])
            hpair_freqs[hpair2idx[hpair]] += 1
            if tpair not in tpair2idx:
                tpair2idx[tpair] = len(tpairs)
                tpairs.append(list(tpair))
                tpair_freqs.append(0)
            fact2tpairs.append(tpair2idx[tpair])
            tpair_freqs[tpair2idx[tpair]] += 1

            for qual in query[1:]:
                quals.append(qual)
                qual2fact.append(fact_ids[i])
                qpair = tuple(qual)
                if qpair not in qpair2idx:
                    qpair2idx[qpair] = len(qpairs)
                    qpairs.append(list(qpair))
                    qpair_freqs.append(0)
                qual2qpairs.append(qpair2idx[qpair])
                qpair_freqs[qpair2idx[qpair]] += 1

            all_answers.append(self.eval_filter_dict[self.valid_query[idx]])


        return torch.tensor(pris).long().cuda(), \
               torch.tensor(quals).long().cuda(), \
               torch.tensor(qual2fact).long().cuda(), \
               torch.tensor(hpairs).long().cuda(), torch.tensor(hpair_freqs).long().cuda(), torch.tensor(fact2hpairs).long().cuda(), \
               torch.tensor(tpairs).long().cuda(), torch.tensor(tpair_freqs).long().cuda(), torch.tensor(fact2tpairs).long().cuda(), \
               torch.tensor(qpairs).long().cuda(), torch.tensor(qpair_freqs).long().cuda(), torch.tensor(qual2qpairs).long().cuda(), \
               all_answers, locs

    def test_inputs(self, idxs):
        fact_ids = torch.arange(len(idxs)).cuda()
        pris = []
        qual2fact = []
        quals = []

        hpairs = []
        hpair2idx = {}
        hpair_freqs = []
        fact2hpairs = []

        tpairs = []
        tpair2idx = {}
        tpair_freqs = []
        fact2tpairs = []

        qpairs = []
        qpair2idx = {}
        qpair_freqs = []
        qual2qpairs = []

        all_answers = []
        locs = []

        ent_idx = self.num_ent_inf
        for i, idx in enumerate(idxs):
            query = [list(comp) for comp in self.test_query[idx]]
            pred_loc = sum(query, []).index(-1)
            pri = query[0]
            if pred_loc == 0:
                pri[0] = ent_idx
                ent_idx += 1
            elif pred_loc == 2:
                pri[2] = ent_idx
                ent_idx += 1
            elif pred_loc % 2 == 0:
                tmp_qual = query[pred_loc//2 - 1]
                tmp_qual[1] = ent_idx
                ent_idx += 1
                query[pred_loc//2 - 1] = tmp_qual
            else:
                raise NotImplementedError
            locs.append(pred_loc)
            pris.append(pri)

            hpair = (pri[0], pri[1])
            tpair = (pri[2], pri[1])
            if hpair not in hpair2idx:
                hpair2idx[hpair] = len(hpairs)
                hpairs.append(list(hpair))
                hpair_freqs.append(0)
            fact2hpairs.append(hpair2idx[hpair])
            hpair_freqs[hpair2idx[hpair]] += 1
            if tpair not in tpair2idx:
                tpair2idx[tpair] = len(tpairs)
                tpairs.append(list(tpair))
                tpair_freqs.append(0)
            fact2tpairs.append(tpair2idx[tpair])
            tpair_freqs[tpair2idx[tpair]] += 1

            for qual in query[1:]:
                quals.append(qual)
                qual2fact.append(fact_ids[i])
                qpair = tuple(qual)
                if qpair not in qpair2idx:
                    qpair2idx[qpair] = len(qpairs)
                    qpairs.append(list(qpair))
                    qpair_freqs.append(0)
                qual2qpairs.append(qpair2idx[qpair])
                qpair_freqs[qpair2idx[qpair]] += 1

            all_answers.append(self.eval_filter_dict[self.test_query[idx]])

        return torch.tensor(pris).long().cuda(), \
               torch.tensor(quals).long().cuda(), \
               torch.tensor(qual2fact).long().cuda(), \
               torch.tensor(hpairs).long().cuda(), torch.tensor(hpair_freqs).long().cuda(), torch.tensor(fact2hpairs).long().cuda(), \
               torch.tensor(tpairs).long().cuda(), torch.tensor(tpair_freqs).long().cuda(), torch.tensor(fact2tpairs).long().cuda(), \
               torch.tensor(qpairs).long().cuda(), torch.tensor(qpair_freqs).long().cuda(), torch.tensor(qual2qpairs).long().cuda(), \
               all_answers, locs