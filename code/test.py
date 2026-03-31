from dataloader import HKG
import importlib
from tqdm import tqdm
from utils import calculate_rank, metrics, TemperatureScaler, StratifiedConformalEngine
import torch.nn.functional as F
import numpy as np
import argparse
import torch
import torch.nn as nn
import datetime
import time
import os
import math
import random
from model import MAYPL
import logging
import copy

# ================= 环境设置 =================
os.environ['OMP_NUM_THREADS'] = '8'
torch.set_num_threads(8)
torch.cuda.empty_cache()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

# ================= 日志设置 =================
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)
logger.addHandler(stream_handler)

# ================= 参数设置 =================
parser = argparse.ArgumentParser()
parser.add_argument('--log_name')
parser.add_argument('--exp')
parser.add_argument('--dataset_name')
parser.add_argument('--test_epoch', type=int)
parser.add_argument('--msg_add_tr', action='store_true')
parser.add_argument('--data_dir', default="../data/", type=str)
parser.add_argument('--setting', default="Transductive", type=str)
parser.add_argument('--dim', default=256, type=int)
parser.add_argument('--val_size', default=100, type=int)
parser.add_argument('--num_init_layer', default=3, type=int)
parser.add_argument('--num_layer', default=4, type=int)
parser.add_argument('--num_head', default=32, type=int)
# [优化] 增加 Auto-Temperature 和 Stratified Conformal 开关
parser.add_argument('--enable_conformal', action='store_true', help='Enable Trustworthy Prediction')
parser.add_argument('--stratified', action='store_true', default=True,
                    help='Enable Structure-Aware Stratified Calibration')
parser.add_argument('--alpha', default=0.05, type=float, help='Error rate (e.g., 0.05 for 95% coverage)')
args = parser.parse_args()

# ================= 文件与模型初始化 =================
os.makedirs(f"./logs/{args.exp}/{args.dataset_name}", exist_ok=True)

file_format = args.log_name
suffix = "_test_trustworthy" if args.enable_conformal else "_test_standard"
file_handler = logging.FileHandler(
    f"./logs/{args.exp}/{args.dataset_name}/{file_format}{suffix}_{args.test_epoch}.log")
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

logger.info(f"PID: {os.getpid()}")
logger.info(f"Configuration: {vars(args)}")

default_answer = []
KG = HKG(args.data_dir, args.dataset_name, logger, setting=args.setting, msg_add_tr=args.msg_add_tr)
if args.msg_add_tr:
    orig_KG = HKG(args.data_dir, args.dataset_name, logger, setting=args.setting)
    for ent in orig_KG.ent2id_train:
        default_answer.append(KG.ent2id_inf[ent])

model = MAYPL(
    dim=args.dim,
    num_head=args.num_head,
    num_init_layer=args.num_init_layer,
    num_layer=args.num_layer,
    logger=logger
).cuda()

model.load_state_dict(
    torch.load(f"./ckpt/{args.exp}/{args.dataset_name}/{file_format}_{args.test_epoch}.ckpt")["model_state_dict"])

model.eval()


# ================= 辅助函数：提取 Qualifier 数量 =================
def get_qualifier_counts(qual2fact_tensor, batch_size):
    """
    计算 batch 中每个 fact 拥有的 qualifier 数量。
    qual2fact: (num_total_qualifiers, ) 存储每个 qualifier 属于哪个 fact index
    """
    if qual2fact_tensor.numel() == 0:
        return torch.zeros(batch_size, dtype=torch.long).cuda()

    # bincount 计算每个 fact index 出现的次数，即 qualifier 的数量
    # minlength 确保即使最后一个 fact 没有 qualifier 也能对齐 batch_size
    counts = torch.bincount(qual2fact_tensor, minlength=batch_size)

    # 截取前 batch_size 个 (防止索引越界，虽然理论上不应该)
    return counts[:batch_size]


# ================= 核心逻辑 =================
with torch.no_grad():
    # 1. 预计算 Embeddings
    emb_ent, emb_rel, init_embs_ent, init_embs_rel = model(
        KG.pri_inf.clone().detach(), KG.qual_inf.clone().detach(), KG.qual2fact_inf,
        KG.num_ent_inf, KG.num_rel_inf,
        KG.hpair_inf.clone().detach(), KG.hpair_freq_inf, KG.fact2hpair_inf,
        KG.tpair_inf.clone().detach(), KG.tpair_freq_inf, KG.fact2tpair_inf,
        KG.qpair_inf.clone().detach(), KG.qpair_freq_inf, KG.qual2qpair_inf
    )

    optimal_T = 1.0
    conformal_engine = None

    if args.enable_conformal:
        logger.info(">>> Phase 1: Auto-Temperature Scaling & Calibration <<<")

        # 收集 logits 和 labels 用于温度缩放
        all_logits_val = []
        all_labels_val = []

        # 收集数据用于 Conformal Calibration
        # 为了避免显存爆炸，我们分两步：先算 T，再算 Conformal

        # Step 1.1: 收集数据
        val_loader = torch.split(torch.arange(len(KG.valid_query)), args.val_size)
        for idxs in tqdm(val_loader, desc="Collecting Val Data"):
            query_pri, query_qual, query_qual2fact, \
            query_hpair, query_hpair_freq, query_fact2hpair, \
            query_tpair, query_tpair_freq, query_fact2tpair, \
            query_qpair, query_qpair_freq, query_qual2qpair, \
            answers, pred_locs = KG.valid_inputs(idxs)

            preds = model.pred(query_pri, query_qual, query_qual2fact,
                               query_hpair, query_hpair_freq, query_fact2hpair,
                               query_tpair, query_tpair_freq, query_fact2tpair,
                               query_qpair, query_qpair_freq, query_qual2qpair,
                               emb_ent, emb_rel, init_embs_ent, init_embs_rel)

            # 为温度缩放准备数据：因为这是 Multi-label 问题，
            # 标准做法是将其视为 Multi-class 问题的一对一 (One-vs-All) 或者只取第一个正确答案做近似优化
            # 这里我们取每个 Query 的第一个正确答案作为 Target 来优化 Temperature
            target_indices = []
            for q_idx in idxs:
                target_indices.append(KG.valid_answer[q_idx.item()][0])  # 取第一个

            all_logits_val.append(preds)
            all_labels_val.append(torch.tensor(target_indices).cuda())

        # Step 1.2: 学习最佳温度 T
        tensor_logits = torch.cat(all_logits_val)
        tensor_labels = torch.cat(all_labels_val)

        T_scaler = TemperatureScaler().cuda()
        optimal_T = T_scaler.set_temperature(tensor_logits, tensor_labels)
        logger.info(f"Optimal Temperature (T) learned: {optimal_T:.4f}")

        # 释放显存
        del tensor_logits, tensor_labels, all_logits_val, all_labels_val
        torch.cuda.empty_cache()

        # Step 1.3: Conformal Calibration (Stratified)
        conformal_engine = StratifiedConformalEngine(alpha=args.alpha)

        for idxs in tqdm(val_loader, desc="Calibrating"):
            query_pri, query_qual, query_qual2fact, \
            query_hpair, query_hpair_freq, query_fact2hpair, \
            query_tpair, query_tpair_freq, query_fact2tpair, \
            query_qpair, query_qpair_freq, query_qual2qpair, \
            answers, pred_locs = KG.valid_inputs(idxs)

            preds = model.pred(query_pri, query_qual, query_qual2fact,
                               query_hpair, query_hpair_freq, query_fact2hpair,
                               query_tpair, query_tpair_freq, query_fact2tpair,
                               query_qpair, query_qpair_freq, query_qual2qpair,
                               emb_ent, emb_rel, init_embs_ent, init_embs_rel)

            batch_answers = [KG.valid_answer[i.item()] for i in idxs]

            # [关键改进] 获取 Qualifier 数量作为组 ID
            if args.stratified:
                # 注意：query_qual2fact 是 batch 内所有 fact 的 qualifier 索引
                # 我们需要知道 batch_size
                qual_counts = get_qualifier_counts(query_qual2fact, len(idxs))
            else:
                qual_counts = torch.zeros(len(idxs), dtype=torch.long).cuda()

            conformal_engine.add_calibration_data(preds, batch_answers, qual_counts, temperature=optimal_T)

        q_hats = conformal_engine.compute_thresholds()
        logger.info(f"Calibration Complete. Thresholds (Q_hat) per group: {q_hats}")

    # 2. Phase 2: Testing
    logger.info(">>> Phase 2: Testing <<<")

    lp_head_list_rank = []
    lp_tail_list_rank = []
    lp_pri_list_rank = []
    lp_qual_list_rank = []
    lp_all_list_rank = []

    cp_coverage = []
    cp_set_sizes = []
    cp_group_stats = {}  # 统计不同组的表现

    test_loader = torch.split(torch.arange(len(KG.test_query)), args.val_size)
    for idxs in tqdm(test_loader, desc="Testing"):
        query_pri, query_qual, query_qual2fact, \
        query_hpair, query_hpair_freq, query_fact2hpair, \
        query_tpair, query_tpair_freq, query_fact2tpair, \
        query_qpair, query_qpair_freq, query_qual2qpair, \
        answers, pred_locs = KG.test_inputs(idxs)

        preds = model.pred(query_pri, query_qual, query_qual2fact,
                           query_hpair, query_hpair_freq, query_fact2hpair,
                           query_tpair, query_tpair_freq, query_fact2tpair,
                           query_qpair, query_qpair_freq, query_qual2qpair,
                           emb_ent, emb_rel, init_embs_ent, init_embs_rel)

        # --- Conformal Prediction Eval ---
        if args.enable_conformal:
            if args.stratified:
                qual_counts = get_qualifier_counts(query_qual2fact, len(idxs))
            else:
                qual_counts = torch.zeros(len(idxs), dtype=torch.long).cuda()

            set_sizes, sorted_indices = conformal_engine.predict_sets(preds, qual_counts, temperature=optimal_T)

            # 计算覆盖率
            batch_answers = [KG.test_answer[i.item()] for i in idxs]

            for i in range(len(idxs)):
                true_ans = set(batch_answers[i])
                pred_set_indices = sorted_indices[i][:set_sizes[i]]

                # 只要预测集中包含任意一个正确答案，就算 covered (Hit)
                is_covered = len(true_ans.intersection(set(pred_set_indices))) > 0

                cp_coverage.append(int(is_covered))
                cp_set_sizes.append(set_sizes[i])

                # 分组统计
                gid = qual_counts[i].item()
                if gid not in cp_group_stats:
                    cp_group_stats[gid] = {'cov': [], 'size': []}
                cp_group_stats[gid]['cov'].append(int(is_covered))
                cp_group_stats[gid]['size'].append(set_sizes[i])

        # --- Standard Ranking Eval ---
        preds_cpu = preds.detach().cpu().numpy()
        for i, idx in enumerate(idxs):
            pred_loc = pred_locs[i]
            answer = answers[i] + default_answer

            for test_answer in KG.test_answer[idx]:
                rank = calculate_rank(preds_cpu[i].copy(), test_answer, answer)
                if pred_loc <= 2:
                    lp_pri_list_rank.append(rank)
                if pred_loc == 0:
                    lp_head_list_rank.append(rank)
                elif pred_loc == 2:
                    lp_tail_list_rank.append(rank)
                else:
                    lp_qual_list_rank.append(rank)
                lp_all_list_rank.append(rank)

    # ================= 结果输出 (修正版) =================
    head_mr, head_mrr, head_hit10, head_hit3, head_hit1 = metrics(np.array(lp_head_list_rank))
    tail_mr, tail_mrr, tail_hit10, tail_hit3, tail_hit1 = metrics(np.array(lp_tail_list_rank))
    pri_ent_mr, pri_ent_mrr, pri_ent_hit10, pri_ent_hit3, pri_ent_hit1 = metrics(np.array(lp_pri_list_rank))

    logger.info("=" * 40)
    logger.info(f"Standard Metrics (Ranking):")
    logger.info("-" * 20)

    # [修正] 完整打印 Pri (主要三元组) 指标
    logger.info(f"Link Prediction (Pri, Count={len(lp_pri_list_rank)}):")
    logger.info(f"  MRR:    {pri_ent_mrr:.4f}")
    logger.info(f"  Hit@1:  {pri_ent_hit1:.4f}")
    logger.info(f"  Hit@3:  {pri_ent_hit3:.4f}")
    logger.info(f"  Hit@10: {pri_ent_hit10:.4f}")
    logger.info(f"  MR:     {pri_ent_mr:.2f}")

    # [补充] 头尾实体预测细分 (分析模型在头尾预测上的偏差)
    logger.info("-" * 20)
    logger.info(f"Detailed Breakdown:")
    logger.info(f"  Head Pred (MRR): {head_mrr:.4f} | Hit@1: {head_hit1:.4f} | Hit@10: {head_hit10:.4f}")
    logger.info(f"  Tail Pred (MRR): {tail_mrr:.4f} | Hit@1: {tail_hit1:.4f} | Hit@10: {tail_hit10:.4f}")

    if len(lp_qual_list_rank) > 0:
        qual_ent_mr, qual_ent_mrr, qual_ent_hit10, qual_ent_hit3, qual_ent_hit1 = metrics(np.array(lp_qual_list_rank))
        logger.info("-" * 20)
        logger.info(f"Link Prediction (Qualifier, Count={len(lp_qual_list_rank)}):")
        logger.info(f"  MRR: {qual_ent_mrr:.4f} | Hit@1: {qual_ent_hit1:.4f} | Hit@10: {qual_ent_hit10:.4f}")

    # --- Trustworthy Metrics ---
    if args.enable_conformal:
        logger.info("=" * 40)
        logger.info(f"Trustworthy Metrics (Conformal Prediction):")
        logger.info("-" * 20)
        logger.info(f"Parameters: Alpha={args.alpha} | Optimal T={optimal_T:.3f}")
        logger.info(f"Target Coverage: {1 - args.alpha:.2%}")
        logger.info(f"Actual Coverage: {np.mean(cp_coverage):.2%}")
        logger.info(f"Avg Set Size:    {np.mean(cp_set_sizes):.2f}")

        if args.stratified:
            logger.info("-" * 20)
            logger.info("Structure-Aware Breakdown (Stratified):")
            for gid in sorted(cp_group_stats.keys()):
                stats = cp_group_stats[gid]
                logger.info(
                    f"  [Qualifiers={gid}] Count: {len(stats['cov'])} | Coverage: {np.mean(stats['cov']):.2%} | Set Size: {np.mean(stats['size']):.2f}")

    logger.info("=" * 40)