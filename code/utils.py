import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


def calculate_rank(score, target, filter_list):
    score_target = score[target]
    score[filter_list] = score_target - 1
    # 使用 numpy 的广播机制加速
    rank = np.sum(score > score_target) + np.sum(score == score_target) // 2 + 1
    return rank


def metrics(rank):
    mr = np.mean(rank)
    mrr = np.mean(1 / rank)
    hit10 = np.sum(rank < 11) / len(rank)
    hit3 = np.sum(rank < 4) / len(rank)
    hit1 = np.sum(rank < 2) / len(rank)
    return mr, mrr, hit10, hit3, hit1


class TemperatureScaler(nn.Module):
    """
    使用验证集自动学习最佳温度参数 T，以最小化 NLL (Negative Log Likelihood)。
    这是校准模型概率分布的标准严谨方法 (Guo et al., 2017)。
    """

    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # 初始化为 1.5

    def forward(self, logits):
        return logits / self.temperature

    def set_temperature(self, logits, labels):
        """
        通过 LBFGS 优化器寻找最佳温度
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # 限制温度范围，防止极端值
        if self.temperature.item() < 0.1:
            self.temperature.data.fill_(0.1)

        return self.temperature.item()


class StratifiedConformalEngine:
    """
    支持分层（基于组）的自适应共形预测 (Stratified APS)。
    针对 HKG，根据 Qualifier 的数量将查询分组，分别计算阈值。
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.calibration_scores = {}  # Key: group_id, Value: list of scores
        self.q_hats = {}  # Key: group_id, Value: threshold

    def add_calibration_data(self, preds, answer_indices, group_ids, temperature=1.0):
        """
        preds: (batch_size, num_entities) logits
        answer_indices: list of list of correct answer indices
        group_ids: (batch_size, ) 每个样本所属的组 (例如 qualifier count)
        """
        # 1. 应用温度缩放并转为概率
        probs = F.softmax(preds / temperature, dim=1).cpu().numpy()

        # 2. 对每个样本计算 Conformity Score (APS score)
        # APS score 是正确答案所在位置的累积概率

        # 排序索引 (降序)
        sorted_indices = np.argsort(-probs, axis=1)

        n = len(answer_indices)
        for i in range(n):
            true_labels = answer_indices[i]
            gid = group_ids[i].item()  # 获取组ID

            # 获取排序后的概率
            sorted_probs = probs[i, sorted_indices[i]]

            # 计算累积概率
            cumsum_probs = np.cumsum(sorted_probs)

            # 找到正确答案在排序后的位置
            # 如果有多个正确答案，取最靠前的那个 (最乐观策略) 或者随机 (标准策略)，
            # 这里采用标准 APS 定义：找到包含所有正确答案所需的最小累积概率?
            # 通常 KG 补全是 One-vs-All，这里我们取所有正确答案中 ranking 最高的那个对应的累积概率作为 score。
            # 这是一种针对 Multi-label 的适配。

            # 简化版 APS for Multi-label:
            # 这里的任务通常被视为: "预测集合需要包含至少一个正确答案" 还是 "包含所有"？
            # 考虑到 Metrics (Hit@K) 的定义是 "是否存在正确答案在前K"，
            # 我们定义 Score 为：累积到遇到 *第一个* 正确答案时的概率质量。

            # 找到正确答案在 sorted_indices 中的位置
            ranks = []
            for label in true_labels:
                # np.where 返回的是 tuple
                r = np.where(sorted_indices[i] == label)[0][0]
                ranks.append(r)

            min_rank = min(ranks)  # 遇到第一个正确答案的位置

            # APS Score = 该位置的累积概率
            score = cumsum_probs[min_rank]

            if gid not in self.calibration_scores:
                self.calibration_scores[gid] = []
            self.calibration_scores[gid].append(score)

    def compute_thresholds(self):
        """
        对每个组分别计算 Q_hat
        """
        for gid, scores in self.calibration_scores.items():
            n = len(scores)
            if n == 0:
                continue
            # 分位数计算: ceil((n+1)(1-alpha)) / n
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(1.0, max(0.0, q_level))  # 截断

            # quantile 函数找的是 value，使得 P(X <= value) >= q_level
            # 在 APS 中，我们希望 Score <= Q_hat 的概率是 1-alpha
            # 所以我们要找的是 scores 的 (1-alpha) 分位数

            self.q_hats[gid] = np.quantile(scores, q_level, method='higher')

        return self.q_hats

    def predict_sets(self, preds, group_ids, temperature=1.0):
        """
        返回预测集合大小列表和覆盖情况
        """
        probs = F.softmax(preds / temperature, dim=1).cpu().numpy()
        sorted_indices = np.argsort(-probs, axis=1)

        set_sizes = []
        prediction_sets_mask = []  # 实际上不需要存具体的集合，只需要大小

        n = len(preds)
        for i in range(n):
            gid = group_ids[i].item()
            # 如果测试集中出现了校准集中没有的组（极少见），回退到默认组或全局平均（这里简单处理为使用最接近的 key 或报错，这里假设使用 Group 0 作为 fallback）
            q_hat = self.q_hats.get(gid, self.q_hats.get(0, 0.95))

            sorted_probs = probs[i, sorted_indices[i]]
            cumsum_probs = np.cumsum(sorted_probs)

            # 找到累积概率 <= Q_hat 的截断点
            # 预测集合包含累积概率达到 Q_hat 的所有实体
            # np.searchsorted 找到第一个 > q_hat 的索引
            cut_idx = np.searchsorted(cumsum_probs, q_hat)

            # 集合大小 = 索引 + 1 (因为索引从0开始)
            set_size = cut_idx + 1
            set_sizes.append(set_size)

        # 实际上我们还需要返回这个集合包含了哪些实体索引，或者直接评估是否覆盖
        # 这里为了性能，我们只返回 set_size，外层函数负责评估覆盖率
        # 因为外层知道正确答案

        return set_sizes, sorted_indices


from tqdm import tqdm
import logging


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)