"""
Step 4：检索评估与调优 —— 本清单的交付核心

这一步才是"会根据结果来优化检索"的真身。前 3 步是"会用"，这步是"会调"。

学习目标：
  1. 建 Ground Truth 评测集
  2. 算 Recall@K / nDCG@K
  3. 扫参数画 Recall-Latency 曲线，找拐点
  4. 建立「召回差/排序差/延迟高」的分层排查反射

运行：python step4_evaluate.py
"""

import math
import time

import numpy as np
from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

from common import get_client, fake_embed, COLLECTION

client = get_client()

# =================================================================
# 1. Ground Truth 评测集
#    真实项目里这是人工标注：query -> 相关 chunk 列表（doc_id+offset 唯一标识）
#    学习阶段手工造，5 条足够看出问题
# =================================================================
EVAL_SET = [
    {"query": "公司新能源业务增长情况",
     "relevant": {("doc_a", 0), ("doc_a", 300), ("doc_c", 0)}},
    {"query": "隆基绿能",
     "relevant": {("doc_a", 600)}},
    {"query": "政策对新能源产业的影响",
     "relevant": {("doc_b", 0), ("doc_b", 400)}},
    {"query": "竞品为什么业绩下滑",
     "relevant": {("doc_c", 0), ("doc_c", 250)}},
    {"query": "价格战风险",
     "relevant": {("doc_a", 900), ("doc_b", 400)}},
]


def run_search(query, mode, limit=10, ef=64, rrf_k=60, weight=None, filter_expr=""):
    """统一检索入口，返回 [(doc_id, chunk_offset), ...] 有序列表"""
    q_vec = fake_embed(query)
    out = ["doc_id", "chunk_offset"]
    common = dict(collection_name=COLLECTION, limit=limit, output_fields=out)
    if filter_expr:
        common["filter"] = filter_expr

    if mode == "dense":
        hits = client.search(
            data=[q_vec], anns_field="dense_vector",
            search_params={"metric_type": "COSINE", "params": {"ef": ef}},
            **common)[0]
    elif mode == "bm25":
        hits = client.search(
            data=[query], anns_field="sparse_vector",
            search_params={"metric_type": "BM25"}, **common)[0]
    else:  # hybrid
        reqs = [
            AnnSearchRequest([q_vec], "dense_vector",
                             {"metric_type": "COSINE", "params": {"ef": ef}}, limit=limit),
            AnnSearchRequest([query], "sparse_vector",
                             {"metric_type": "BM25"}, limit=limit),
        ]
        ranker = WeightedRanker(*weight) if weight else RRFRanker(rrf_k)
        hits = client.hybrid_search(reqs=reqs, ranker=ranker, **common)[0]

    return [(h["entity"]["doc_id"], h["entity"]["chunk_offset"]) for h in hits]


# =================================================================
# 2. 指标
# =================================================================
def recall_at_k(retrieved, relevant, k):
    """前 k 条里召回了多少比例的相关项"""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def dcg_at_k(ranked, relevant, k):
    """折损累计增益：相关项排得越靠前贡献越大"""
    dcg = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved, relevant, k):
    """归一化 DCG：除以理想排序的 DCG"""
    ideal = dcg_at_k(list(relevant), relevant, k)
    return dcg_at_k(retrieved, relevant, k) / ideal if ideal else 0.0


def evaluate(mode, K=3, **kw):
    """跑完整评测集，返回 (平均Recall, 平均nDCG, 平均延迟ms)"""
    recalls, ndcgs, latencies = [], [], []
    for case in EVAL_SET:
        t0 = time.perf_counter()
        retrieved = run_search(case["query"], mode, limit=max(K, 10), **kw)
        latencies.append((time.perf_counter() - t0) * 1000)
        recalls.append(recall_at_k(retrieved, case["relevant"], K))
        ndcgs.append(ndcg_at_k(retrieved, case["relevant"], K))
    return np.mean(recalls), np.mean(ndcgs), np.mean(latencies)


# =================================================================
# 3. 三路对照 —— 用数据支撑"为什么走 hybrid"
# =================================================================
print("=" * 78)
print(f"{'模式':<24}{'Recall@3':<14}{'nDCG@3':<14}{'延迟(ms)':<12}")
print("=" * 78)

for mode, label in [("dense", "纯 Dense 语义"),
                    ("bm25", "纯 BM25 关键词"),
                    ("hybrid", "Hybrid + RRF(60)"),
                    ("hybrid", "Hybrid + Weighted(.5,.5)")]:
    kw = {"weight": (0.5, 0.5)} if "Weighted" in label else {}
    r, n, l = evaluate(mode, K=3, **kw)
    print(f"{label:<24}{r:<14.4f}{n:<14.4f}{l:<12.2f}")

# =================================================================
# 4. Recall-Latency 曲线 —— 找拐点（最实用的调优手段）
# =================================================================
print("\n" + "=" * 78)
print("扫 ef（HNSW 查询期参数）：找「再加大参数收益骤减」的拐点")
print("=" * 78)
print(f"{'ef':<8}{'Recall@3':<13}{'nDCG@3':<13}{'延迟(ms)':<12}{'边际收益'}")
print("-" * 78)

prev_r = prev_l = None
for ef in [16, 32, 64, 128, 256, 512]:
    r, n, l = evaluate("dense", K=3, ef=ef)
    note = ""
    if prev_r is not None and prev_l:
        gain = (r - prev_r) / max(prev_r, 1e-9)
        cost = (l - prev_l) / max(prev_l, 1e-9)
        if gain < 0.02 and cost > 0.5:
            note = f"← 拐点附近 (收益 {gain:+.1%} / 代价 {cost:+.1%})"
        else:
            note = f"(收益 {gain:+.1%} / 代价 {cost:+.1%})"
    print(f"{ef:<8}{r:<13.4f}{n:<13.4f}{l:<12.2f}{note}")
    prev_r, prev_l = r, l

# =================================================================
# 5. RRF k 参数扫描
# =================================================================
print("\n" + "=" * 78)
print("扫 RRF 的 k 参数")
print("=" * 78)
print(f"{'k':<8}{'Recall@3':<13}{'nDCG@3':<13}")
print("-" * 78)
for k in [1, 10, 30, 60, 100, 300]:
    r, n, _ = evaluate("hybrid", K=3, rrf_k=k)
    print(f"{k:<8}{r:<13.4f}{n:<13.4f}")

# =================================================================
# 6. 分层排查演练 —— 故意加严 filter，看召回怎么塌
# =================================================================
print("\n" + "=" * 78)
print("演练：filter 选择性过高导致召回塌陷（最常被误判成「索引参数问题」）")
print("=" * 78)
for label, fexpr in [
    ("无过滤", ""),
    ("只 doc_a", 'doc_id == "doc_a"'),
    ("只 opinion 维度", 'dimension == "opinion"'),
    ("只 meta 维度", 'dimension == "meta"'),
]:
    r, n, _ = evaluate("hybrid", K=3, filter_expr=fexpr)
    print(f"  {label:<16} Recall@3 = {r:.4f}   nDCG@3 = {n:.4f}")

# =================================================================
# 7. 逐 query 诊断 —— 定位"到底哪类 query 在拖后腿"
# =================================================================
print("\n" + "=" * 78)
print("逐 query 诊断（hybrid + RRF60，K=3）—— 找出问题 query")
print("=" * 78)
print(f"{'query':<26}{'Recall@3':<12}{'命中':<10}{'标注数':<10}")
print("-" * 78)
for case in EVAL_SET:
    ret = run_search(case["query"], "hybrid", limit=10)
    hit = len(set(ret[:3]) & case["relevant"])
    flag = "  ← 需排查" if hit < len(case["relevant"]) else ""
    print(f"{case['query']:<26}{recall_at_k(ret, case['relevant'], 3):<12.4f}"
          f"{hit:<10}{len(case['relevant']):<10}{flag}")


# =================================================================
# 【该观察什么 / 该得出什么结论】
#
#   1. 三路对照表：
#      - BM25 在 "隆基绿能" 型 query 上 Recall 高
#      - Dense 在语义型 query 上更好，专有名词型会掉
#      - Hybrid 综合两者，是最稳的
#      → 这就是本项目默认走 hybrid 的**数据依据**，不是拍脑袋
#
#   2. ef 扫描：
#      - 记录哪一行出现"收益 <2% 但代价 >50%"
#      → 结论：那个 ef 值就是你的拐点。真实数据集拐点更靠后，必须自己扫。
#        **绝不能抄别人的参数值**，拐点取决于数据分布 + 向量维度 + 数据量。
#
#   3. RRF k 扫描：
#      - 观察哪个 k 开始 nDCG 不再提升
#      → 结论：没有评估数据支撑就别动 k，保持 60。
#
#   4. filter 演练：
#      - "只 meta 维度"时 Recall 骤降
#      → 结论：这是**评测集与 filter 意图不匹配**，不是检索问题。
#        真实场景先问："这个 filter 和 query 意图匹配吗？"
#        不匹配的 filter，任何索引参数都救不了。
#
#   5. 逐 query 诊断：
#      - 标了"需排查"的 query 就是你下一步调优的目标
#      → 结论：先看它的 relevant 集合，判断是
#        (a) 标注错了 / (b) filter 不匹配 / (c) 真的没召回到
#        三种情况处理方式完全不同。
#
#   6. 分层排查反射（要背下来）：
#
#      召回差 ?
#        ├─ 先查 filter 是否与 query 语义匹配（最常见）
#        ├─ 再查 eval set 标注是否正确
#        ├─ 再看 embedding 模型是否适配领域
#        └─ 最后才调索引参数（ef / nprobe）
#
#      排序差 ?
#        ├─ 先看 Ranker 选型（RRF vs Weighted）
#        ├─ 再看是否需要加 Reranker（Qwen3-Reranker-0.6B）
#        └─ 最后调两路候选数 limit
#
#      延迟高 ?
#        ├─ 先查标量索引建没建（最常见，没建=全表扫描）
#        ├─ 再看 ef / nprobe 是否过大
#        ├─ 再看数据是否溢出到磁盘（mmap 状态）
#        └─ 最后看 segment 数量（大量小 segment 拖慢查询）
#
#      顺序不能乱。新手 90% 时间浪费在瞎调 ef，而问题其实在 filter。
# =================================================================
