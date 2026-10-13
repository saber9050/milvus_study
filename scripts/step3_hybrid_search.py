"""
Step 3：Hybrid Search + RRF 融合 —— 本项目的检索主形态

学习目标：
  1. AnnSearchRequest 怎么构造多路请求
  2. RRFRanker vs WeightedRanker 的实际差异（重点，要亲眼看到）
  3. RRF 的 k 参数怎么影响结果
  4. 带标量过滤的 hybrid search 怎么写（★ 重点坑，见下）
  5. 各路候选数 limit 的影响

【★ 关键坑 —— hybrid 的过滤参数放哪】
  hybrid_search() 的签名里 **没有 filter 参数**：
      collection_name / reqs / ranker / limit / output_fields / timeout / partition_names
  如果你写 client.hybrid_search(..., filter='xxx')，
  它会进 **kwargs 被【静默丢弃】—— 不报错，但过滤完全不生效。
  这是最危险的坑：结果看起来"正常"，其实过滤根本没做事。

  正确写法：把 expr（或 filter）放进【每个】 AnnSearchRequest：
      AnnSearchRequest(data=[...], anns_field="dense_vector",
                       param={...}, limit=10,
                       expr='dimension == "opinion"')   # ← 这里
  注意 expr 和 filter 二选一，同时传会抛 ParamError。

运行：python step3_hybrid_search.py
"""

from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

from common import get_client, embed_cached_batch, print_hits, COLLECTION

client = get_client()

# 查询的问题
QUERY = "新能源业务的盈利能力和竞争风险"
q_vec = embed_cached_batch([QUERY])[0]

# 输出的字段（）
OUT = ["doc_id", "dimension", "chunk_offset", "content"]


def dense_req(limit=10, ef=64, expr=""):
    kw = dict(
        data=[q_vec],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"ef": ef}},
        limit=limit,
    )
    if expr:
        kw["expr"] = expr          # ★ 过滤写在这里，不是 hybrid_search 的顶层
    return AnnSearchRequest(**kw)


def bm25_req(query=QUERY, limit=10, expr=""):
    # 原文直接传，Milvus 用 content 的 analyzer 分词
    kw = dict(
        data=[query],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=limit,
    )
    if expr:
        kw["expr"] = expr
    return AnnSearchRequest(**kw)


def hsearch(ranker, limit=5, expr="", cand_limit=10, score_label=None):
    """两路 hybrid 检索。expr 必须传进每个 AnnSearchRequest 才生效"""
    hits = client.hybrid_search(
        collection_name=COLLECTION,
        reqs=[dense_req(limit=cand_limit, expr=expr),
              bm25_req(limit=cand_limit, expr=expr)],
        ranker=ranker,
        limit=limit,
        output_fields=OUT,
    )[0]
    return hits


# ---------------------------------------------------------------
# 1. RRF 融合 —— k 值的影响
#    公式：score = Σ 1/(k + rank_i)
# ---------------------------------------------------------------
print_hits("RRFRanker(k=60) —— Milvus 默认值",
           hsearch(RRFRanker(60)), score_label="RRF 融合分(无物理意义)")

print_hits("RRFRanker(k=1) —— 极端小值，排名第1的极度独裁",
           hsearch(RRFRanker(1)), score_label="RRF 融合分(无物理意义)")

print_hits("RRFRanker(k=1000) —— 极端大值，排名差异被抹平",
           hsearch(RRFRanker(1000)), score_label="RRF 融合分(无物理意义)")

# ---------------------------------------------------------------
# 2. WeightedRanker —— 需要归一化 + 手调权重
# ---------------------------------------------------------------
print_hits("WeightedRanker(0.7, 0.3) —— 偏语义",
           hsearch(WeightedRanker(0.7, 0.3)), score_label="加权归一化分")

print_hits("WeightedRanker(0.3, 0.7) —— 偏关键词",
           hsearch(WeightedRanker(0.3, 0.7)), score_label="加权归一化分")

# ---------------------------------------------------------------
# 3. ★ 带标量过滤的 hybrid —— 消费轮真实形态
#    对照组：先看不带过滤，再看带过滤，确认过滤真的生效
# ---------------------------------------------------------------
print_hits('【对照】Hybrid 无过滤（RRF60）',
           hsearch(RRFRanker(60)), score_label="RRF 融合分(无物理意义)")

print_hits('【★生效验证】Hybrid + expr: dimension == "opinion"',
           hsearch(RRFRanker(60), expr='dimension == "opinion"'),
           score_label="RRF 融合分(无物理意义)")

print_hits('【★生效验证】Hybrid + expr: doc_id == "doc_c"',
           hsearch(RRFRanker(60), expr='doc_id == "doc_c"'),
           score_label="RRF 融合分(无物理意义)")

print_hits('【★生效验证】Hybrid + expr: dimension == "meta"（几乎无相关内容）',
           hsearch(RRFRanker(60), expr='dimension == "meta"'),
           score_label="RRF 融合分(无物理意义)")

# 自检：确认过滤生效
print("\n" + "-" * 76)
print("【自动校验】检查带 expr 的结果是否全部满足过滤条件")
print("-" * 76)
for label, ex, field, val in [
    ('dimension == "opinion"', 'dimension == "opinion"', "dimension", "opinion"),
    ('doc_id == "doc_c"', 'doc_id == "doc_c"', "doc_id", "doc_c"),
]:
    hits = hsearch(RRFRanker(60), expr=ex)
    ok = all((h.get("entity", h).get(field) == val) for h in hits if hits)
    mark = "✅ 过滤生效" if ok else "❌ 过滤未生效！"
    print(f"  {label:<26} 返回 {len(hits)} 条 → {mark}")

# ---------------------------------------------------------------
# 4. 各路候选数 limit 的影响
#    limit 是"每路先各取多少条候选"参与融合，不是最终返回条数
# ---------------------------------------------------------------
print("\n" + "=" * 76)
print("各路候选数 limit 的影响（每路先各取 limit 条，再融合取 top5）")
print("=" * 76)
print(f"{'各路候选数':<14}{'最终 top5 (doc_id/维度/offset)'}")
print("-" * 76)
for cl in [2, 5, 10]:
    hits = hsearch(RRFRanker(60), limit=5, cand_limit=cl)
    brief = " | ".join(
        f"{h.get('entity', h).get('doc_id')}/{h.get('entity', h).get('dimension')}"
        f"@{h.get('entity', h).get('chunk_offset')}"
        for h in hits
    )
    print(f"{cl:<14}{brief}")

# 候选数过小时，某些路凑不满，融合效果下降
for cl in [1, 2]:
    hits = hsearch(RRFRanker(60), limit=5, cand_limit=cl)
    print(f"  [i] 候选数={cl} 时实际返回 {len(hits)} 条"
          f"（候选池太小，两路交集少，hybrid 增益下降）")


# ---------------------------------------------------------------
# 【该观察什么 / 该得出什么结论】
#
#   1. RRF 的 k 值：
#      k=1    -> 1/(1+1)=0.5, 1/(1+2)=0.333 ... 跨度巨大，第1名是第5名的 1.67 倍
#      k=60   -> 1/61≈0.0164 vs 1/65≈0.0154，差异被压缩到千分位，稳健
#      k=1000 -> 1/1001≈0.000999 vs 1/1005≈0.000995，几乎全相等
#      → 结论：k 越大越"平均化"。k 极大时 RRF 退化成"谁被更多路命中谁靠前"，
#        排名位置失去作用。默认 60 是稳健选择。
#
#   2. WeightedRanker 两组对比：
#      仔细看两者的【分数绝对值】—— 偏关键词那组普遍更高
#      因为 BM25 的归一化分通常在 0.6-0.8，而 COSINE 归一化后在 0.5-0.6
#      → 结论：权重 0.3/0.7 的"实际影响力"不一定是 3:7，因为两路归一化后
#        的分布本身就不重合。这就是 Weighted 难调的根因。
#
#   3. ★ 过滤生效验证：
#      带 expr 之后，返回结果的 dimension 应该【全部】是 opinion。
#      如果看到 event/entity 混进来，说明 expr 没生效 —— 检查是不是写到了
#      hybrid_search 的顶层参数（会被静默丢弃）。
#      脚本末尾的"自动校验"就是帮你盯这一点的。
#
#   4. 各路候选数 limit：
#      候选数=1 时，每路只有 1 条，两路交集几乎为空，RRF 退化成"交替排列"
#      候选数=2 类似。候选数=10 时融合才充分。
#      → 结论：cand_limit 要 ≥ 最终 top-k 的 2-3 倍，否则融合没意义。
#        生产环境常用 50-100（因为要留给 Reranker 重排）。
#
#   5. filter 在向量检索内部生效（不是先查后筛）：
#      如果 expr 筛掉 99% 数据，HNSW 图遍历可能找不到足够候选 → 召回骤降。
#      这时要调大 ef 或候选数补偿。Step 4 有专门演练。
# ---------------------------------------------------------------
