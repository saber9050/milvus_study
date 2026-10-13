"""
Step 3：Hybrid Search + RRF 融合 —— 本项目的检索主形态

学习目标：
  1. AnnSearchRequest 怎么构造多路请求
  2. RRFRanker vs WeightedRanker 的实际差异（重点，要亲眼看到）
  3. RRF 的 k 参数怎么影响结果
  4. 带标量过滤的 hybrid search 怎么写（消费轮真实形态）

运行：python step3_hybrid_search.py
"""

from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

from common import get_client, fake_embed, print_hits, COLLECTION

client = get_client()

QUERY = "新能源业务的盈利能力和竞争风险"
q_vec = fake_embed(QUERY)


def dense_req(limit=10, ef=64):
    return AnnSearchRequest(
        data=[q_vec],
        anns_field="dense_vector",
        param={"metric_type": "COSINE", "params": {"ef": ef}},
        limit=limit,
    )


def bm25_req(query=QUERY, limit=10):
    # 原文直接传，Milvus 用 content 的 analyzer 分词
    return AnnSearchRequest(
        data=[query],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        limit=limit,
    )


OUT = ["doc_id", "dimension", "chunk_offset", "content"]


def hsearch(ranker, limit=5, filter_expr=""):
    kwargs = dict(
        collection_name=COLLECTION,
        reqs=[dense_req(), bm25_req()],
        ranker=ranker,
        limit=limit,
        output_fields=OUT,
    )
    if filter_expr:
        kwargs["filter"] = filter_expr
    return client.hybrid_search(**kwargs)[0]


# ---------------------------------------------------------------
# 1. RRF 融合
#    公式：score = Σ 1/(k + rank_i)
#    特点：只看排名不看原始分数 → 对 dense/sparse 分数尺度差异免疫
# ---------------------------------------------------------------
print_hits("RRFRanker(k=60) —— Milvus 默认值",
           hsearch(RRFRanker(60)))

print_hits("RRFRanker(k=1) —— 极端值，注意排序变化",
           hsearch(RRFRanker(1)))

print_hits("RRFRanker(k=1000) —— 极端大值，排名差异被抹平",
           hsearch(RRFRanker(1000)))

# ---------------------------------------------------------------
# 2. WeightedRanker —— 需要归一化 + 手调权重
# ---------------------------------------------------------------
print_hits("WeightedRanker(0.7, 0.3) —— 偏语义",
           hsearch(WeightedRanker(0.7, 0.3)))

print_hits("WeightedRanker(0.3, 0.7) —— 偏关键词",
           hsearch(WeightedRanker(0.3, 0.7)))

# ---------------------------------------------------------------
# 3. 带标量过滤的 hybrid —— 消费轮真实形态
#    比如「只在竞品分析报告的观点维度里找盈利风险」
# ---------------------------------------------------------------
print_hits('Hybrid + filter: dimension == "opinion"',
           hsearch(RRFRanker(60), filter_expr='dimension == "opinion"'))

print_hits('Hybrid + filter: doc_id == "doc_c"',
           hsearch(RRFRanker(60), filter_expr='doc_id == "doc_c"'))


# ---------------------------------------------------------------
# 【该观察什么 / 该得出什么结论】
#
#   1. k 的影响：
#      k=1   -> 单路排名第1的贡献是 1/1=1.0，第2名只有 0.5，极度"独裁"
#      k=60  -> 1/61≈0.0164 vs 1/62≈0.0161，差异被压缩，稳健
#      k=1000-> 1/1001 vs 1/1002，几乎等价于"各路平均"
#      → 结论：k=60 是平滑稳健的默认。只有当你确信某一路明显更可信时才调小。
#
#   2. WeightedRanker(0.7,0.3) vs (0.3,0.7)：
#      排序应出现可见差异 —— 这就是"需要调权重"的代价
#      → 结论：本项目**默认选 RRF**。理由：
#        dense 用 COSINE（0~1），BM25 分数范围是另一套（可到几十），
#        两者不可比。WeightedRanker 依赖内部归一化，调不好比 RRF 更差。
#        只有在 Step 4 评估数据支撑下才切 Weighted。
#
#   3. filter 的执行位置：
#      filter 在向量检索**内部**生效，不是先查后筛
#      → 结论：如果 filter 筛掉 99% 数据，HNSW 图遍历可能找不到足够候选，
#        召回率会骤降。这时要调大 ef 或 search_list 补偿。
#        Step 4 有个演练专门让你看到这个现象。
# ---------------------------------------------------------------
