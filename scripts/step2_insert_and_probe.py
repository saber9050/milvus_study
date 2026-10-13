"""
Step 2：灌数据 + 看三路检索各自返回什么

学习目标：
  1. 写入时 BM25 稀疏向量是自动生成的（你完全不用管）
  2. 三种单路检索的形态：dense / sparse(BM25) / 纯标量过滤
  3. 建立直觉：哪类 query 该由哪一路负责

运行：python step2_insert_and_probe.py
"""

import time

from common import (
    get_client, fake_embed, print_hits,
    COLLECTION, SAMPLE_DOCS,
)

client = get_client()

# ---------------------------------------------------------------
# 1. 造数据 —— 故意设计成：语义相似但关键词不同 / 关键词相同但语义不同
#    这样你能看出 dense 和 sparse 的分工差异
# ---------------------------------------------------------------
rows = []
now = int(time.time())
for doc_id, dim_name, offset, title, content in SAMPLE_DOCS:
    rows.append({
        "dense_vector": fake_embed(content, doc_id),
        "doc_id": doc_id,
        "chunk_offset": offset,
        "dimension": dim_name,
        "skill_id": "skill_finance_extract_v1",
        "content": content,
        "title": title,
        "source_page": (offset // 100) + 1,
        "created_at": now,
    })

res = client.insert(collection_name=COLLECTION, data=rows)
print(f"[OK] 插入 {res['insert_count']} 条")

# flush：让数据真正落到 segment
# 注意 BM25 的 TF/IDF 是基于已落盘数据的实时统计，不 flush 统计不完整
client.flush(COLLECTION)
print("[OK] flushed")
time.sleep(1)

stats = client.get_collection_stats(COLLECTION)
print(f"[i] collection 当前行数：{stats['row_count']}")


# ---------------------------------------------------------------
# 2. 三路检索对比 —— 本步核心
# ---------------------------------------------------------------

# --- 2.1 Dense 语义检索 ---
q_dense = fake_embed("公司新能源业务增长情况如何", "doc_a")
hits = client.search(
    collection_name=COLLECTION,
    data=[q_dense],
    anns_field="dense_vector",
    search_params={"metric_type": "COSINE", "params": {"ef": 64}},  # 3.x 是 search_params
    limit=5,
    output_fields=["doc_id", "dimension", "chunk_offset", "content"],
)[0]
print_hits("【Dense 语义检索】query = '公司新能源业务增长情况如何'", hits)


# --- 2.2 Sparse / BM25 关键词检索 ---
# 关键：直接传原始文本，Milvus 用 content 字段的同一个 analyzer 处理
hits = client.search(
    collection_name=COLLECTION,
    data=["隆基绿能"],
    anns_field="sparse_vector",
    search_params={"metric_type": "BM25"},
    limit=5,
    output_fields=["doc_id", "dimension", "chunk_offset", "content"],
)[0]
print_hits("【BM25 关键词检索】query = '隆基绿能'（专有名词）", hits)

hits = client.search(
    collection_name=COLLECTION,
    data=["价格战"],
    anns_field="sparse_vector",
    search_params={"metric_type": "BM25"},
    limit=5,
    output_fields=["doc_id", "dimension", "chunk_offset", "content"],
)[0]
print_hits("【BM25 关键词检索】query = '价格战'（领域术语）", hits)


# --- 2.3 纯标量过滤（不走向量，纯结构化筛选）---
# 这一步同时验证 INVERTED 索引是否真的生效
hits = client.query(
    collection_name=COLLECTION,
    filter='dimension == "opinion" and doc_id in ["doc_a", "doc_c"]',
    output_fields=["doc_id", "dimension", "chunk_offset", "content"],
    limit=10,
)
print_hits("【纯标量过滤】dimension=='opinion' and doc_id in ['doc_a','doc_c']", hits)


# --- 2.4 补一个：范围查询验证 STL_SORT 索引 ---
hits = client.query(
    collection_name=COLLECTION,
    filter='doc_id == "doc_a" and chunk_offset >= 300 and chunk_offset < 900',
    output_fields=["doc_id", "chunk_offset", "content"],
    limit=10,
)
print_hits("【范围过滤】doc_id=='doc_a' and 300<=chunk_offset<900", hits)


# ---------------------------------------------------------------
# 【该观察什么 / 该得出什么结论】
#
#   a) Dense 检索 "公司新能源业务增长情况如何"：
#      - 会召回 doc_a 的 event（营收增长45%）—— 语义对上
#      - 也可能召回 doc_c 的 event（竞品下滑12%）—— 语义相关但**方向相反**
#      → 结论：dense 分不清"涨"和"跌"。纯向量检索的固有缺陷，本项目无解，
#        只能靠 hybrid + 结构化字段（比如给 event 加 trend 标记）来补。
#
#   b) BM25 检索 "隆基绿能"：
#      - 应该精确命中 doc_a entity 那条，其他 9 条完全不出现
#      → 结论：专有名词的精确性是 dense 给不了的
#
#   c) BM25 检索 "价格战"：
#      - 命中 doc_a opinion
#      → 结论：领域术语走 BM25 明显更准
#
#   d) 标量过滤：应返回 3 条（doc_a/doc_b/doc_c 各一条 opinion）
#      范围过滤：应返回 doc_a 的 offset=300 和 offset=600 两条
#      → 这是消费轮"只看某文档的观点"这类需求的底层能力
#
#   核心结论：三路各自都有盲区 → 本项目必须走 hybrid_search + RRF 融合
# ---------------------------------------------------------------
