"""
Step 2：灌数据 + 看三路检索各自返回什么

学习目标：
  1. 写入时 BM25 稀疏向量是自动生成的（你完全不用管）
  2. 三种单路检索的形态：dense / sparse(BM25) / 纯标量过滤
  3. 建立直觉：哪类 query 该由哪一路负责

本次改动：
  - 用真实 embedding（BAAI/bge-m3，硅基流动）替代伪向量
  - 插入前【幂等清空】：重复跑不会累积数据
  - 插入后【去重自检】：主动发现脏数据

运行：python step2_insert_and_probe.py
"""

import time

from common import (
    get_client, embed_cached_batch, print_hits,
    COLLECTION, SAMPLE_DOCS,
)

client = get_client()

# ---------------------------------------------------------------
# 0. 幂等清空 —— 无论跑多少次，数据都是 10 条
#    教训：之前靠"手动记着先跑 step1"防重复，踩了两次坑。
#    脚本应该自己防住，不该依赖人的记忆。
# ---------------------------------------------------------------
if client.has_collection(COLLECTION):
    client.delete(collection_name=COLLECTION, filter="pk > 0")
    client.flush(COLLECTION)
    print("[i] 已清空旧数据（幂等保证：重复跑不会累积）")
else:
    raise RuntimeError(
        f"collection '{COLLECTION}' 不存在，请先运行 step1_schema.py"
    )

# ---------------------------------------------------------------
# 1. 造数据 —— 用真实 embedding 批量生成
# ---------------------------------------------------------------
contents = [c for _, _, _, _, c in SAMPLE_DOCS]
print(f"[i] 正在为 {len(contents)} 条切片生成向量（BAAI/bge-m3, 1024 维）...")
vectors = embed_cached_batch(contents)

rows = []
now = int(time.time())
for (doc_id, dim_name, offset, title, content), vec in zip(SAMPLE_DOCS, vectors):
    rows.append({
        "dense_vector": vec,
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

# flush 的作用（准确表述，别记错）：
#   ✗ 不是"让数据可见" —— growing segment 在内存里就能被检索
#   ✗ 不是"BM25 统计才准" —— TF/IDF 是实时统计的
#   ✓ 是把内存段【持久化并封存】成不可变 segment，让结果【确定性可复现】
#   ✓ 顺带让 get_collection_stats 的 row_count 准确、ANN 索引能建在 sealed segment 上
# 生产环境【不要手动 flush】—— 由 dataCoord 按 segment 大小自动触发，
# 手动 flush 会产生大量小 segment，拖慢查询。
client.flush(COLLECTION)
print("[OK] flushed（学习脚本用，保证结果可复现；生产环境交给 dataCoord 自动做）")
time.sleep(1)

stats = client.get_collection_stats(COLLECTION)
print(f"[i] collection 当前行数：{stats['row_count']}")

# ---------------------------------------------------------------
# 1.1 去重自检 —— 主动发现"重复插入"这类脏数据
# ---------------------------------------------------------------
allrows = client.query(
    collection_name=COLLECTION,
    filter="pk > 0",
    output_fields=["doc_id", "chunk_offset"],
    limit=1000,
)
keys = [(r["doc_id"], r["chunk_offset"]) for r in allrows]
dups = len(keys) - len(set(keys))
if dups > 0:
    print(f"[!] 数据自检失败：{len(keys)} 条中有 {dups} 条重复！"
          f"后续检索结果不可信，请检查插入逻辑")
else:
    print(f"[OK] 数据自检通过：{len(keys)} 条，无重复")



# ---------------------------------------------------------------
# 2. 三路检索对比 —— 本步核心
# ---------------------------------------------------------------

# --- 2.1 Dense 语义检索 ---
# 用真实 embedding 编码 query（bge-m3，1024 维）
q_dense = embed_cached_batch(["公司新能源业务增长情况如何"])[0]
hits = client.search(
    collection_name=COLLECTION,
    data=[q_dense],
    anns_field="dense_vector",
    search_params={"metric_type": "COSINE", "params": {"ef": 64}},  # 3.x 是 search_params
    limit=5,
    output_fields=["doc_id", "dimension", "chunk_offset", "content"],
)[0]
print_hits("【Dense 语义检索】query = '公司新能源业务增长情况如何'",
           hits, score_label="COSINE 相似度")


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
print_hits("【BM25 关键词检索】query = '隆基绿能'（专有名词）",
           hits, score_label="BM25 相关度")

hits = client.search(
    collection_name=COLLECTION,
    data=["价格战"],
    anns_field="sparse_vector",
    search_params={"metric_type": "BM25"},
    limit=5,
    output_fields=["doc_id", "dimension", "chunk_offset", "content"],
)[0]
print_hits("【BM25 关键词检索】query = '价格战'（领域术语）",
           hits, score_label="BM25 相关度")


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
# 【坑】output_fields 必须写全！Milvus 只返回你点名的字段。
# 上一版这里漏了 dimension，所以打印出来是 None —— 不是数据问题，是没请求该字段。
hits = client.query(
    collection_name=COLLECTION,
    filter='doc_id == "doc_a" and chunk_offset >= 300 and chunk_offset < 900',
    output_fields=["doc_id", "dimension", "chunk_offset", "content"],  # ← dimension 已补回
    limit=10,
)
print_hits("【范围过滤】doc_id=='doc_a' and 300<=chunk_offset<900", hits)


# ---------------------------------------------------------------
# 【该观察什么 / 该得出什么结论】
#
#   a) Dense 检索 "公司新能源业务增长情况如何"：
#      用真实 bge-m3 之后，结果应该符合语义直觉：
#      - 「营收增长45%」「毛利率提升至28%」排最前（都是"增长"语义）
#      - 「竞品下滑12%」排后（语义方向相反）
#      → 结论：这才是真实 embedding 的行为。伪向量时代那些"排序乱"的现象
#        是向量质量问题，不是 Milvus 的问题。
#
#   b) BM25 检索 "隆基绿能"：
#      - 精确命中 doc_a entity 那条，其他 9 条完全不出现
#      - 【关键】不是"其他条分数低"，而是 BM25 分数为 0 的【直接被丢弃】。
#        dense 会硬凑满 top-k（哪怕分数是负数），BM25 只给真命中的。
#      → 结论：这个差异正是 hybrid 有效的原因 —— BM25 的精确性
#        补上了 dense 的"硬凑"。
#
#   c) BM25 检索 "价格战"：
#      - 命中 doc_a opinion
#      → 结论：领域术语走 BM25 明显更准
#
#   d) 标量过滤：doc_id in ['doc_a','doc_c'] 且 dimension=='opinion'
#      → 返回 2 条（doc_a opinion@900 + doc_c opinion@250）
#      【注意】不是 3 条 —— doc_b 被 doc_id 条件排除了。
#      范围过滤：doc_a 里 300<=offset<900 → 返回 offset=300 和 600 两条
#
#   核心结论：三路各自都有盲区 → 本项目必须走 hybrid_search + RRF 融合
# ---------------------------------------------------------------
