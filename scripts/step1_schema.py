"""
Step 1：建 Collection —— 把 PRD 的四维度 + 三元溯源落成真实 schema

学习目标（不是"会建表"，是理解每个字段为什么这么设计）：
  1. 四维度为什么建议放同一个 collection 用 dimension 字段区分
  2. 三元溯源字段为什么必须显式定义（不定义就没法建标量索引）
  3. 标量索引为什么不是可选项
  4. BM25 Function 怎么挂，中文 analyzer 怎么配

前置：Milvus 已启动（docker compose up -d，等健康检查通过）
运行：python step1_schema.py
"""

from pymilvus import DataType, Function, FunctionType

from common import get_client, COLLECTION, DIM, show_collection_info

client = get_client()

# ---------------------------------------------------------------
# 1. 建 schema
#    字段分三类：
#      A. 主键 + 向量（Milvus 必需）
#      B. 三元溯源（本项目跨系统契约）
#      C. 内容与元信息（BM25 源字段 + 展示用）
# ---------------------------------------------------------------
schema = client.create_schema(
    auto_id=True,
    enable_dynamic_field=True,   # 预留：元信息扩展字段可动态写入
    description="LLM Wiki 生产轮抽取结果存储",
)

# --- A. 主键 ---
schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)

# --- A. Dense 向量（语义检索用）---
schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DIM)

# --- B. 三元溯源 ---
# 为什么必须显式定义：这几个是消费轮过滤主力，必须建标量索引
schema.add_field(
    "doc_id", DataType.VARCHAR, max_length=128,
    description="源文档 ID —— 溯源第一元",
)
schema.add_field(
    "chunk_offset", DataType.INT64,
    description="切片在原文中的偏移 —— 溯源第二元，用于定位回原文段落",
)
schema.add_field(
    "dimension", DataType.VARCHAR, max_length=32,
    description="四维度之一：meta / entity / event / opinion",
)
schema.add_field(
    "skill_id", DataType.VARCHAR, max_length=128,
    description="产出该记录的 Skill —— FR-1.5 skill_id 溯源",
)

# --- C. 内容字段：BM25 的源 ---
# 三个关键参数缺一不可：
#   enable_analyzer=True     -> 开启分词，BM25 依赖
#   analyzer_params=chinese  -> 中文文档必须走这条
#   enable_match=True        -> 开启标量文本精确匹配
schema.add_field(
    "content", DataType.VARCHAR, max_length=65535,
    enable_analyzer=True,
    analyzer_params={"type": "chinese"},
    enable_match=True,
    description="抽取出的文本内容 —— BM25 全文检索的源字段",
)

# --- C. 稀疏向量字段：BM25 的【输出容器】---
# 【关键坑】这个字段必须在 schema 里【先显式定义】，
# 否则 create_collection 时 schema.verify() 会抛：
#   ParamError: Function output field not found in collection schema.: sparse_vector
# 原因：pymilvus 会校验 Function 的 output_field_names 是否存在于 schema 中。
# 它不需要你手动赋值（Milvus 写入时自动算 BM25 填进去），但字段本身要先占位。
schema.add_field(
    "sparse_vector", DataType.SPARSE_FLOAT_VECTOR,
    description="BM25 Function 自动写入的稀疏向量 —— 不用手动赋值",
)

# --- C. 结构化辅助字段（可空，因为不同维度抽出的字段不同）---
schema.add_field("title", DataType.VARCHAR, max_length=512, nullable=True)
schema.add_field("source_page", DataType.INT64, nullable=True)
schema.add_field("created_at", DataType.INT64, description="unix 秒")

# ---------------------------------------------------------------
# 2. 挂 BM25 Function
#    Milvus 自动把 content 转成 sparse 向量，写入时自动生成
#    注意：output_field_names 里的字段**必须在 schema 里已定义**
#          （上面已加 sparse_vector 字段），否则 verify 直接报：
#          ParamError: Function output field not found in collection schema.
# ---------------------------------------------------------------
bm25_function = Function(
    name="content_bm25",
    function_type=FunctionType.BM25,
    input_field_names=["content"],
    output_field_names=["sparse_vector"],
)
schema.add_function(bm25_function)

# ---------------------------------------------------------------
# 3. 建索引 —— 每一类字段都要有索引，缺一个数据量上来就炸
# ---------------------------------------------------------------
index_params = client.prepare_index_params()

# 3.1 Dense：学习阶段用 HNSW（召回好、参数直观）
#     注意 metric_type 和 params 都走 kwargs
index_params.add_index(
    field_name="dense_vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={
        "M": 16,                 # 图连接度：16 是常用起点
        "efConstruction": 200,   # 建索引质量
    },
)

# 3.2 Sparse（BM25 自动生成的那个字段）
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
)

# 3.3 标量索引 —— 本项目过滤全靠它，不建就是全表扫描
for f in ["doc_id", "dimension", "skill_id"]:
    index_params.add_index(field_name=f, index_type="INVERTED")

# chunk_offset 需要范围查询（取原文上下文区间），用 STL_SORT
index_params.add_index(field_name="chunk_offset", index_type="STL_SORT")

# ---------------------------------------------------------------
# 4. 创建 collection
# ---------------------------------------------------------------
if client.has_collection(COLLECTION):
    print(f"[!] {COLLECTION} 已存在，先删除（学习环境可随意重建）")
    client.drop_collection(COLLECTION)

client.create_collection(
    collection_name=COLLECTION,
    schema=schema,
    index_params=index_params,
)
print(f"[OK] collection '{COLLECTION}' 创建完成")

# ---------------------------------------------------------------
# 5. 回看实际生效的配置 —— 这步很重要，别跳过
# ---------------------------------------------------------------
show_collection_info(client)

# ---------------------------------------------------------------
# 【该观察什么 / 该得出什么结论】
#
#   1. Analyzer 检查区块应该出现：
#      [!] content 已开启分词: {'type': 'chinese'} | enable_match=True
#      → 如果没有，说明 enable_analyzer 没生效，BM25 检索会退化成按字符切
#
#   2. Functions 区块应该有 content_bm25
#      → 如果显示"(无)"，BM25 根本没挂上，后面 hybrid_search 会直接报错
#
#   3. Indexes 区块应该有 6 个：
#      dense_vector / sparse_vector / doc_id / dimension / skill_id / chunk_offset
#      → 缺 sparse_vector 是最常见的坑
#
#   4. 记住这个判断方法：**建完必须回看**。
#      很多"检索效果不好"的问题，根因是 schema 建错了但没人发现。
# ---------------------------------------------------------------
