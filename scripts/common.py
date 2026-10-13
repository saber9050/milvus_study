"""
公共配置与工具 —— 所有 step 脚本共用

重要环境说明（实测结论，2026-10-13）：
  - Windows + Python 3.13 上 milvus_lite **没有可用 wheel**，Lite 路线走不通
    → 只能走 Docker 部署的 standalone 版
  - pymilvus 当前版本 3.0.2，search() 的参数名是 `search_params`（不是旧版的 `param`）
  - BM25 Function / 中文 analyzer 需要 Milvus server >= 2.5，Lite 也不支持

用法：所有脚本从本模块 import，改 MILVUS_URI 即可切换环境
"""

import os
import numpy as np

# ---------------------------------------------------------------
# 连接配置：默认连 Docker 起的 standalone
# 如果你以后部署到服务器，只改这一行
# ---------------------------------------------------------------
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = "llm_wiki_chunks"
DIM = 1536          # DashScope text-embedding-v2 维度

# 返回结构的字段列表，统一提出来避免各脚本写不一致
OUTPUT_FIELDS = ["doc_id", "dimension", "chunk_offset", "content", "title"]


def get_client():
    from pymilvus import MilvusClient
    return MilvusClient(uri=MILVUS_URI)


def fake_embed(text: str, doc_id: str = "query") -> list:
    """
    【仅学习用】确定性伪向量生成器

    真实项目替换为：
        import dashscope
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v2", input=text)
        return resp.output["embeddings"][0]["embedding"]

    为什么用伪向量：让你在**没有 API key、没有网络**的情况下也能跑通全部流程，
    把注意力放在 Milvus 的行为上，而不是被 embedding 服务卡住。

    构造逻辑：同 doc_id 的向量共享一个基向量（模拟"同文档语义相关"），
    再叠加与文本内容相关的扰动。这样 Dense 检索会表现出"同文档优先"的倾向，
    足以观察 hybrid 融合行为。
    """
    rng = np.random.default_rng(abs(hash(doc_id)) % (2**32))
    base = rng.normal(0, 1, DIM)
    rng2 = np.random.default_rng(abs(hash(text)) % (2**32))
    v = base + rng2.normal(0, 0.3, DIM)
    return (v / np.linalg.norm(v)).tolist()


# 模拟切片数据 —— 四维度 × 三文档，覆盖本项目真实形态
SAMPLE_DOCS = [
    ("doc_a", "event", 0, "某公司2026年Q1财报",
     "该公司2026年第一季度营业收入达到12.3亿元，同比增长45%，主要由新能源业务驱动。"),
    ("doc_a", "event", 300, "某公司2026年Q1财报",
     "管理层在业绩会上表示，新能源板块毛利率提升至28%，预计全年维持该水平。"),
    ("doc_a", "entity", 600, "某公司2026年Q1财报",
     "宁德时代、比亚迪、隆基绿能是该公司新能源业务的主要供应商。"),
    ("doc_a", "opinion", 900, "某公司2026年Q1财报",
     "分析师认为该公司的估值仍处于合理区间，但需警惕行业竞争加剧带来的价格战风险。"),
    ("doc_b", "event", 0, "行业政策解读",
     "国家发改委发布关于促进新能源产业健康发展的指导意见，提出到2030年装机容量翻两番。"),
    ("doc_b", "entity", 200, "行业政策解读",
     "发改委、能源局、工信部三部门联合推动该政策落地。"),
    ("doc_b", "opinion", 400, "行业政策解读",
     "业内普遍认为政策利好长期需求，但短期产能过剩问题依然突出。"),
    ("doc_c", "event", 0, "竞品分析报告",
     "竞品公司在2026年Q1实现营收8.7亿元，同比下滑12%，主因是新能源业务转型不及预期。"),
    ("doc_c", "opinion", 250, "竞品分析报告",
     "该竞品的组织架构调整过于频繁，导致战略执行连续性不足。"),
    ("doc_c", "meta", 500, "竞品分析报告",
     "本报告数据来源：公开财报、券商研报、行业访谈，统计截止2026年3月31日。"),
]


def print_hits(title: str, hits, max_len: int = 38):
    """统一的结果打印格式"""
    print(f"\n{'='*76}\n{title}\n{'='*76}")
    if not hits:
        print("  (无结果)")
        return
    for i, h in enumerate(hits, 1):
        e = h.get("entity", h)      # search 返回有 entity 包一层，query 返回没有
        score = h.get("distance", h.get("score", 0))
        content = str(e.get("content", ""))[:max_len]
        print(f"  {i}. score={score:.4f} | {e.get('doc_id')} "
              f"| {str(e.get('dimension')):<8} | off={e.get('chunk_offset')} "
              f"| {content}...")


def show_collection_info(client, name=COLLECTION):
    """建完 collection 后回看实际生效的 schema / functions / indexes"""
    desc = client.describe_collection(name)
    print("\n=== Fields ===")
    for f in desc["fields"]:
        extra = {k: v for k, v in f.items()
                 if k not in ("name", "type", "description", "params")}
        params = f.get("params", {})
        interesting = {k: v for k, v in params.items()
                       if k in ("enable_analyzer", "analyzer_params", "enable_match", "max_length")}
        print(f"  {f['name']:<16} {f['type']:<22} {interesting if interesting else ''}")

    # pymilvus 3.x 里 analyzer 配置在 params 下，单独打一遍确保能看到
    print("\n=== Analyzer / Match 检查 ===")
    for f in desc["fields"]:
        p = f.get("params", {})
        if p.get("enable_analyzer"):
            print(f"  [!] {f['name']} 已开启分词: {p.get('analyzer_params')} "
                  f"| enable_match={p.get('enable_match')}")

    print("\n=== Functions ===")
    fns = desc.get("functions", [])
    if not fns:
        print("  (无) ← BM25 Function 没挂上！检查 add_function 是否被调用")
    for fn in fns:
        print(f"  {fn}")

    print("\n=== Indexes ===")
    for idx in client.list_indexes(name):
        info = client.describe_index(name, idx)
        print(f"  {idx:<18} type={info.get('index_type'):<26} metric={info.get('metric_type')}")
