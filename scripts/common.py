"""
公共配置与工具 —— 所有 step 脚本共用

重要环境说明（实测结论，2026-10-13）：
  - Windows + Python 3.13 上 milvus_lite **没有可用 wheel**，Lite 路线走不通
    → 只能走 Docker 部署的 standalone 版
  - pymilvus 当前版本 3.0.2，search() 的参数名是 `search_params`（不是旧版的 `param`）
  - BM25 Function / 中文 analyzer 需要 Milvus server >= 2.5，Lite 也不支持

【重大 API 差异 —— 必看】
  hybrid_search() **没有顶层 filter 参数**（签名只有 collection_name/reqs/
  ranker/limit/output_fields/timeout/partition_names）。
  传 filter=... 会进 **kwargs 被静默丢弃 → 过滤完全不生效！
  正确做法：把 expr（或 filter）写进每个 AnnSearchRequest。

【Embedding 接入】
  模型：BAAI/bge-m3（硅基流动，免费，1024 维，中英文能力强）
  接口：OpenAI 兼容  https://api.siliconflow.cn/v1/embeddings
  key ：从环境变量 SILICONFLOW_API_KEY 读取，不硬编码

用法：所有脚本从本模块 import，改 MILVUS_URI 即可切换环境
"""

import os
import json
import hashlib
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------
# 加载 .env（放在 learning/milvus/ 下）
# 优先级：真实环境变量 > .env 文件
# 这样你既能用 .env 存 key，也能用临时环境变量覆盖
# ---------------------------------------------------------------
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        if _k and _k not in os.environ:
            os.environ[_k] = _v

# ---------------------------------------------------------------
# 连接配置
# ---------------------------------------------------------------
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = "llm_wiki_chunks"

# ---------------------------------------------------------------
# Embedding 配置
# ---------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-m3"
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBED_DIM = 1024            # bge-m3 输出维度

# DIM 是 collection schema 用的维度，必须和 embedding 模型一致
DIM = EMBED_DIM

# 本地缓存目录：避免重复调用 API（省钱 + 提速）
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "embed_cache"

OUTPUT_FIELDS = ["doc_id", "dimension", "chunk_offset", "content", "title"]


def get_client():
    from pymilvus import MilvusClient
    return MilvusClient(uri=MILVUS_URI)


def _get_api_key() -> str:
    key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "未设置 SILICONFLOW_API_KEY 环境变量。\n"
            "  Windows PowerShell:  $env:SILICONFLOW_API_KEY='sk-xxx'\n"
            "  Windows CMD:         set SILICONFLOW_API_KEY=sk-xxx\n"
            "  或永久写入项目 .env 后由脚本加载"
        )
    return key


# ---------------------------------------------------------------
# Embedding：真实模型 + 本地缓存
# ---------------------------------------------------------------
def _cache_path(text: str) -> Path:
    h = hashlib.sha256(f"{EMBED_MODEL}::{text}".encode("utf-8")).hexdigest()[:32]
    return CACHE_DIR / f"{h}.json"


def embed(text: str) -> list:
    """
    单条文本 -> 1024 维向量（带本地缓存）

    为什么加缓存：
      - 建 collection / 反复调试时同一条文本会被算很多次
      - 免费额度虽然够用，但没必要浪费，也省网络延迟
    """
    cp = _cache_path(text)
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass  # 缓存损坏则重算

    vec = embed_batch([text])[0]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(vec), encoding="utf-8")
    return vec


def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    批量文本 -> 向量列表（不查缓存，直接调 API）

    硅基流动单次请求最多 1024 条，超出会自动分批。
    """
    if not texts:
        return []

    api_key = _get_api_key()
    out: list[list[float]] = []
    BATCH = 64      # 保守批量，避免单请求过大

    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        payload = json.dumps({
            "model": EMBED_MODEL,
            "input": chunk,
            "encoding_format": "float",
        }).encode("utf-8")

        req = urllib.request.Request(
            EMBED_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"embedding API HTTPError {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"embedding API 网络失败: {e}") from e

        # OpenAI 兼容格式：{"data": [{"embedding": [...], "index": 0}, ...]}
        data = sorted(body["data"], key=lambda d: d.get("index", 0))
        out.extend(d["embedding"] for d in data)

    got = len(out)
    if got != len(texts):
        raise RuntimeError(f"embedding 返回条数不符：请求 {len(texts)}，返回 {got}")
    return out


def embed_cached_batch(texts: list[str]) -> list[list[float]]:
    """批量 + 缓存：只对未命中的文本调 API"""
    results: list[list[float] | None] = [None] * len(texts)
    miss_idx, miss_texts = [], []

    for i, t in enumerate(texts):
        cp = _cache_path(t)
        if cp.exists():
            try:
                results[i] = json.loads(cp.read_text(encoding="utf-8"))
                continue
            except Exception:
                pass
        miss_idx.append(i)
        miss_texts.append(t)

    if miss_texts:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        vecs = embed_batch(miss_texts)
        for i, t, v in zip(miss_idx, miss_texts, vecs):
            results[i] = v
            _cache_path(t).write_text(json.dumps(v), encoding="utf-8")
        print(f"[i] embedding: 新增计算 {len(miss_texts)} 条"
              f"（命中缓存 {len(texts) - len(miss_texts)} 条）")
    else:
        print(f"[i] embedding: 全部命中缓存（{len(texts)} 条），未调 API")

    return results  # type: ignore[return-value]


# ---------------------------------------------------------------
# 模拟切片数据 —— 四维度 × 三文档，覆盖本项目真实形态
# ---------------------------------------------------------------
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


def print_hits(title: str, hits, max_len: int = 38, score_label: str | None = None):
    """
    统一的结果打印格式

    score_label 说明：
      None            -> 自动判断。有 distance 就显示，没有就显示 "(无分数)"
      "RRF"           -> 标注该分数是 RRF 融合分，无物理意义
      "BM25" / "COSINE" -> 标注分数的度量类型
    """
    print(f"\n{'='*76}\n{title}\n{'='*76}")
    if not hits:
        print("  (无结果)")
        return

    has_score = any("distance" in h for h in hits)
    if not has_score:
        header = "(纯标量查询，无相似度分数)"
    elif score_label:
        header = f"(分数类型: {score_label})"
    else:
        header = ""
    if header:
        print(f"  {header}")

    for i, h in enumerate(hits, 1):
        e = h.get("entity", h)      # search 返回有 entity 包一层，query 返回平铺
        content = str(e.get("content", ""))[:max_len]

        if has_score:
            score_str = f"score={h.get('distance', 0):>9.4f}"
        else:
            score_str = "score=     ---"

        print(f"  {i}. {score_str} | {e.get('doc_id')} "
              f"| {str(e.get('dimension')):<8} | off={e.get('chunk_offset')} "
              f"| {content}...")


def show_collection_info(client, name=COLLECTION):
    """建完 collection 后回看实际生效的 schema / functions / indexes"""
    desc = client.describe_collection(name)
    print("\n=== Fields ===")
    for f in desc["fields"]:
        params = f.get("params", {})
        interesting = {k: v for k, v in params.items()
                       if k in ("enable_analyzer", "analyzer_params", "enable_match", "max_length")}
        print(f"  {f['name']:<16} {f['type']:<22} {interesting if interesting else ''}")

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
