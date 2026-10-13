# Milvus 动手学习路径 —— LLM Wiki 项目专用

> 目标不是"会用 API"，而是**看到检索结果知道哪里不对、该怎么改**。
> 前 3 步练"会用"，第 4 步练"会调"。第 4 步才是本项目真正要的能力。

---

## 已知报错速查

### 1. `ParamError: Function output field not found in collection schema.: sparse_vector`

**原因**：BM25 Function 的 `output_field_names=["sparse_vector"]`，但 schema 里没有定义 `sparse_vector` 字段。

pymilvus 会在 `create_collection` 时调 `schema.verify()` 做校验，Function 的输出字段必须**先存在**。

**修复**：在 `add_function` 之前加：

```python
schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
```

注意：这个字段**不需要手动赋值**，Milvus 写入时自动算 BM25 填进去，但字段本身必须先占位。

### 2. ⚠️ `hybrid_search` 的 filter 被静默丢弃（最危险的坑）

**现象**：写了 `filter='dimension == "opinion"'`，结果里却混进了 `event`/`entity` 等不满足条件的记录，**而且不报任何错**。

**原因**：`hybrid_search()` 的签名里**没有 `filter` 参数**：

```
collection_name / reqs / ranker / limit / output_fields / timeout / partition_names
```

传 `filter=...` 会进 `**kwargs` 被忽略。

**修复**：过滤条件必须写进**每个** `AnnSearchRequest`：

```python
AnnSearchRequest(
    data=[q_vec], anns_field="dense_vector",
    param={...}, limit=10,
    expr='dimension == "opinion"',      # ← 这里
)
```

注意 `expr` 和 `filter` 二选一，同时传会抛 `ParamError`。

**单路 `search()` 的 `filter=` 是顶层参数，可以正常用。只有 hybrid 不一样。**

**这个坑的危险性在于它不报错。** 防坑手段：拿到结果后，检查返回项是否全部满足过滤条件（step3 脚本里有"自动校验"块专门做这个）。

### 3. `query()` 返回的字段出现 `None`

**原因**：`output_fields` 没写全。`search()` 和 `query()` 都只返回你点名的字段。

**修复**：把需要的字段都列进 `output_fields`。

---

## 环境现状（实测结论，2026-10-13）

| 项 | 状态 | 说明 |
|---|---|---|
| Docker | **未安装** | 必须先用 Docker 起 Milvus standalone |
| WSL | **被安全策略禁用**（agent 沙箱内） | 用户本机 WSL 可用，Milvus 已部署成功 |
| milvus_lite | **不可用** | Windows + Python 3.13 没有预编译 wheel，装不上 |
| pymilvus | 3.0.2 已装 | 在 `C:\Users\asus\.workbuddy\binaries\python\envs\default` |
| **Embedding** | **BAAI/bge-m3（硅基流动，免费，1024 维）** | 已接通，带本地缓存 |
| 结论 | 走 Docker/WSL 部署 | Lite 路线在本机走不通，不用浪费时间试 |

### Embedding 配置

| 项 | 值 |
|---|---|
| 模型 | `BAAI/bge-m3` |
| 服务 | 硅基流动 https://api.siliconflow.cn |
| 维度 | **1024**（schema 里的 `DIM` 必须一致） |
| API 格式 | OpenAI 兼容（`/v1/embeddings`） |
| Key 存放 | `learning/milvus/.env` 的 `SILICONFLOW_API_KEY`，由 `common.py` 自动加载 |
| 缓存 | `learning/milvus/data/embed_cache/`，按 sha256 去重，重复文本不重复调 API |

**为什么选 bge-m3 而不是 DashScope：** 免费、无需备案、中英文能力都强、OpenAI 兼容接口好接。

**换 key 的方法：** 改 `.env` 文件，或用临时环境变量覆盖（真实环境变量优先级更高）：
```bash
# PowerShell
$env:SILICONFLOW_API_KEY='sk-xxx'
# CMD
set SILICONFLOW_API_KEY=sk-xxx
```

### 关键 API 差异（踩坑点）

pymilvus 3.x 与网上老教程不同：

| 老写法（2.x） | 新写法（3.x） |
|---|---|
| `search(..., param={...})` | `search(..., search_params={...})` |
| `add_index` 的 `metric_type` 是位置参数 | 走 `**kwargs`，写法 `add_index(field_name="x", index_type="HNSW", metric_type="COSINE", params={...})` |

---

## 第 0 步：起 Milvus（先解决 Docker）

### 方案 A：装 Docker Desktop（推荐）

1. 官网下载 Docker Desktop for Windows：https://www.docker.com/products/docker-desktop/
2. 安装时**取消勾选** "Use WSL 2 based engine"（因为你的 WSL 被禁用）
   - 勾选 Hyper-V 后端即可
3. 装完确认：
   ```bash
   docker --version
   docker compose version
   docker ps
   ```

### 方案 B：装 Podman（Docker 装不上的替代）

Podman 命令与 Docker 基本兼容，不需要 WSL：

```bash
# Windows 用 winget 装
winget install RedHat.Podman-Desktop
```
之后把下文所有 `docker` 换成 `podman` 即可。

### 起 Milvus

```bash
cd "C:/Users/asus/Desktop/项目/qysc/llm_wiki_2/learning/milvus"
docker compose up -d

# 等待健康检查通过（首次拉镜像较慢，约 2-5 分钟）
docker compose ps          # 三个容器都应是 healthy / running

# 确认 gRPC 端口通了
curl http://localhost:9091/healthz
```

看到 `OK` 就说明 Milvus 起来了。

**如果拉镜像超时**（国内网络常见）：在 `docker-compose.yml` 顶部加镜像加速，或改用阿里云镜像源。这是你 blogProject 部署时已经踩过的坑，同理。

---

## 第 1 步：建 Collection —— 把 PRD 落成 schema

```bash
cd learning/milvus/scripts
python step1_schema.py
```

### 这一步在练什么

| 设计点 | 为什么 |
|---|---|
| 四维度放同一个 collection，用 `dimension` 标量字段区分 | 跨维度联合检索方便；分 collection 会隔离但检索要并发多路 |
| 三元溯源字段（`doc_id`/`chunk_offset`/`dimension`/`skill_id`）**显式定义** | 不显式定义就建不了标量索引，消费轮过滤会全表扫描 |
| `content` 开 `enable_analyzer=True` + `analyzer_params={"type":"chinese"}` | BM25 依赖分词，中文必须指定 chinese |
| `content` 开 `enable_match=True` | 支持标量文本精确匹配（短语过滤） |
| 标量字段全建 `INVERTED` 索引 | 等值过滤最高效；不建＝全表扫描 |
| `chunk_offset` 建 `STL_SORT` | 范围查询（取原文上下文区间）专用 |
| `BM25 Function` 挂到 `content` → `sparse_vector` | Milvus 自动生成稀疏向量，不用自己算 |
| **`sparse_vector` 字段必须先显式定义** | 坑！pymilvus 会校验 Function 的 output 字段是否存在于 schema，缺了直接报 `ParamError: Function output field not found in collection schema` |

### 该看什么

脚本末尾会打印三块：**Analyzer 检查 / Functions / Indexes**。

- Analyzer 区块应出现 `[!] content 已开启分词: {'type': 'chinese'}`
- Functions 应有 `content_bm25`
- Indexes 应有 **6 个**：`dense_vector`、`sparse_vector`、`doc_id`、`dimension`、`skill_id`、`chunk_offset`

**缺 `sparse_vector` 是最常见的坑** —— 说明 BM25 Function 没挂上。

### 该得什么结论

**建完必须回看。** 很多"检索效果差"的根因是 schema 建错但没人发现，然后大家在索引参数上瞎调一整天。

---

## 第 2 步：灌数据 + 三路检索对比

```bash
python step2_insert_and_probe.py
```

### 这一步在练什么

| 内容 | 要点 |
|---|---|
| 插入后必须 `flush` | BM25 的 TF/IDF 基于已落盘数据实时统计，不 flush 统计不完整 |
| Dense 语义检索 | 传向量，`anns_field="dense_vector"` |
| BM25 关键词检索 | **直接传原始文本**，Milvus 用同一个 analyzer 处理；`anns_field="sparse_vector"` |
| 纯标量过滤 | 用 `client.query()` 而非 `search()`，不走向量 |
| 范围过滤 | `chunk_offset >= 300 and chunk_offset < 900`，验证 STL_SORT 生效 |

### 该看什么

四组结果对比，重点看**同一句话在不同检索方式下的召回差异**。

### 该得什么结论

| 现象 | 结论 |
|---|---|
| Dense 检索"公司新能源增长"，同时召回 doc_a（增长45%）和 doc_c（下滑12%） | **dense 分不清"涨"和"跌"**，纯向量的固有缺陷 |
| BM25 检索"隆基绿能"，只精确命中 1 条 | **专有名词的精确性是 dense 给不了的** |
| BM25 检索"价格战"，精准命中 opinion 那条 | **领域术语走 BM25 明显更准** |
| 标量过滤能一次捞出 3 个文档的 opinion | 结构化维度是消费轮的底层能力 |

**核心结论：三路各有盲区 → 本项目必须走 hybrid_search + RRF。**

---

## 第 3 步：Hybrid Search + RRF 融合

```bash
python step3_hybrid_search.py
```

### 这一步在练什么

```python
req_dense = AnnSearchRequest(data=[query_vec], anns_field="dense_vector",
                             param={"metric_type": "COSINE", "params": {"ef": 64}}, limit=10)
req_bm25  = AnnSearchRequest(data=[query_text], anns_field="sparse_vector",
                             param={"metric_type": "BM25"}, limit=10)

hits = client.hybrid_search(collection_name=..., reqs=[req_dense, req_bm25],
                            ranker=RRFRanker(60), limit=5, output_fields=[...])
```

**★ 过滤条件写法（见上文坑 2，hybrid 特有）**：

```python
# ❌ 错 —— 被静默丢弃，过滤完全不生效，且不报错
client.hybrid_search(reqs=[...], ranker=..., filter='dimension == "opinion"')

# ✅ 对 —— 塞进【每个】 AnnSearchRequest
req_dense = AnnSearchRequest(..., expr='dimension == "opinion"')
req_bm25  = AnnSearchRequest(..., expr='dimension == "opinion"')
```

单路 `search()` 的 `filter=` 是顶层参数，能正常用；只有 hybrid 不一样。

### 该看什么

脚本输出 8 组结果，重点对比四类：

1. `RRFRanker(k=60)` vs `k=1`（分数跨度 0.625→0.375）vs `k=1000`（全为 0.0020，差异抹平）
2. `WeightedRanker(0.7,0.3)` vs `WeightedRanker(0.3,0.7)` —— 注意分数**绝对值**，偏关键词那组普遍更高
3. **★ 过滤生效验证**：脚本末尾有"自动校验"块，直接告你过滤有没有真的生效
4. **各路候选数 limit**：`2` / `5` / `10` 对最终结果的影响

### 该得什么结论

| 结论 | 说明 |
|---|---|
| **RRF 只看排名，不看分数** | 公式 `Σ 1/(k+rank)`，对 dense/sparse 分数尺度差异免疫 |
| **k 越小越"独裁"** | k=1 时单路第1名权重 1.0，第2名只有 0.5；k=60 时 1/61≈0.0164 vs 1/62≈0.0161；k 极大时退化成"谁被更多路命中谁靠前" |
| **本项目默认用 RRF 不用 Weighted** | dense 是 COSINE(0~1)，BM25 分数是另一套尺度（可到几十），不可比。WeightedRanker 依赖归一化，调不好比 RRF 更差 |
| **候选数 limit 要 ≥ 最终 top-k 的 2-3 倍** | 候选池太小 → 两路交集少 → 融合没意义。生产常用 50-100（留给 Reranker 重排） |
| **filter 在向量检索内部执行** | 不是先查后筛。filter 筛掉 99% 数据时，HNSW 图遍历找不到足够候选，召回率骤降 |

---

## 第 4 步：检索评估与调优 —— **本项目真正要的能力**

```bash
python step4_evaluate.py
```

这一步才是"会根据结果优化检索"。前 3 步是"会用"，这步是"会调"。

### 4.1 建 Ground Truth 评测集

真实项目里是人工标注：`query → 相关 chunk 列表`（用 `doc_id + chunk_offset` 唯一标识）。

推荐做法：
1. 从真实业务场景出 50-100 条 query
2. 每条标注 2-5 个相关 chunk
3. 存在 `data/eval_set.json`，脚本里改成从文件读

**没有评测集，后面所有调优都是玄学。**

### 4.2 四个核心指标

| 指标 | 看什么 | 本项目重点 |
|---|---|---|
| **Recall@K** | 前 K 条召回多少相关项 | 主指标，四维度抽取的检索完整性 |
| **nDCG@K** | 排序质量（相关的排得靠前吗） | 决定喂给 LLM 的上下文质量 |
| **QPS** | 吞吐 | 定时任务批量跑时看 |
| **P99 Latency** | 长尾延迟 | 消费轮交互问答看这个，**不看平均值** |

### 4.3 Recall-Latency 曲线找拐点

脚本会扫 `ef = [16, 32, 64, 128, 256, 512]`，并在满足"收益 <2% 且代价 >50%"时标注拐点。

**这是最实用的调优手段，比背参数表有用一百倍。**

关键：**绝不能抄别人的参数值**。拐点取决于你的数据分布、向量维度、数据量，必须自己扫。

### 4.4 参数调优速查

| 参数 | 影响 | 方向 |
|---|---|---|
| HNSW `M` | 图连接度 | 大→召回高、内存大 |
| HNSW `efConstruction` | 建索引质量 | 大→索引好、建得慢 |
| HNSW `ef` / `efSearch` | 搜索广度 | 大→召回高、延迟高（**查询期调**） |
| IVF `nlist` | 聚类数 | 建议 √N |
| IVF `nprobe` | 探查簇数 | 大→召回高、延迟高（**查询期调**） |
| Dense `limit`（hybrid 各路候选数） | 融合候选池 | 大→融合更好、延迟高 |
| `drop_ratio_search`（sparse） | 裁剪低频词 | 大→快、召回降 |

### 4.5 分层排查反射（**背下来**）

```
召回差 ?
  ├─ 1. 先查 filter 是否与 query 语义匹配      ← 最常见
  ├─ 2. 再查 eval set 标注是否正确
  ├─ 3. 再看 embedding 模型是否适配领域
  └─ 4. 最后才调索引参数（ef / nprobe）

排序差 ?
  ├─ 1. 先看 Ranker 选型（RRF vs Weighted）
  ├─ 2. 再看是否需要加 Reranker（Qwen3-Reranker-0.6B）
  └─ 3. 最后调两路候选数 limit

延迟高 ?
  ├─ 1. 先查标量索引建没建                    ← 最常见，没建=全表扫描
  ├─ 2. 再看 ef / nprobe 是否过大
  ├─ 3. 再看数据是否溢出到磁盘（mmap 状态）
  └─ 4. 最后看 segment 数量（大量小 segment 拖慢查询）
```

**顺序不能乱。** 新手 90% 时间浪费在瞎调 `ef`，而问题其实在 `filter`。

### 4.6 脚本内置的两个演练

- **filter 加严演练**：`只 meta 维度` 时 Recall 骤降 → 让你亲眼看到"评测集与 filter 意图不匹配"长什么样
- **逐 query 诊断**：标出"需排查"的 query，这是你下一步调优的目标

---

## 第 5 步：Embedding 质量对检索的影响（已配好，可直接验证）

**当前已接入真实 embedding（BAAI/bge-m3，硅基流动）。** 用真实模型跑一遍 step2，你会看到正确的语义排序：

```
query = '公司新能源业务增长情况如何'
  1. +0.7708 | 该公司2026年第一季度营业收入达到12.3亿元，同比增长45%      ← 最相关
  2. +0.7123 | 管理层在业绩会上表示，新能源板块毛利率提升至28%
  3. +0.6579 | 国家发改委发布关于促进新能源产业健康发展的指导意见
  4. +0.6401 | 竞品公司在2026年Q1实现营收8.7亿元，同比下滑12%          ← ★ 注意这条
  5. +0.6355 | 宁德时代、比亚迪、隆基绿能是该公司新能源业务的主要供应商
  ...
 10. +0.4662 | 该竞品的组织架构调整过于频繁，导致战略执行连续性不足
```

**★ 第 4 条是 dense 缺陷的活证据：**

query 问的是"**增长**情况"，而这条讲的是"同比**下滑**12%" —— 语义方向完全相反，但 dense 给了它 0.6401、排到了第 4 名。

为什么？因为它含"营收""同比""新能源业务"这些**共现词**，向量空间里距离被拉近了。**dense 分不清"涨"和"跌"。**

这不是 Milvus 的 bug，是稠密向量检索的固有局限。**补它的正是 BM25** —— 如果 query 里明确带"下滑"，BM25 会精准锁定这条。

### 你要做的对比实验

把 step2 的 query 从 `'公司新能源业务增长情况如何'` 改成：

| 改成的 query | 观察什么 |
|---|---|
| `'公司新能源业务下滑情况'` | dense 会不会还把"增长45%"排前面？ |
| `'价格战风险'` | dense 和 BM25 谁更准？ |
| `'隆基绿能'` | dense 能精准锁定吗？BM25 能 |

**这就是分层排查里第 3 层"embedding 模型是否适配领域"的验证手段** —— 看到排序违背直觉时，先怀疑向量质量，再怀疑索引参数。

### 换 embedding 模型时要注意

1. **维度必须和 schema 一致** —— 换模型后 dimension 变了，collection 必须重建（step1 重跑）
2. **清了缓存** —— `data/embed_cache/` 是按模型名 + 文本做 key 的，换模型不会串，但旧向量要重算
3. **重跑顺序**：step1（重建表）→ step2（重灌数据）→ step3/4

---

## 第 6 步：加重排（当 nDCG 上不去时的标准解法）

如果第 4 步发现 **Recall 还行但 nDCG 低**（召回到了但排序不好），加 Reranker：

```python
# 1. hybrid_search 时多取候选（比如 limit=50）
# 2. 用 Qwen3-Reranker-0.6B 对候选重排
# 3. 取 Top-K 喂 LLM
```

重排链路是**本项目预留的优化空间**，PRD 里没写，但消费轮的产出质量直接取决于它。

---

## 附：本项目落地检查清单

| 检查项 | 对应 PRD |
|---|---|
| 四维度字段齐备，`dimension` 标量索引已建 | FR-1.3 |
| `doc_id` + `chunk_offset` + `dimension` 三元溯源可用 | FR-1.5 / 三元溯源 |
| `skill_id` 字段可过滤 | FR-1.5 |
| BM25 + 中文 analyzer 生效 | 消费轮中文检索 |
| hybrid_search + RRF 为默认检索路径 | FR-2.2 |
| 评测集存在且 Recall@K / nDCG@K 有基线记录 | 效果可回归 |
| 参数（ef / limit）有扫过拐点的记录 | 性能可解释 |

---

## 文件清单

```
learning/milvus/
├── docker-compose.yml            # Milvus standalone 部署
├── MILVUS_学习路径.md             # 本文件
└── scripts/
    ├── common.py                 # 公共配置 + 模拟数据 + 打印工具
    ├── step1_schema.py           # 建 collection（四维度 + 三元溯源 + BM25）
    ├── step2_insert_and_probe.py # 灌数据 + 三路检索对比
    ├── step3_hybrid_search.py    # Hybrid + RRF/Weighted 对比
    └── step4_evaluate.py         # 评估指标 + 参数扫描 + 排查演练
```

运行顺序：`step1 → step2 → step3 → step4`（step2 必须在 step1 之后）
