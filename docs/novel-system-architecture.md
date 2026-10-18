# AI 小说自动学习系统 · 架构设计（P1）

> 2026-10-16 · 对应总路线 P1。
> 路线立场：先让 AI 学会"什么小说值得学、怎么研究一本小说、怎么把小说拆成
> 知识"，最后才是让它写。参考架构分析见
> `webnovel-house-style/docs/aihot-for-novel-analysis.md`（AIHOT 机制借鉴）。

## 1. 目标与边界

系统不靠人工指定书目，而是：

```text
自动发现热门小说 → 评分选优 → 找合法公开资料 → 清洗去重 →
深度理解（人物/剧情/世界观/风格）→ 带溯源的知识库 →
自动生成训练数据 → 质量门禁 → SFT/LoRA/QLoRA
```

**版权红线（不可妥协的架构差异）**：

- 发现层只读公开榜单元数据（书名/排名/在读/字数/更新时间/简介）。
- 资料层只收合法公开资料（书评/百科/访谈/公开试读），不绕付费墙、登录墙、
  反爬保护；不做批量正文抓取。
- 训练数据只能从 rights ∈ {authorized, user_provided, open_license} 的语料生成。
- 知识库存事实与分析（可溯源、短引用+链接），不存盗版全文。

**P1 暂不做**（沿用总路线"第一阶段暂时不要做"清单）：自动训练、部署、
多 Agent、自动写小说、全网抓取、绕站限制。

## 2. 总体架构

```text
┌──────────────────────────────────────────────────────┐
│ Discovery 层     榜单轮询（番茄74板块→候选池）【已建成】 │
├──────────────────────────────────────────────────────┤
│ Popularity 层    多来源独立计数+快照增长 → TopN        │
├──────────────────────────────────────────────────────┤
│ Ingestion 层     资料采集→判重→清洗→Rights Gate【P3】  │
├──────────────────────────────────────────────────────┤
│ Understanding 层 预筛→结构化→知识抽取（LLM，回执+熔断） │
├──────────────────────────────────────────────────────┤
│ Knowledge 层     SQLite：novel/document/knowledge 溯源 │
├──────────────────────────────────────────────────────┤
│ Dataset 层       样本工厂→质量门禁→train/val/test      │
├──────────────────────────────────────────────────────┤
│ Training 层      LLaMA Factory 适配【后期】            │
└──────────────────────────────────────────────────────┘
```

分层规则（借鉴 AIHOT 不变规则）：

1. **流水线不直接调模型**：所有 LLM 调用走 `llm/client.py`，带回执与预算熔断；
   回执未复用前重试不重复计费。
2. **知识必带溯源**：knowledge 表每行必须挂 document（榜单观察也作为
   rank_snapshot 型 document 入库），禁止无来源事实。
3. **人工修正不被覆盖**：knowledge 表 `human_verified` 行自动流程只增不删。
4. **提示词版本 = 内容哈希**：改提示词只影响新资料，已抽取知识不重算。

## 3. 目录结构

```text
ai-novel-lab/
├── docs/                      # 架构文档
├── novel_lab/
│   ├── config.toml            # 全部可调参数
│   ├── settings.py            # 配置加载 / 路径
│   ├── models.py              # NovelRecord 等数据结构
│   ├── db.py                  # SQLite schema + 读写
│   ├── popularity.py          # 跨来源热度聚合
│   ├── cli.py                 # discover / top 子命令
│   ├── discovery/             # 榜单适配器（种子来自调研引擎）
│   │   ├── fanqie.py
│   │   └── fanqie_boards.py
│   └── llm/
│       └── client.py          # 回执+熔断客户端（P1 默认禁用）
├── tests/                     # 单测 + 真实页面 fixture
├── data/                      # novel_lab.db（gitignore）
└── README.md
```

技术栈：Python 3.12 + SQLite + requests。MVP 不引入 Docker/PostgreSQL/
消息队列；数据量大或需要常驻调度时再迁移。

## 4. 数据模型（SQLite）

四张业务表 + 一张回执表：

```sql
-- 书：一书一行
CREATE TABLE novels (
    id INTEGER PRIMARY KEY,
    platform TEXT NOT NULL,            -- fanqie / qidian / ...
    book_id  TEXT NOT NULL,
    title    TEXT NOT NULL,
    author   TEXT DEFAULT '',
    url      TEXT DEFAULT '',
    first_seen_at TEXT, last_seen_at TEXT,
    UNIQUE(platform, book_id)
);

-- 榜单观察：每次轮询每个板块的观察记录（快照的表化，增长率的数据底座）
CREATE TABLE observations (
    id INTEGER PRIMARY KEY,
    novel_rowid INTEGER NOT NULL REFERENCES novels(id),
    board TEXT NOT NULL,               -- "1_2_751"
    gender TEXT, rank_type TEXT, category TEXT,
    rank INTEGER, read_count REAL,
    word_count INTEGER, chapter_count INTEGER,
    creation_status INTEGER,
    updated_at TEXT, observed_at TEXT NOT NULL,
    UNIQUE(novel_rowid, board, observed_at)
);

-- 资料：一份公开资料一行（P3 起使用；P1 先建表）
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    novel_rowid INTEGER REFERENCES novels(id),
    source_url TEXT NOT NULL UNIQUE,
    source_type TEXT,                  -- rank_snapshot/blurb/review/wiki/interview/public_chapter/authorized_text/user_provided
    rights TEXT NOT NULL DEFAULT 'metadata_only',
    title TEXT, published_at TEXT, crawl_time TEXT,
    content_hash TEXT, raw_path TEXT, clean_path TEXT
);

-- 知识：一条结构化知识一行（P5 起使用；P1 先建表）
CREATE TABLE knowledge (
    id INTEGER PRIMARY KEY,
    novel_rowid INTEGER NOT NULL REFERENCES novels(id),
    document_id INTEGER NOT NULL REFERENCES documents(id),   -- 溯源，必填
    kind TEXT NOT NULL,                -- character/event/world_rule/relationship/style_fact
    payload TEXT NOT NULL,             -- JSON
    confidence REAL, extractor TEXT, prompt_version TEXT,
    human_verified INTEGER DEFAULT 0,
    created_at TEXT
);

-- LLM 回执（借鉴 AIHOT receipts）
CREATE TABLE llm_receipts (
    id INTEGER PRIMARY KEY,
    purpose TEXT, model TEXT, prompt_hash TEXT,
    status TEXT, response TEXT, created_at TEXT
);
```

## 5. 资料类型白名单与 rights 字段

| source_type | 内容 | rights 默认值 | 可进训练集 |
|---|---|---|---|
| rank_snapshot | 榜单页元数据 | metadata_only | 否（可做特征） |
| blurb | 平台公开简介 | metadata_only | 否 |
| review | 公开书评/长评 | metadata_only（摘录+链接） | 否 |
| wiki | 百科词条 | metadata_only | 否 |
| interview | 作者公开访谈 | metadata_only | 否 |
| discussion | 公开论坛讨论 | metadata_only | 否 |
| public_chapter | 平台免费试读 | platform_free | 否（默认） |
| authorized_text | 明确授权语料 | authorized | **是** |
| user_provided | 自己拥有权利的文本 | user_provided | **是** |
| open_license | 开放许可文本 | open_license | **是** |

规则：Dataset Factory 只消费 rights ∈ {authorized, user_provided, open_license}
的 documents；Knowledge 层可用 metadata_only 资料抽事实（短引用+溯源），
但生成"写作任务"类样本时只引用可训练语料。

## 6. PopularityScore v1

输入（全部来自已有观察，无 LLM）：

```text
score = board_weight  × norm(上榜板块数)
      + read_weight   × norm(log( max在读 ))
      + rank_weight   × norm( 1 / best_rank )
      + growth_weight × norm(快照间在读增长率)   # 快照≥2期后启用，当前=0
```

- 归一化在候选池内做 min-max；缺分量权重重归一（与调研引擎热度分同规则）。
- 当前番茄一书一板块，board 信号≈常数；**等 P2/P3 接入第二来源
  （起点榜/豆瓣/讨论量）后该分量才有区分度**——这正是"多来源独立计数"
  优于单榜的原因。
- 增长率：observations 表按 (novel, board) 相邻两期算 read 增量。

## 7. LLM 调用层（P1 只建骨架，默认禁用）

`novel_lab/llm/client.py`：

- `enabled = false` 时任何调用抛 `LLMDisabledError`——P1 闭环完全不花钱。
- 回执复用：同 (purpose, prompt_hash, model) 的成功回执直接返回，重试不重复扣费。
- 预算熔断：每分钟/小时/天请求上限（config），超限抛 `BudgetExceededError`。
- transport 可注入（测试用假服务，不访问外网）。

P5 接 Understanding 层时才打开，且每一步（预筛/结构化/抽取）独立 purpose
与独立模型路由（借鉴 AIHOT 分步换模型）。

## 8. 阶段路线

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | AIHOT 架构分析 | ✅ 完成 |
| P1 | 架构设计 + 仓库骨架 + 榜单→候选→落库闭环 | ✅ 完成 |
| P2 | Discovery 增强：起点m站榜 + 增长率启用 | ✅ 完成（2026-10-16） |
| P3 | 资料采集器 + 判重 + Rights Gate 实装 | ✅ 完成（2026-10-16） |
| P4 | 预筛/结构化 prompt + 校准集 gold.jsonl | ✅ 完成（2026-10-16，mock 干跑验收） |
| P5 | 知识抽取（人物/剧情/世界观/风格）→ knowledge 表 | ✅ 完成（2026-10-17，确定性+LLM 双路验收） |
| P6 | 知识图谱视图 + 知识缺口发现 | ✅ 完成（2026-10-18） |

**P5 记录**：
- **分层抽取策略**：能确定性算的不给 LLM——风格统计（句长/段长/对话占比）
  与人物出场（`data/character_dicts/<书名>.txt` 词典匹配，一行一名）零
  幻觉零成本；书页 profile 是结构化 JSON 直接映射为 genre_fact/market_fact；
  语义抽取（人物身份/关系/剧情事件/世界观规则/手法观察）走 LLM，
  提示词外置 `prompts/extract_chapter.txt`，**每条必须带原文 quote，无
  quote 一律丢弃（防幻觉硬门禁）**。
- `knowledge_extract.py`：写入幂等（同 document+extractor+version 先删
  后插）；抽取只处理 P4 预筛 PASS 的文档；每条知识必挂 document 溯源。
- CLI：`extract [--mock]`（mock=仅确定性）/ `knowledge [--novel-title]`
  （分布 + 人物出场聚合）。
- 验收（确定性）：25 文档 → **125 条知识**（人物出场 82 + 风格 20 +
  题材 5 + 市场 18），全部有溯源；《双月降临》ch1-20 人物分布：林晚
  20章/216次、李哥 19/133、袁立军 14/110、沈确 16/90、老郑 7/38、
  小高 6/33。测试 33/33。
- **LLM 真跑验收（2026-10-17，本地 sick-cloud Ollama qwen3.8:27b-iq3s，
  SSH 隧道 + think:false 关思考，40-90s/次）**：校准集 8/8 满分
  （prefilter 1.0 / doc_kind 1.0 / coverage 1.0，含 mock 判不出的
  广告与薄内容负例）；20 章语义抽取 → 451 条候选，**quote 逐字校验
  门禁**（非连续原文拼接一律丢弃，含存量清洗 13 条）后留 438 条。
  最终知识库 **563 条**：character 180（LLM 补出词典外人物：小周/
  梁老板/周老板/S…）、event 90、relationship 73、world_rule 98、
  style_note 79 + 确定性 125。抽检 7/8 逐字真实促成了门禁 v2。
- LLM 语义抽取就绪度：提示词+解析+门禁已测（FakeClient），接 key 后
  `extract`（不带 --mock）即用。
| P7 | Dataset Factory（六类样本） | ✅ 完成（2026-10-18） |
| P8 | Dataset Quality Gate + train/val/test | ✅ 完成（2026-10-18） |
| P9 | LLaMA Factory Adapter | ✅ 完成（2026-10-18） |
| P10 | 10 本书 MVP 完整 Demo | ✅ 完成（2026-10-18，`status` 全景命令） |

**P3 记录**：
- `collectors.py`：书页元数据采集（番茄 `/page/{id}`、起点m站 `/book/{id}/`），
  统一为 book_profile，rights=metadata_only。**番茄书页的书名/简介是干净文本**
  （无榜单页的字体反爬乱码），顺带解决了 PUA 显示问题；categoryV2 是双重
  编码的 JSON 字符串需二次解析。起点书页富字段：月票/收藏/总推荐/点击/
  章节数/标签/荣誉。
- `ingest.py`：三级判重——URL（UNIQUE）→ 内容哈希（同书范围）→ 标题归一化
  （仅 novel_rowid 非空时启用，无归属文档不做标题判重防误合并）；rights
  白名单硬校验，未知值直接拒绝。
- CLI：`collect --top N [--platform]` / `ingest <目录> --novel-title X
  [--rights]` / `docs`。用户自有语料经 ingest 登记为 user_provided，
  训练就绪计数即时可见（training_ready）。
- 验收：Top5 书页采集 5/5、重跑全判重；《双月降临》ch1-20 登记为可训练
  语料；documents 25 份（5 metadata_only + 20 user_provided）。测试 16/16。

**P2 记录**：
- 起点接入走 **m 站**（`m.qidian.com/rank/<slug>/`，pageContext JSON 服务端直出）。
  PC 站有 JS 反爬挑战（HTTP 202 + probe.js），按红线不绕；女生网 m.qdmm.com
  被试过被 WAF 拦，暂不接。
- 已确认板块：`yuepiao` 月票榜（rankCnt=月票数）、`readindex` 阅读指数榜、
  `sign` 签约榜、`newfans` 书友榜（rankCnt=新粉数），各 top20。
- read_count 语义随平台/板块不同（番茄=在读、起点=月票/新粉），
  PopularityScore 的读数分量**按平台内部归一化**再合并。
- 增长率：observations ≥2 期后按 (novel, board) 最近两期算相对增长，
  growth_weight=0.15 已启用；单期内增长接近 0 属正常，跨天快照后有意义。
- 起点一书可出现在多个板块（board_count 2~3），多来源信号开始有区分度。

**P4 记录**：
- 提示词外置 `novel_lab/understanding/prompts/*.txt`，版本=内容哈希前 12 位
  （借鉴 AIHOT：改词即换版，已分析文档不重算，doc_structures 按
  (document_id, prompt_versions) 唯一）。
- 管线 `UnderstandingPipeline`：预筛（PASS/BLOCK/UNKNOWN，宽进严出）→
  BLOCK 跳过结构化省一次调用 → 结构化（doc_kind/aspects/facts，每条
  fact 必须带 ≤20 字原文 source_quote，禁止编造）。输出走稳健 JSON
  抽取（兼容代码块/闲话/截断），解析失败标 parse_error 不中断批次。
- **mock 干跑**：`--mock` 用确定性假分析跑通全链路（零 LLM 花费，
  is_mock=1 标记），真跑只需 config [llm] 填 base_url/api_key/model 并
  enabled=true（回执复用+预算熔断自动生效）。
- 校准 `eval/gold.jsonl` + `eval-gold` 命令（SelectBench 思想简化版）：
  coverage / prefilter 准确率 / doc_kind 准确率。种子集 8 条（5 真实
  书页 + 3 负例），标注者为 ai-seed **待用户复核**——mock 评测
  prefilter 0.75 恰好暴露了假分析判不出广告/薄内容的差距，证明校准
  循环有效。
- 验收：测试 27/27；25 份真实文档 mock 全管线落 doc_structures。

## 9. P1 验收标准

**P8-P10 记录（2026-10-18，路线收官）**：
- **P8 质检**`quality_gate.py`：五项检查（格式/完全与近重复/事实溯源
  /来源/评分）+分层切分（按类别 8:1:1 固定种子）。**事实溯源三类硬校验**：
  引用型样本 quote 必须在 output、出场次数与知识库重算一致、写作切片
  与原文段落连续（测试故意造假的三类样本全部被拦）。实测 343 条
  全过（0 淘汰——上游逐字门禁已挡住脏数据；23 条薄答案扣分保留），
  切分 273/35/35。
- **P9 训练适配**`export_llamafactory.py`：alpaca 纯格式导出 +
  dataset_info 注册片段 + QLoRA yaml 模板（RTX 5080 16GB 可跑，
  343 条配 3 epoch 防过拟合）+ 训练步骤文档（含扩量路径说明：
  文风微调建议 1k-10k 切片，当前 343 属流程验证版规模）。
- **P10 全景**`pipeline_status.py` → `status` 命令：一屏验明
  发现→资料→理解→知识→数据→训练包六层产物链，含全库引用逐字
  复检（当前残留 0）。收官快照：803 本书/1530 次观察/563 条知识/
  343 条样本/273-35-35 切分。测试 45/45。

**P7 记录（2026-10-18）**：
- `dataset_factory.py` 六类样本全部确定性生成（零 LLM 成本）：
  qa（人物出场统计+关系问答带引用）、character（LLM 身份聚合+引用）、
  plot（事件起因结果+引用）、world（规则+原文依据）、summary（章级
  事件聚合要点）、writing（**自监督续写切片**：input=前文段落、
  output=原文后续，每章普通+对白锚点两片，模型学作者自己的文风）。
- **Rights Gate 数据集层落地**：只从 authorized/user_provided/
  open_license 文档及其知识生成；metadata_only 的书页资料被硬性
  排除（有测试）。
- 指令里的章节引用统一转中文（ch2_deai→第二章）。
- CLI `dataset --novel-title`。实测：《双月降临》→ **343 条样本**
  （world 98 / plot 90 / qa 79 / writing 39 / character 19 /
  summary 18），20 份可训练文档 + 540 条知识，输出
  `data/datasets/v1/samples.jsonl` + metadata.json。测试 41/41。

**P6 记录（2026-10-18）**：
- `knowledge_graph.py`：实体归一（括注变体"袁经理（袁立军）"→本名、
  前缀保守合并、词典统计与 LLM 身份合流）→ 人物索引；关系边按
  （甲,乙）聚合带覆盖章区间，悬空端点标记；缺口规则 R1主角断档/
  R2早期沉底/R3悬空关系/R4泛称幽灵 全确定性；R5 规则冲突走 LLM
  **分块扫描**（98 条对 27B 单次会超 600s，改每块 25 条限报 5 对，
  局部编号映射回全局行号）。
- CLI：`graph --novel-title`（人物图+关系边）/ `gaps --novel-title
  [--llm]`（体检报告）。
- 双月降临实测（563 条知识）：**机器自动复现人工审计的核心发现**
  ——"S 首次第3章、最后第3章，此后17章未再出现"；另报出周小雨
  ch5 后沉底、林家亲属闪回人物悬空、恒禾商贸/47号等非人物端点、
  泛称幽灵 1 项，共 17 项发现。R5 报 1 条冲突候选（"十件容量 vs
  10m³"），经人工核对为**合同上限与物理容量的语义混淆**——非真
  吃书，但说明体检报告必须保留人工复核环节。测试 39/39。

1. `python -m novel_lab.cli discover`：轮询全部板块，novels + observations 落库。
2. `python -m novel_lab.cli top`：按 PopularityScore 输出 TopN，含上榜信息。
3. 单测覆盖：适配器解析、落库幂等（重复 discover 不产生重复书）、
   LLM 禁用/熔断/回执复用。
4. 全程零 LLM 调用。
