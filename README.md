# ai-novel-lab

AI 小说自动学习系统：自动发现热门小说 → 评分选优 → 找合法公开资料 →
深度理解（人物/剧情/世界观/风格）→ 带溯源的知识库 → 训练数据 → SFT/LoRA。

路线立场：先让 AI 学会"什么小说值得学、怎么研究一本小说、怎么把小说拆成
知识"，最后才是让它写。

## 版权红线

- 发现层只读公开榜单元数据，不碰正文。
- 资料层只收合法公开资料，不绕付费墙/登录墙/反爬保护。
- 训练数据只从 authorized / user_provided / open_license 语料生成。

详见 `docs/novel-system-architecture.md`。

## 当前进度（P4 ✅ mock 验收）

```powershell
python -m novel_lab.cli discover            # 多平台榜单轮询落库
python -m novel_lab.cli top --limit 10      # PopularityScore TopN
python -m novel_lab.cli collect --top 5     # 书页元数据采集（三级判重）
python -m novel_lab.cli ingest <目录> --novel-title "书名"   # 语料登记
python -m novel_lab.cli understand --mock   # 预筛+结构化（干跑，零LLM）
python -m novel_lab.cli extract --mock      # 知识抽取（确定性部分，零LLM）
python -m novel_lab.cli knowledge --novel-title "双月降临"  # 查询知识库
python -m novel_lab.cli graph --novel-title "双月降临"      # 知识图谱（人物/关系）
python -m novel_lab.cli gaps --novel-title "双月降临"       # 知识缺口体检（--llm 加规则冲突扫描）
python -m novel_lab.cli dataset --novel-title "双月降临"    # 生成训练数据集（六类样本 jsonl）
python -m novel_lab.cli quality-gate --novel-title "双月降临"  # 质检+切分 train/val/test
python -m novel_lab.cli export-lf       # 导出 LLaMA-Factory 训练包
python -m novel_lab.cli status          # P10：流水线全景状态
python -m novel_lab.cli eval-gold --mock    # 校准集评测
python -m unittest discover -v              # 测试
```

**接真模型**（P5 需要）：`novel_lab/config.toml` 的 `[llm]` 填
`base_url`/`api_key`/`model`（OpenAI 兼容，智谱/DeepSeek 均可）并
`enabled = true`，然后把 `--mock` 去掉即可；回执复用+预算熔断自动生效。

## 阶段路线（全部完成 ✅ 2026-10-18）

P0 AIHOT 架构分析 ✅ · P1 架构+骨架+落库闭环 ✅ · P2 起点榜+增长率 ✅ ·
P3 书页采集+判重+Rights Gate ✅ · P4 预筛/结构化+校准集 ✅ ·
P5 知识抽取（确定性+LLM 双路，校准 8/8）✅ · P6 知识图谱+缺口体检 ✅ ·
P7 Dataset Factory（343 条样本）✅ · P8 Quality Gate（273/35/35）✅ ·
P9 LLaMA Factory 适配 ✅ · P10 全景 Demo（status 命令）✅

## 依赖

Python 3.11+（tomllib），`pip install requests`。
