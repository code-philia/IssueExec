<div align="center">

# IssueExec

### 面向软件工程问题定位的测试驱动方法

[![ISSTA 2026](https://img.shields.io/badge/ISSTA-2026-6f42c1.svg)](https://conf.researchr.org/details/issta-2026/issta-2026-research-papers/199/IssueExec-A-Test-Driven-Approach-for-Localizing-Software-Engineering-Issues)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![SWE-bench Lite](https://img.shields.io/badge/Benchmark-SWE--bench%20Lite-1f883d.svg)](https://www.swebench.com/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.17286-b31b1b.svg)](https://arxiv.org/abs/2607.17286)

**将测试视为可执行需求，用于连接 issue 与代码实现。**

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

IssueExec 是论文 **“IssueExec: A Test-Driven Approach for Localizing Software Engineering Issues”** 的配套 artifact，论文已被 **ISSTA 2026 接收**。该框架将测试套件视为可执行规格：首先用与 issue 相关的测试建立需求层面的语义桥梁，再利用这些测试的运行时执行轨迹，将语义证据落到具体的代码位置。

> **论文：** Jiawei Liu, Yun Lin, Chenyan Liu, Yu Qian, Yiming Liu, Jiaxin Chang, Weinan Zhang, and Linpeng Huang. [ISSTA 2026 论文详情](https://conf.researchr.org/details/issta-2026/issta-2026-research-papers/199/IssueExec-A-Test-Driven-Approach-for-Localizing-Software-Engineering-Issues) · [arXiv:2607.17286](https://arxiv.org/abs/2607.17286)

## 为什么需要 IssueExec？

直接将 issue 与代码进行匹配经常会失败：issue 描述的是行为需求，而代码标识符通常体现的是实现结构。IssueExec 采用如下两跳证据链：

```text
Issue 描述 ──语义检索──▶ 相关测试
                         │
                         └─ 执行轨迹 ─▶ 候选代码位置
                                          │
                         代码结构 + 补充检索 + 重排序
                                          ▼
                                      排序后的修改位置
```

框架主要解决两个问题：

- **领域术语鸿沟：** 从历史代码变更中挖掘缩写、API 别名等项目知识，增强测试表示，使 issue 更容易检索到真正相关的测试。
- **执行轨迹噪声：** 对调用层级进行分析，过滤偶然执行的基础设施代码，突出与需求最相关的实现位置。

## Dynamic trace collection

IssueExec 批量动态测试轨迹的收集工具位于配套仓库：

**[Dynamic trace collection](https://github.com/AWGiaGia/swe-tools)**

该工具在 SWE-bench Docker 环境中批量运行测试，注册 Python `call`/`return` hook，过滤插桩噪声，并导出每个测试对应的 `tests-info.json` 与 `traces.json`。IssueExec 在后续测试驱动定位阶段读取这些轨迹。安装方式、Docker 编排、输出 schema 和复现实验说明请参考该仓库的 README。

## 实验结果

在 **SWE-bench Lite** 上，IssueExec 相比最强基线提升：

| 指标 | 提升 |
| --- | ---: |
| 文件级 Recall@1 | **17.78%** |
| 模块级 Recall@1 | **25.98%** |
| 函数级 Recall@1 | **41.57%** |
| Agentless 端到端 issue 修复率 | **17.72%** |

配套研究还表明：现有测试覆盖了 **96.98% 的真实修改文件**；在 18 个仓库中，`issue → tests → code` 两跳路径在 **82.4% 的案例**里具有比直接匹配更强的语义连接。

## 仓库结构

| 路径 | 用途 |
| --- | --- |
| [`localize.py`](localize.py) | 命令行入口及多阶段定位流程 |
| [`Localizer.py`](Localizer.py) | 测试检索、轨迹分析、候选定位与重排序组件 |
| [`prompt.py`](prompt.py) | 各阶段使用的提示词模板 |
| [`merge.py`](merge.py) | 合并测试驱动定位与补充检索结果 |
| [`util/`](util/) | 数据准备、仓库索引、API 封装、领域知识和后处理工具 |
| [`example_data.tar.gz`](example_data.tar.gz) | 示例 issue 输入及辅助 artifact |
| [`requirements.txt`](requirements.txt) | Python 依赖 |

## 快速开始

### 1. 安装依赖

```bash
git clone git@github.com:AWGiaGia/IssueExec.git
cd IssueExec
python -m pip install -r requirements.txt
```

IssueExec 支持 OpenAI 兼容、DeepSeek 兼容和 Anthropic 兼容的模型后端。请根据所选后端配置凭据，不要把密钥写入仓库：

```bash
# OpenAI 兼容后端示例
export OPENAI_BASE_URL="<openai_base_url>"
export OPENAI_API_KEY="<your_api_key>"
```

### 2. 准备示例数据

```bash
tar -xzf example_data.tar.gz
```

该命令会生成 `example_data/issues/test` 以及下面示例命令需要的覆盖图等辅助文件。

### 3. 运行完整定位流程

每个阶段都会在 `example/<stage>/` 下写入 `loc_outputs.jsonl`；后续阶段通过 `--start_file` 读取上一个阶段的结果。

#### 阶段 1：检索需求相关测试

```bash
python localize.py \
  --stage related_tests_retrieval \
  --output_folder example \
  --num_threads 2 \
  --skip_existing \
  --model gpt-4o-2024-05-13 \
  --backend openai \
  --top_n 5 \
  --dataset example_data/issues/test \
  --coverage_graph_path example_data/coverage_graph
```

输出：`example/related_tests_retrieval/loc_outputs.jsonl`。

#### 阶段 2：分析测试假阴性与执行轨迹

```bash
python localize.py \
  --stage blind_spot_analysis \
  --output_folder example \
  --num_threads 2 \
  --skip_existing \
  --start_file example/related_tests_retrieval/loc_outputs.jsonl \
  --model gpt-4o-2024-05-13 \
  --backend openai \
  --dataset example_data/issues/test
```

输出：`example/blind_spot_analysis/loc_outputs.jsonl`。

#### 阶段 3：补充检索

```bash
python localize.py \
  --stage suppletory_retrieval \
  --output_folder example \
  --num_threads 2 \
  --skip_existing \
  --start_file example/blind_spot_analysis/loc_outputs.jsonl \
  --model gpt-4o-2024-05-13 \
  --backend openai \
  --dataset example_data/issues/test
```

输出：`example/suppletory_retrieval/loc_outputs.jsonl`。

#### 阶段 4：合并候选来源

```bash
python merge.py --target_folder example
```

输出：`example/merge/loc_outputs.jsonl`。

#### 阶段 5：重排序最终修改位置

```bash
python localize.py \
  --stage reranking \
  --output_folder example \
  --num_threads 2 \
  --skip_existing \
  --start_file example/merge/loc_outputs.jsonl \
  --model gpt-4o-2024-05-13 \
  --backend openai \
  --context_expansion \
  --dataset example_data/issues/test
```

最终输出为 `example/reranking/loc_outputs.jsonl`，其中包含可供后续补丁生成使用的排序后代码位置。

## 配置说明

- `--model` 支持 `gpt-4o-2024-05-13`、`gpt-4o-mini-2024-07-18`、`deepseek-coder` 和 `claude-3-5-sonnet-20241022`。
- `--backend` 可选 `openai`、`deepseek` 或 `anthropic`，应与模型提供方匹配。
- `--use_online_domain_knowledge` 只为 BM25 筛选后的测试实时收集领域知识；也可以通过 `--domain_knowledge_path` 传入预计算的逐实例文件。
- `--context_expansion` 会在重排序阶段展开完整代码上下文；`--suppletory_context_level module` 可将补充检索上下文从文件级切换为模块级。
- `--repo_cache_dir` 控制 SWE-bench 仓库的临时检出目录，默认是 `/tmp/swe_bench_repos`。
- 输出和日志默认被 Git 忽略。请将 API 密钥放在环境变量或不会提交的本地 `.env` 文件中。

## 复现检查清单

- [ ] 根据 `requirements.txt` 安装依赖。
- [ ] 配置所选模型后端和凭据。
- [ ] 解压 `example_data.tar.gz`。
- [ ] 按顺序运行阶段 1–5，并保留每个 `loc_outputs.jsonl` 的路径。
- [ ] 记录每次运行使用的模型、后端、数据集路径和并行度。

该仓库面向研究复现与扩展。API 调用可能产生服务商相关的费用和延迟；中断后可使用 `--skip_existing` 继续运行。

## 引用

```bibtex
@article{liu2026issueexec,
  title   = {IssueExec: A Test-Driven Approach for Localizing Software Engineering Issues},
  author  = {Liu, Jiawei and Lin, Yun and Liu, Chenyan and Qian, Yu and Liu, Yiming and Chang, Jiaxin and Zhang, Weinan and Huang, Linpeng},
  journal = {arXiv preprint arXiv:2607.17286},
  year    = {2026}
}
```

## 联系方式

如有 artifact 使用问题，欢迎提交 [GitHub Issue](https://github.com/AWGiaGia/IssueExec/issues)，或联系论文中列出的作者。
