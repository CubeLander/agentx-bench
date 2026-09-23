# AgentX bench：Agent 场景吞吐测试

[English / 完整运行说明](README.md)

这个仓库使用 **官方 AgentX 256k 数据集和官方回放工具**，测试推理 engine
在长上下文、多轮对话、并行子任务和前缀复用场景下的吞吐与延迟。
我们只做版本固定、启动和结果留存，不另写一套压测调度器。

## 数据集里有什么？

固定版本包含 **393 个会话、68,266 个模型请求**。它来自真实 Claude Code
会话的脱敏轨迹，但不包含原始提示词、代码或工具结果，保留的是：

- 每次请求的输入、输出 token 数；
- 前后请求共享哪些前缀块；
- 请求时间、轮次间等待；
- 主 agent 与子 agent 的分支、并行和汇合关系。

256k 版本已按每个请求 **输入＋输出不超过 256,000 tokens** 做官方过滤。
本仓库使用完整会话池，不再抽样、截断长输入或缩短输出。

## 这些轨迹怎样用来测吞吐？

回放器用目标模型的 tokenizer 生成确定性的合成文本，重建请求长度和前缀关系，
再按照会话依赖发送给实际运行的 engine。**不需要真的启动编程 agent 或执行工具。**

```text
主会话：请求 → 等待工具 → 请求 ───────────→ 后续请求
                           ├─ 子任务 A ─┤
                           └─ 子任务 B ─┘
                         并行执行，按依赖汇合
```

`--concurrency 8` 表示维持 **8 棵活跃的 agent 会话树**，不是固定同时发 8 个 HTTP
请求：子任务展开时，在途请求可以更多；等待工具时，也可能暂时更少。
一次会话树全部结束后，回放器从会话池中选择下一棵继续运行。

这是**闭环回放**：下一次请求要等它依赖的请求完成，再遵守轨迹中的等待时间。
因此 engine 越快，在相同测量窗口内通常能推进更多请求、输出更多 token。
长历史与重复前缀同时考验 prefill、decode、调度和 KV cache；并发增加后，
也可能暴露换出、恢复或重算的代价。是否真的发生 spill，要看服务端观测，不能仅凭并发数断言。

同一轮会话内保留前缀复用；重新回放时加入独立标记，避免重复播放使缓存命中率虚高。

## 测多久？

| 模式 | 预热后的测量时间 | 用途 |
|---|---:|---|
| `smoke` | 15 分钟 | 日常开发回归 |
| `formal` | 60 分钟 | 与官网测量时长一致的正式运行 |

两种模式使用相同数据与预热：固定种子，从会话 25%–75% 的位置选取起点，
先建立前缀状态，再额外预热每条回放通道的 10 个请求；**只改变测量时长**。

15 分钟不是总耗时：首次数据准备、请求重建、预热、最多 30 秒排空和结果导出另计。
完整会话池参与选择，也不意味着一次运行会执行完全部 68,266 个请求。

## 怎么跑？

需要预先安装 `uv`，并准备一个已获授权、正在运行的 OpenAI 兼容推理服务。
本仓库不启动模型，也不分配或占用设备来部署服务。

```bash
git clone git@github.com:CubeLander/agentx-bench.git
cd agentx-bench
uv sync --locked
uv run --frozen python prepare.py

cp examples/target.json .cache/target.json
# 编辑 .cache/target.json，填写真实配置后再运行

# 只检查请求配置，不向服务发送请求
uv run --frozen python bench.py plan --target .cache/target.json \
  --profile smoke --concurrency 8

# 预热后测量 15 分钟；正式版改为 --profile formal
uv run --frozen python bench.py run --target .cache/target.json \
  --profile smoke --concurrency 8
```

目标配置需要填写服务地址、模型和 engine 版本、本地 tokenizer 目录及版本、
硬件配置和 host KV 预算。服务需支持流式返回、token 用量统计和 `ignore_eos`，
上下文容量还要为 chat template 等开销留余量。不要往配置或命令中放密钥。

并发 8 只是示例，不是最佳配置。比较候选版本时保持模型、资源、并发及测量模式一致；
需要吞吐—延迟曲线时，分别运行多个并发点。投机解码、DRAM 配额等条件见
[完整说明](README.md#protocol-choices-and-comparison-boundaries)。

## 结果怎么看？

结果写入 `artifacts/<run>/`：`run.json` 保存运行配置与状态，`harness.log` 查看进度，
`aiperf/` 保留官方指标、逐请求记录和日志，不重新计算一套“更好看”的分数。

- **输出吞吐（tokens/s）**：单位时间内输出多少 token，沿用官方统计与排空口径。
- **请求吞吐（requests/s）**：单位时间内完成多少请求；请求长短不同，不能单独比较。
- **TTFT 与输出交互速度**：等多久开始回答、开始后输出是否流畅；吞吐提高不一定意味着体验更好。
- **错误与有效性**：检查失败、上下文溢出和 `submission_valid`，无效或未完成结果不能当成功成绩。

固定种子不保证不同 engine 在相同时长内执行完全相同的请求：较快的 engine
会推进得更远。因此要把吞吐、延迟和实际请求构成一起看。

**这不是精度评测。** 合成内容不能判断模型是否答对、代码是否正确、agent 是否完成任务。
smoke 即使通过工具校验，也仍是 15 分钟开发结果，不等同于官方一小时成绩或官方认证。

## 官方来源

- [AgentX 256k 数据集](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k)
- [AgentX 方法说明](https://inferencex.semianalysis.com/agentx/methodology)
- [AgentX Harness](https://github.com/SemiAnalysisAI/agentx-harness)
