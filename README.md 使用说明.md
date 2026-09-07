# mini_agent —— 仿 nanoAgent 的极简 Agent（无框架版）

不依赖任何 Agent 框架（LangChain / AutoGen / LlamaIndex 等），仅用 `openai` SDK 的
**Function Calling** 实现一个能操作本机系统的 Agent。核心循环与
[nanoAgent](https://github.com/RealSerendipity/nanoAgent) 的 `agent.py` 同构，约 140 行。

## 快速开始

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-xxx                 # 必填
export OPENAI_BASE_URL=https://api.deepseek.com   # 可选，默认已是 DeepSeek
export OPENAI_MODEL=deepseek-chat            # 可选，默认 deepseek-chat

# 单次任务
python3 mini_agent.py "列出当前目录所有 py 文件，并统计总行数"
python3 mini_agent.py "把 hello world 写进 hello.py 然后运行它"

# 交互对话（保留上下文）
python3 mini_agent.py --chat
```

> 想用 OpenAI 或其他 OpenAI 兼容端点，改环境变量即可：
> `export OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_MODEL=gpt-4o-mini`

## 三个组成部分

| 部分 | 代码位置 | 作用 |
| --- | --- | --- |
| 工具声明 `TOOLS` | JSON Schema 列表 | 告诉模型"能调用什么、参数长什么样" |
| 工具实现 `TOOL_IMPLS` | Python 函数 | 真正执行 bash / 读文件 / 写文件 |
| Agent 循环 `run_agent` | while-for 循环 | 调模型 → 执行工具 → 结果回填 → 再调模型 |

## 核心循环（与 nanoAgent 一致）

```
user_message
    │
    ▼
┌───────────────────────────┐
│  调模型 (messages + tools) │◄────────────────────────┐
└───────────────────────────┘                          │
    │                                                   │
    ├─ 模型返回无 tool_calls → 返回 content（最终答案） │
    │                                                   │
    └─ 模型返回 tool_calls → 逐个执行工具               │
        │                  结果以 role=tool 回填 ───────┘
        └─ 超过 max_iterations → 返回"已达最大迭代"
```

## 运行测试（不需要真实 API key）

```bash
python3 -m unittest test_mini_agent -v
```

用 `unittest.mock` 模拟 LLM 响应，覆盖：工具真实执行、直接回答、
工具调用后回答、未知工具、非法参数 JSON、最大迭代保护、接口异常兜底。

## 与原版 nanoAgent 的差异

1. **默认 DeepSeek**：原版默认 OpenAI；本版默认 DeepSeek（国内直连），端点/模型均可覆盖。
2. **错误回填**：工具执行、参数解析、未知工具出错时，把错误文本回填给模型让它自纠，而不是崩溃。
3. **交互模式**：`--chat` 保留 messages，模型能记住上下文；原版仅单次任务。
4. **超时可配**：bash 超时提为 `BASH_TIMEOUT` 常量，防止模型让系统卡死。

## 安全提醒

`execute_bash` 以 `shell=True` 执行模型生成的命令，能力等同当前用户。
只应在可信环境中使用，不要把它暴露给不可信输入。
