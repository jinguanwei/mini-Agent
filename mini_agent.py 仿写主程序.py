#!/usr/bin/env python3
"""
mini_agent.py —— 仿 nanoAgent 的极简 Agent（无框架版）

nanoAgent-main 主程序（agent.py，约 100 行）的核心只有三件事：
  1. 声明工具：用 JSON Schema 告诉模型"你能做什么、参数长什么样"
  2. 实现工具：用普通 Python 函数真正去执行（bash / 读文件 / 写文件）
  3. Agent 循环：调用模型 → 若模型要求调工具就执行并把结果回填给模型
     → 重复，直到模型直接给出答案

本文件保留同样的思路，只做了三处小改进：
  - 默认接入 DeepSeek 的 OpenAI 兼容接口（环境变量可切换到任意端点）
  - 工具执行失败时把错误文本回填给模型，让它自己修正，而不是崩溃
  - 支持"单次任务"（CLI）和"--chat 交互对话"两种模式

用法：
  export OPENAI_API_KEY=sk-xxx
  python3 mini_agent.py "列出当前目录所有 py 文件"
  python3 mini_agent.py --chat
"""

import json
import os
import subprocess
import sys

BASH_TIMEOUT = 30  # bash 命令超时（秒），防止模型让系统卡死

# ============================================================
# 一、LLM 客户端
# ============================================================

def get_client():
    """创建 OpenAI 兼容客户端。默认 DeepSeek，可被环境变量覆盖。"""
    from openai import OpenAI  # 懒加载：不装 openai 也能 import 本文件

    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
    )


DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "deepseek-chat")

# ============================================================
# 二、工具：声明（JSON Schema） + 实现（Python 函数） + 注册表
# ============================================================

# 2.1 工具声明：这段 JSON 会原样发给模型，模型据此决定调用谁、传什么参
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "在系统上执行一条 bash 命令，返回标准输出和错误输出",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的 bash 命令"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定文件的文本内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "把内容写入指定文件（覆盖已有内容）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
]

# 2.2 工具实现：每个函数返回一段文本，作为"工具观察结果"回填给模型
def execute_bash(command):
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=BASH_TIMEOUT
        )
        return (result.stdout or "") + (result.stderr or "")
    except Exception as e:
        return f"Error: {e}"


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


def write_file(path, content):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {path}"
    except Exception as e:
        return f"Error: {e}"


# 2.3 工具注册表：模型返回的工具名 → 真正的函数
TOOL_IMPLS = {
    "execute_bash": execute_bash,
    "read_file": read_file,
    "write_file": write_file,
}

# ============================================================
# 三、Agent 核心循环（与 nanoAgent 同构）
# ============================================================

def run_agent(user_message, client=None, max_iterations=5, messages=None):
    """
    核心循环：
    1. 把消息（含工具声明）发给模型
    2. 模型返回两种可能：
       - 没有 tool_calls  → 这就是最终答案，直接返回
       - 有 tool_calls    → 逐个执行工具，把结果以 role=tool 回填，回到第 1 步
    3. 超过 max_iterations 仍未结束 → 返回提示，防止死循环

    client: 可注入 mock，便于测试；默认用全局客户端。
    messages: 可传入历史消息（交互模式用）；None 时新建会话。
    """
    client = client or get_client()
    if messages is None:
        messages = [
            {"role": "system", "content": "你是一个能操作本机系统的助手，回答要简洁。"},
            {"role": "user", "content": user_message},
        ]
    else:
        messages.append({"role": "user", "content": user_message})

    for step in range(1, max_iterations + 1):
        print(f"\n── 第 {step} 轮 ──")
        try:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL, messages=messages, tools=TOOLS
            )
        except Exception as e:
            return f"调用模型失败: {e}"

        message = response.choices[0].message
        messages.append(message)  # 模型完整回复（含工具调用意图）进入上下文

        # 模型没有要求调工具 → 任务完成
        if not message.tool_calls:
            return message.content

        # 模型要求调工具 → 逐个执行，结果回填
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                result = f"Error: 参数不是合法 JSON: {e}"
            else:
                func = TOOL_IMPLS.get(name)
                if func is None:
                    result = f"Error: 未知工具 {name}"
                else:
                    print(f"  [工具] {name}{args}")
                    result = func(**args)
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": result}
            )

    return "已达到最大迭代次数，任务未完成。"

# ============================================================
# 四、入口：CLI 单次任务 / --chat 交互对话
# ============================================================

def chat():
    """交互对话：跨轮保留 messages，模型能记住上下文。"""
    client = get_client()
    messages = [{"role": "system", "content": "你是一个能操作本机系统的助手，回答要简洁。"}]
    print("mini_agent 交互模式（输入 exit 退出）")
    while True:
        try:
            user_input = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input in ("exit", "quit"):
            break
        if not user_input:
            continue
        result = run_agent(user_input, client=client, messages=messages)
        print(f"\n助手> {result}")


def main():
    args = sys.argv[1:]
    if not args:
        print("用法:")
        print("  python3 mini_agent.py '任务描述'   # 单次任务")
        print("  python3 mini_agent.py --chat       # 交互对话")
        sys.exit(1)
    if args[0] == "--chat":
        chat()
        return
    task = " ".join(args)
    result = run_agent(task)
    print(f"\n{result}")


if __name__ == "__main__":
    main()
