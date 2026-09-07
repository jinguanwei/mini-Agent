#!/usr/bin/env python3
"""
test_mini_agent.py —— 不依赖真实 API 的验证
用 unittest.mock 模拟 LLM 响应，验证：
  1. 三个工具函数真实可执行（写/读文件、bash、错误处理）
  2. Agent 循环：直接回答 / 工具调用后回答 / 未知工具 / 最大迭代
运行：python3 -m unittest test_mini_agent -v
"""

import json
import unittest
from unittest import mock

import mini_agent


# ---------- 构造假的 OpenAI 响应对象 ----------

class FakeMessage:
    def __init__(self, content, tool_calls=None, role="assistant"):
        self.content = content
        self.tool_calls = tool_calls
        self.role = role


class FakeChoice:
    def __init__(self, message):
        self.message = message


class FakeResponse:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]


def fake_tool_call(call_id, name, args_str):
    """构造一个假的 tool_call：id + 函数名 + 参数 JSON 字符串。"""
    fn = mock.Mock()
    fn.name = name
    fn.arguments = args_str
    tc = mock.Mock()
    tc.id = call_id
    tc.function = fn
    return tc


def find_tool_messages(msgs):
    """在消息列表中找出 role=tool 的回填消息（dict 类型）。"""
    return [m for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]


def make_client(*responses):
    """返回一个按顺序吐出响应的 mock 客户端。"""
    client = mock.Mock()
    client.chat.completions.create.side_effect = list(responses)
    return client


# ---------- 1. 工具函数测试 ----------

class TestTools(unittest.TestCase):
    def test_write_then_read(self):
        path = "/tmp/mini_agent_test.txt"
        r = mini_agent.write_file(path, "hello agent")
        self.assertIn("已写入", r)
        self.assertEqual(mini_agent.read_file(path), "hello agent")

    def test_bash_output(self):
        r = mini_agent.execute_bash("echo hello-bash")
        self.assertIn("hello-bash", r)

    def test_bash_timeout_graceful(self):
        """命令超时被捕获，返回错误文本而不是崩溃。"""
        mini_agent.BASH_TIMEOUT = 1  # 缩短超时加速测试
        try:
            r = mini_agent.execute_bash("sleep 3")
            self.assertIn("Error", r)
        finally:
            mini_agent.BASH_TIMEOUT = 30

    def test_read_missing_file(self):
        r = mini_agent.read_file("/no/such/file/xyz")
        self.assertIn("Error", r)


# ---------- 2. Agent 循环测试 ----------

class TestAgentLoop(unittest.TestCase):
    def test_direct_answer(self):
        """模型第一轮就直接回答 → 立即返回，不调工具。"""
        client = make_client(FakeResponse(FakeMessage("你好")))
        result = mini_agent.run_agent("hi", client=client)
        self.assertEqual(result, "你好")

    def test_tool_then_answer(self):
        """模型先要求写文件，拿到工具结果后再回答。"""
        write_call = fake_tool_call(
            "call_1", "write_file",
            json.dumps({"path": "/tmp/mock_loop.txt", "content": "abc"}),
        )
        client = make_client(
            FakeResponse(FakeMessage(None, [write_call])),
            FakeResponse(FakeMessage("文件已创建")),
        )
        result = mini_agent.run_agent("创建一个文件", client=client)
        self.assertEqual(result, "文件已创建")
        # 工具真的被执行了（副作用落盘）
        with open("/tmp/mock_loop.txt") as f:
            self.assertEqual(f.read(), "abc")
        # 发给模型的第二轮消息里，包含 role=tool 的结果回填（与 call_1 对应）
        second_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        tool_msgs = find_tool_messages(second_messages)
        self.assertTrue(tool_msgs)
        self.assertEqual(tool_msgs[-1]["tool_call_id"], "call_1")

    def test_unknown_tool_returns_error_to_model(self):
        """模型要求调用一个不存在的工具 → 错误回填给模型，循环继续。"""
        bad_call = fake_tool_call("call_x", "no_such_tool", "{}")
        client = make_client(
            FakeResponse(FakeMessage(None, [bad_call])),
            FakeResponse(FakeMessage("我不认识这个工具，已放弃")),
        )
        result = mini_agent.run_agent("随便做点啥", client=client)
        self.assertEqual(result, "我不认识这个工具，已放弃")
        msgs = client.chat.completions.create.call_args_list[0].kwargs["messages"]
        self.assertTrue(any("未知工具" in m["content"] for m in find_tool_messages(msgs)))

    def test_bad_json_args_returns_error_to_model(self):
        """模型返回非法 JSON 参数 → 不崩溃，错误回填。"""
        bad_call = fake_tool_call("call_y", "read_file", "{not json")
        client = make_client(
            FakeResponse(FakeMessage(None, [bad_call])),
            FakeResponse(FakeMessage("参数格式错了，我重新来")),
        )
        result = mini_agent.run_agent("读文件", client=client)
        self.assertEqual(result, "参数格式错了，我重新来")
        msgs = client.chat.completions.create.call_args_list[0].kwargs["messages"]
        self.assertTrue(any("合法 JSON" in m["content"] for m in find_tool_messages(msgs)))

    def test_max_iterations(self):
        """模型每轮都要求调工具 → 达到上限后返回提示，不死循环。"""
        loop_call = fake_tool_call("c", "execute_bash", json.dumps({"command": "echo x"}))
        client = make_client(*[FakeResponse(FakeMessage(None, [loop_call]))] * 3)
        result = mini_agent.run_agent("循环", client=client, max_iterations=3)
        self.assertIn("最大迭代", result)
        self.assertEqual(client.chat.completions.create.call_count, 3)

    def test_api_error_graceful(self):
        """模型接口报错 → 返回错误信息而不是抛异常。"""
        client = mock.Mock()
        client.chat.completions.create.side_effect = Exception("network down")
        result = mini_agent.run_agent("hi", client=client)
        self.assertIn("调用模型失败", result)


if __name__ == "__main__":
    unittest.main()
