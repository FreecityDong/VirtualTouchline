import argparse
import json
import os
from typing import Any, Dict, List

from openai import OpenAI


# 按你的要求保留默认 Key；生产环境建议仅使用环境变量 DASHSCOPE_API_KEY
DEFAULT_API_KEY = "sk-bac18f9772b14f35ad33a8e53e5d9809"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "deepseek-v3.2"


def _sanitize_user_prompt(user_prompt: Any) -> Any:
    if not isinstance(user_prompt, dict):
        return user_prompt
    cleaned = dict(user_prompt)
    # 移除结构化 JSON 输出约束，避免模型继续按对象格式返回
    cleaned.pop("output_json_schema", None)
    req = cleaned.get("requirements", [])
    if not isinstance(req, list):
        req = [str(req)]
    req.extend(
        [
            "最终只输出一整段中文文字，不要分点，不要分段。",
            "不要输出JSON，不要输出Markdown代码块，不要出现字段名。",
        ]
    )
    cleaned["requirements"] = req
    return cleaned


def build_messages(payload: Dict[str, Any], target_chars: int) -> List[Dict[str, str]]:
    concise_rule = (
        "你是一名足球战术分析助手。"
        "请仅输出最终分析结论，不要输出思考过程。"
        "请始终使用中文。"
        "最终只输出一整段文字，不要分点，不要分段，不要标题。"
        "不要输出 JSON，不要输出 Markdown 代码块。"
        f"全文控制在约{target_chars}字（建议范围180-240字），语言简洁、专业。"
    )
    if "system_prompt" in payload and "user_prompt" in payload:
        sanitized_user_prompt = _sanitize_user_prompt(payload["user_prompt"])
        return [
            {"role": "system", "content": f"{payload['system_prompt']}\n{concise_rule}"},
            {
                "role": "user",
                "content": json.dumps(sanitized_user_prompt, ensure_ascii=False, separators=(",", ":")),
            },
        ]
    return [
        {"role": "system", "content": concise_rule},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def load_payload(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek API 测试脚本（阿里云百炼兼容接口）")
    parser.add_argument(
        "--input",
        default="deepseek_input_halftime_compact.json",
        help="输入 JSON 文件路径（支持包含 system_prompt/user_prompt 的完整结构）",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名，默认 deepseek-v3.2")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="阿里云兼容接口地址")
    parser.add_argument("--target-chars", type=int, default=200, help="目标输出字数，默认 200")
    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY", DEFAULT_API_KEY)
    payload = load_payload(args.input)
    messages = build_messages(payload, args.target_chars)

    client = OpenAI(api_key=api_key, base_url=args.base_url)
    completion = client.chat.completions.create(
        model=args.model,
        messages=messages,
        extra_body={"enable_thinking": False},
        stream=True,
        stream_options={"include_usage": True},
    )

    answer_content = ""
    print("\n" + "=" * 20 + "模型输出" + "=" * 20 + "\n")
    for chunk in completion:
        if not chunk.choices:
            print("\n" + "=" * 20 + "Token 消耗" + "=" * 20 + "\n")
            print(chunk.usage)
            continue

        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            print(delta.content, end="", flush=True)
            answer_content += delta.content

    print("\n\n" + "=" * 20 + "完成" + "=" * 20)
    print(f"回复长度: {len(answer_content)} 字符")


if __name__ == "__main__":
    main()
