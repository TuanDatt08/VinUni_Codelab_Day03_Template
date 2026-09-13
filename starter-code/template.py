"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import os
import re
from typing import Dict, List, Any, Tuple

from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast


def describe_tools() -> str:
    """Convert TOOL_DEFINITIONS to a formatted string for the prompt."""
    descriptions = []
    for tool in TOOL_DEFINITIONS:
        name = tool["name"]
        desc = tool["description"]
        params = tool.get("parameters", {})
        param_str = ", ".join(f"{k}: {v}" for k, v in params.items())
        descriptions.append(f"- {name}({param_str}): {desc}")
    return "\n".join(descriptions)


SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO: Trả về câu trả lời tĩnh hoặc gọi LLM 1 lượt (không dùng tool)
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn du lịch. Hãy trả lời "
                    f"KHÔNG dùng tool hay internet: {user_input}"
                )
                return {
                    "response": response.text,
                    "model": "gemini-1.5-flash"
                }
            except Exception as e:
                return {
                    "response": f"[Chatbot Baseline] Lỗi: {e}",
                    "model": "gemini-1.5-flash"
                }

        return {
            "response": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "model": "static"
        }

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    def run(self, user_input: str) -> str:
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []
        history: List[str] = []   # scratchpad đưa vào prompt vòng sau
        called: set = set()       # safeguard: chống gọi trùng tool
        iteration = 0

        # TODO 2: Thiết lập vòng lặp while iteration < self.max_iterations
        while iteration < self.max_iterations:
            iteration += 1

            # TODO 3: Phân tích Thought / Action từ Agent
            decision = self._think(user_input, history, iteration)
            thought = decision.get("thought", "")
            action = decision.get("action", "final_answer")
            args = decision.get("args", {}) or {}

            if action == "final_answer":
                answer = decision.get("answer", "Không có câu trả lời.")
                self.trace.append({
                    "step": iteration,
                    "thought": thought,
                    "action": "final_answer",
                    "observation": None,
                })
                return answer

            # Safeguard: cùng tool + cùng args -> dừng, tránh lặp vô ích
            signature = f"{action}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
            if signature in called:
                self.trace.append({
                    "step": iteration,
                    "thought": "Phát hiện gọi trùng tool -> kết thúc sớm",
                    "action": "final_answer",
                    "observation": None,
                })
                return self._summarize(history)
            called.add(signature)

            # TODO 4: Thực thi Tool trong TOOL_MAP nếu có Action
            observation = self._call_tool(action, args)

            # TODO 5: Ghi lại Observation và lặp lại cho tới khi ra Final Answer
            history.append(f"Action: {action}({args})\nObservation: {observation}")
            self.trace.append({
                "step": iteration,
                "thought": thought,
                "action": action,
                "args": args,
                "observation": observation,
            })

        # Safeguard: hết vòng lặp mà chưa có Final Answer
        return (
            f"Đã đạt giới hạn {self.max_iterations} vòng lặp mà chưa hoàn thành.\n"
            + self._summarize(history)
        )

    # ------------------------------------------------------------
    # Action: gọi tool an toàn, mọi lỗi đều thành Observation
    # ------------------------------------------------------------

    def _call_tool(self, action: str, args: Dict[str, Any]) -> str:
        if action not in TOOL_MAP:
            return f"LỖI: tool '{action}' không tồn tại trong TOOL_MAP"

        try:
            result = TOOL_MAP[action](**args)
        except TypeError as e:
            return f"LỖI: sai tham số cho {action}: {e}"
        except KeyError as e:
            return f"LỖI: dữ liệu thiếu trường {e}"
        except Exception as e:
            return f"LỖI: {action} thất bại: {e}"

        # Chuẩn hoá output vì mỗi tool trả một kiểu khác nhau
        if isinstance(result, list):
            if not result:
                return "LỖI: không tìm thấy kết quả nào khớp điều kiện"
            return json.dumps(result, ensure_ascii=False)

        if isinstance(result, dict):
            if "error" in result:
                return f"LỖI: {result['error']}"
            return json.dumps(result, ensure_ascii=False)

        return str(result)

    # ------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------

    def _think(self, user_input: str, history: List[str], iteration: int) -> Dict[str, Any]:
        """Gọi LLM sinh quyết định JSON. Không có API key -> dùng luật."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return self._fallback(user_input, history, iteration)

        scratchpad = "\n".join(history) if history else "(chưa có)"
        prompt = f"""{SYSTEM_PROMPT.format(tools=describe_tools())}

Mục tiêu người dùng: {user_input}

Lịch sử đã thực hiện:
{scratchpad}

Đây là vòng {iteration}/{self.max_iterations}. Nếu sắp hết lượt hãy chốt câu trả lời.
CHỈ trả về JSON thuần, không markdown:
{{"thought": "...", "action": "ten_tool", "args": {{...}}}}
hoặc
{{"thought": "...", "action": "final_answer", "answer": "..."}}"""

        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            raw = model.generate_content(prompt).text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            decision = json.loads(raw)
            if isinstance(decision, dict):
                return decision
        except Exception:
            pass  # LLM lỗi hoặc JSON sai -> không để agent chết

        return self._fallback(user_input, history, iteration)

    def _fallback(self, user_input: str, history: List[str], iteration: int) -> Dict[str, Any]:
        """Luật rule-based để chạy demo khi chưa có API key."""
        low = user_input.lower()
        origin, destination = self._parse_route(user_input)
        done = {h.split("(")[0].replace("Action: ", "") for h in history}

        if "get_flight_info" not in done and any(
            k in low for k in ("bay", "vé", "chuyến")
        ):
            return {
                "thought": f"Cần tìm chuyến bay {origin} -> {destination}",
                "action": "get_flight_info",
                "args": {
                    "origin": origin,
                    "destination": destination,
                    "max_price": self._parse_budget(user_input),
                },
            }

        if "get_weather_forecast" not in done and any(
            k in low for k in ("thời tiết", "mưa", "nắng", "mặc gì", "trang phục")
        ):
            return {
                "thought": f"Cần tra thời tiết {destination}",
                "action": "get_weather_forecast",
                "args": {"city_code": destination},
            }

        return {
            "thought": "Đã đủ thông tin để trả lời",
            "action": "final_answer",
            "answer": self._summarize(history),
        }

    # ------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------

    def _parse_route(self, text: str) -> Tuple[str, str]:
        """Tách (origin, destination) theo thứ tự xuất hiện trong câu."""
        upper = text.upper()
        aliases = {
            "SGN": ["SGN", "HỒ CHÍ MINH", "SÀI GÒN", "TP.HCM"],
            "HAN": ["HAN", "HÀ NỘI"],
            "DAD": ["DAD", "ĐÀ NẴNG"],
        }
        found = []
        for code, names in aliases.items():
            positions = [upper.find(n) for n in names if n in upper]
            if positions:
                found.append((min(positions), code))
        found.sort()
        codes = [c for _, c in found]

        if len(codes) >= 2:
            return codes[0], codes[1]
        if len(codes) == 1:
            return ("SGN", codes[0]) if codes[0] != "SGN" else ("HAN", "SGN")
        return "HAN", "SGN"

    def _parse_budget(self, text: str, default: int = 5_000_000) -> int:
        """Đọc ngân sách: '2 triệu' -> 2000000, '1.5 triệu' -> 1500000."""
        low = text.lower().replace(",", ".")
        m = re.search(r"(\d+(?:\.\d+)?)\s*(triệu|tr\b|củ)", low)
        if m:
            return int(float(m.group(1)) * 1_000_000)
        m = re.search(r"(\d{6,})", low.replace(".", ""))
        if m:
            return int(m.group(1))
        return default

    def _summarize(self, history: List[str]) -> str:
        if not history:
            return "Chưa thu thập được thông tin nào."
        return "Thông tin thu thập được:\n" + "\n".join(history)

    def print_trace(self) -> None:
        print("=" * 60)
        for s in self.trace:
            print(f"[Vòng {s['step']}]")
            print(f"  Thought    : {s.get('thought', '')}")
            print(f"  Action     : {s.get('action', '')}")
            if s.get("args"):
                print(f"  Args       : {s['args']}")
            if s.get("observation"):
                print(f"  Observation: {s['observation'][:150]}")
            print("-" * 60)

def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result)
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()