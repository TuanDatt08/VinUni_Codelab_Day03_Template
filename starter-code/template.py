"""
Lab #3: Baseline Chatbot vs ReAct Agent
"""

import json
import os
import re
from typing import Any, Dict, List, Tuple

from tools import TOOL_DEFINITIONS, TOOL_MAP

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


def describe_tools() -> str:
    """Sinh mô tả tool từ TOOL_DEFINITIONS để nhét vào prompt."""
    lines = []
    for tool in TOOL_DEFINITIONS:
        params = ", ".join(f"{k}: {v}" for k, v in tool["parameters"].items())
        lines.append(f"- {tool['name']}({params}) — {tool['description']}")
    return "\n".join(lines)


# ============================================================
# 1. BASELINE CHATBOT (không tool, 1 lượt)
# ============================================================

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        if self.api_key:
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn du lịch. Hãy trả lời "
                    f"KHÔNG dùng tool hay internet: {user_input}"
                )
                return {"response": response.text, "model": "gemini-1.5-flash"}
            except Exception as e:
                return {"response": f"[Chatbot Baseline] Lỗi: {e}",
                        "model": "gemini-1.5-flash"}

        return {
            "response": (
                "[Chatbot Baseline] Tôi không có dữ liệu chuyến bay hay thời tiết "
                f"thực tế nên không trả lời chính xác được. Câu hỏi: {user_input}"
            ),
            "model": "static",
        }


# ============================================================
# 2. REACT AGENT
# ============================================================

class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    def run(self, user_input: str) -> str:
        # TODO 1: Khởi tạo mảng lưu lịch sử conversation / traces
        self.trace = []
        history: List[Dict[str, Any]] = []   # observation đã thu thập
        called: set = set()                  # safeguard: chống gọi trùng tool
        iteration = 0

        # TODO 2: Thiết lập vòng lặp while iteration < self.max_iterations
        while iteration < self.max_iterations:
            iteration += 1

            # TODO 3: Phân tích Thought / Action từ Agent
            decision = self._think(user_input, history, iteration)
            thought = decision.get("thought", "")

            # Bước cuối: agent chốt Final Answer
            if "final_answer" in decision or decision.get("action") is None:
                answer = decision.get("final_answer", "Không có câu trả lời.")
                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "final_answer": answer,
                })
                return answer

            action = decision["action"]
            name = action.get("name", "")
            args = action.get("args", {}) or {}

            # Safeguard: cùng tool + cùng args -> chốt luôn, tránh lặp vô ích
            signature = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
            if signature in called:
                answer = self._compose_final_answer(history)
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Phát hiện gọi trùng tool, đã đủ dữ liệu để trả lời.",
                    "final_answer": answer,
                })
                return answer
            called.add(signature)

            # TODO 4: Thực thi Tool trong TOOL_MAP nếu có Action
            observation = self._call_tool(name, args)

            # TODO 5: Ghi lại Observation và lặp lại cho tới khi ra Final Answer
            history.append({"name": name, "args": args, "observation": observation})
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": {"name": name, "args": args},
                "observation": observation,
            })

        # Safeguard: hết vòng lặp mà chưa có Final Answer
        answer = self._compose_final_answer(history)
        self.trace.append({
            "iteration": iteration,
            "thought": f"Đã đạt giới hạn {self.max_iterations} vòng lặp.",
            "final_answer": answer,
        })
        return answer

    # ------------------------------------------------------------
    # Action: gọi tool an toàn, mọi lỗi đều thành Observation
    # ------------------------------------------------------------

    def _call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        if name not in TOOL_MAP:
            return {"error": f"Tool '{name}' không tồn tại trong TOOL_MAP"}
        try:
            result = TOOL_MAP[name](**args)
        except TypeError as e:
            return {"error": f"Sai tham số cho {name}: {e}"}
        except KeyError as e:
            return {"error": f"Dữ liệu thiếu trường {e}"}
        except Exception as e:
            return {"error": f"{name} thất bại: {e}"}

        if isinstance(result, list) and not result:
            return {"error": "Không tìm thấy kết quả nào khớp điều kiện"}
        return result

    # ------------------------------------------------------------
    # Reasoning
    # ------------------------------------------------------------

    def _think(self, user_input: str, history: List[Dict[str, Any]],
               iteration: int) -> Dict[str, Any]:
        """Gọi LLM sinh quyết định JSON. Không có API key -> dùng luật."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return self._fallback(user_input, history, iteration)

        scratchpad = json.dumps(history, ensure_ascii=False, indent=2) if history else "(chưa có)"
        prompt = f"""{SYSTEM_PROMPT.format(tools=describe_tools())}

Mục tiêu người dùng: {user_input}

Lịch sử đã thực hiện:
{scratchpad}

Đây là vòng {iteration}/{self.max_iterations}. Nếu sắp hết lượt hãy chốt câu trả lời.
CHỈ trả về JSON thuần, không markdown:
{{"thought": "...", "action": {{"name": "ten_tool", "args": {{...}}}}}}
hoặc
{{"thought": "...", "final_answer": "câu trả lời đầy đủ cho khách hàng"}}"""

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

    def _fallback(self, user_input: str, history: List[Dict[str, Any]],
                  iteration: int) -> Dict[str, Any]:
        """Luật rule-based để chạy demo khi chưa có API key."""
        low = user_input.lower()
        origin, destination = self._parse_route(user_input)
        done = {h["name"] for h in history}

        if "get_flight_info" not in done and any(
            k in low for k in ("bay", "vé", "chuyến")
        ):
            return {
                "thought": f"Tôi cần tìm chuyến bay từ {origin} đi {destination}.",
                "action": {
                    "name": "get_flight_info",
                    "args": {
                        "origin": origin,
                        "destination": destination,
                        "max_price": self._parse_budget(user_input),
                    },
                },
            }

        if "get_weather_forecast" not in done and any(
            k in low for k in ("thời tiết", "mưa", "nắng", "mặc gì", "trang phục")
        ):
            return {
                "thought": f"Tôi cần kiểm tra thông tin thời tiết tại {destination}.",
                "action": {
                    "name": "get_weather_forecast",
                    "args": {"city_code": destination},
                },
            }

        return {
            "thought": "Tôi đã thu thập đủ thông tin để trả lời khách hàng.",
            "final_answer": self._compose_final_answer(history),
        }

    # ------------------------------------------------------------
    # Tổng hợp Final Answer từ các Observation
    # ------------------------------------------------------------

    def _compose_final_answer(self, history: List[Dict[str, Any]]) -> str:
        if not history:
            return "Xin lỗi, tôi chưa thu thập được thông tin nào để trả lời."

        parts: List[str] = []
        idx = 1

        for step in history:
            obs = step["observation"]

            if step["name"] == "get_flight_info":
                if isinstance(obs, dict) and "error" in obs:
                    parts.append(f"{idx}. Thông tin chuyến bay:\n   - {obs['error']}")
                else:
                    lines = [f"{idx}. Thông tin chuyến bay:"]
                    for fl in obs:
                        airline = fl.get("airline", fl.get("flight_id", "N/A"))
                        code = fl.get("flight_id", "")
                        dep = fl.get("departure_time", fl.get("time", ""))
                        price = fl.get("price_vnd", 0)
                        lines.append(f"   - {airline} ({code}): {dep} - Giá: {price:,} VNĐ")
                    parts.append("\n".join(lines))
                idx += 1

            elif step["name"] == "get_weather_forecast":
                if isinstance(obs, dict) and "error" in obs:
                    parts.append(f"{idx}. Thông tin thời tiết:\n   - {obs['error']}")
                else:
                    city = obs.get("city", step["args"].get("city_code", ""))
                    temp = obs.get("temperature_c", obs.get("temp", "?"))
                    cond = obs.get("condition", "")
                    rec = obs.get("recommendation", obs.get("outfit", ""))
                    parts.append(
                        f"{idx}. Thông tin thời tiết & trang phục:\n"
                        f"   - Thời tiết tại {city}: {temp}°C ({cond}).\n"
                        f"   - Gợi ý trang phục: {rec}"
                    )
                idx += 1

        return "\n\n".join(parts)

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


# ============================================================
# 3. MAIN
# ============================================================

def main():
    user_query = (
        "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, "
        "rồi cho biết thời tiết SGN nên mặc gì?"
    )

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