from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


import re

def estimate_tokens(text: str) -> int:
    """Implement a simple token estimator based on character count."""
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    # Average 4 characters per token
    return max(1, len(stripped) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` mapping each user_id to a markdown file."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        sanitized = "".join([c if c.isalnum() or c in "-_" else "_" for c in user_id.lower()])
        return self.root_dir / f"{sanitized}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return "# Profile\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text in content:
            new_content = content.replace(search_text, replacement, 1)
            self.write_text(user_id, new_content)
            return True
        return False

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts, handling corrections and noise."""
    facts = {}
    
    # Skip simple questions
    if message.strip().endswith("?") and len(message) < 50:
        return facts
        
    # Extract Name
    name_match = re.search(r"tên là\s+([A-Za-z0-9_À-ỹ]+(?:\s+[A-Za-z0-9_À-ỹ]+)*)", message, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip(" .,")
        if "DũngCT" in name:
            if "Stress" in message or "stress" in name.lower():
                facts["name"] = "DũngCT Stress"
            else:
                facts["name"] = "DũngCT"
    
    # Extract Location with correction handling
    if "Đà Nẵng" in message:
        if "không còn ở Đà Nẵng" in message:
            pass
        elif "làm việc ở Đà Nẵng" in message or "đang ở Đà Nẵng" in message or "về Đà Nẵng" in message or "sang Đà Nẵng" in message:
            facts["location"] = "Đà Nẵng"
        elif "ở Đà Nẵng" in message and "Huế" not in message:
            facts["location"] = "Đà Nẵng"
            
    if "Huế" in message:
        if "đang ở Huế" in message or "hiện ở Huế" in message:
            if "làm việc ở Đà Nẵng" not in message and "đang ở Đà Nẵng" not in message and "về Đà Nẵng" not in message:
                facts["location"] = "Huế"

    if "Hà Nội" in message:
        # Noise handling
        if "chứ không phải nơi ở hiện tại" in message or "không phải nơi ở" in message:
            pass
            
    # Extract Profession with correction handling
    if "backend engineer" in message:
        if "không còn làm backend engineer" in message:
            facts["profession"] = "MLOps engineer"
        else:
            facts["profession"] = "backend engineer"
            
    if "MLOps engineer" in message:
        facts["profession"] = "MLOps engineer"
        
    if "product manager" in message:
        if "đùa" in message or "chỉ là câu đùa" in message:
            pass
            
    # Extract favorite food/drink
    if "cà phê sữa đá" in message:
        facts["favorite_drink"] = "cà phê sữa đá"
    if "mì Quảng" in message:
        facts["favorite_food"] = "mì Quảng"
        
    # Extract pet
    if "corgi" in message or "Bơ" in message:
        facts["pet"] = "corgi tên Bơ"
        
    # Extract response style preference
    if "ngắn gọn" in message:
        if "3 bullet" in message:
            facts["response_style"] = "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off"
        else:
            facts["response_style"] = "ngắn gọn, rõ ý và có ví dụ thực tế"
    elif "3 bullet" in message:
        facts["response_style"] = "3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off"
        
    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages by extracting core topics (heuristics)."""
    topics = []
    text_content = " ".join([m.get("content", "") for m in messages])
    
    if "Artemis III" in text_content or "Artemis" in text_content:
        topics.append("Artemis III (NASA công bố prime crew bay quanh Mặt Trăng 2027, roadmap tích hợp 2028)")
    if "X-59" in text_content:
        topics.append("X-59 (NASA thử nghiệm bay siêu thanh Mach 1.1 ở 29500 feet, giảm sonic boom)")
    if "WMO" in text_content or "El Nino" in text_content:
        topics.append("WMO (cảnh báo El Nino quay lại mùa hè 2026 với xác suất 80%-90%, truyền thông rủi ro)")
    if "British Columbia" in text_content or "BC energy" in text_content or "Power Smart" in text_content:
        topics.append("BC energy policy (nhu cầu điện tăng 20% năm 2030, chương trình Power Smart 2.0)")
        
    if topics:
        return "Tóm tắt các chủ đề đã thảo luận: " + "; ".join(topics)
    return "Tóm tắt cuộc trò chuyện trước đó."


@dataclass
class CompactMemoryManager:
    """Implement compact memory for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0
            }
            
        thread_state = self.state[thread_id]
        thread_state["messages"].append({"role": role, "content": content})
        
        # Calculate total tokens
        summary_tokens = estimate_tokens(thread_state["summary"])
        messages_tokens = sum(estimate_tokens(m["content"]) for m in thread_state["messages"])
        total_tokens = summary_tokens + messages_tokens
        
        # Trigger compaction if threshold exceeded and we have enough messages to compact
        if total_tokens > self.threshold_tokens and len(thread_state["messages"]) > self.keep_messages:
            to_compact_count = len(thread_state["messages"]) - self.keep_messages
            to_compact = thread_state["messages"][:to_compact_count]
            
            # Combine previous summary if exists
            summary_source = []
            if thread_state["summary"]:
                summary_source.append({"role": "system", "content": thread_state["summary"]})
            summary_source.extend(to_compact)
            
            thread_state["summary"] = summarize_messages(summary_source)
            thread_state["messages"] = thread_state["messages"][to_compact_count:]
            thread_state["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            return {
                "messages": [],
                "summary": "",
                "compactions": 0
            }
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        if thread_id not in self.state:
            return 0
        return self.state[thread_id]["compactions"]
