from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
            
        session = self.sessions[thread_id]
        
        if not self.force_offline and self.config.model.api_key:
            self._maybe_build_langchain_agent()
            if self.langchain_agent:
                session.messages.append({"role": "user", "content": message})
                
                # Build chat messages context (within thread only)
                from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
                lc_messages = [SystemMessage(content="Bạn là Baseline Agent. Hãy trả lời ngắn gọn, bám sát lịch sử chat.")]
                for m in session.messages:
                    if m["role"] == "user":
                        lc_messages.append(HumanMessage(content=m["content"]))
                    else:
                        lc_messages.append(AIMessage(content=m["content"]))
                
                prompt_text = " ".join([m["content"] for m in session.messages[:-1]])
                session.prompt_tokens_processed += estimate_tokens(prompt_text)
                
                response = self.langchain_agent.invoke(lc_messages)
                reply_content = response.content
                
                session.messages.append({"role": "assistant", "content": reply_content})
                gen_tokens = estimate_tokens(reply_content)
                session.token_usage += gen_tokens
                
                return {
                    "content": reply_content,
                    "agent_tokens": gen_tokens,
                    "prompt_tokens": estimate_tokens(prompt_text)
                }
                
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions[thread_id]
        session.messages.append({"role": "user", "content": message})
        
        # Estimate prompt context tokens (all raw history except last message)
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages[:-1])
        session.prompt_tokens_processed += prompt_tokens
        
        # Scrape facts from history in the same thread only
        from memory_store import extract_profile_updates
        current_facts = {}
        for m in session.messages:
            if m["role"] == "user":
                extracted = extract_profile_updates(m["content"])
                current_facts.update(extracted)
                
        # Generate deterministic offline reply based on current facts in this thread
        ans_parts = []
        msg_lower = message.lower()
        if "tên" in msg_lower:
            if "name" in current_facts:
                ans_parts.append(f"Tên bạn là {current_facts['name']}.")
            else:
                ans_parts.append("Mình chưa biết tên bạn.")
        if "đồ uống" in msg_lower or "thức uống" in msg_lower:
            if "favorite_drink" in current_facts:
                ans_parts.append(f"Đồ uống yêu thích của bạn là {current_facts['favorite_drink']}.")
            else:
                ans_parts.append("Mình chưa biết đồ uống yêu thích của bạn.")
        if "nghề" in msg_lower or "công việc" in msg_lower:
            if "profession" in current_facts:
                ans_parts.append(f"Nghề nghiệp của bạn là {current_facts['profession']}.")
            else:
                ans_parts.append("Mình chưa biết nghề nghiệp của bạn.")
        if "ở đâu" in msg_lower or "nơi ở" in msg_lower:
            if "location" in current_facts:
                ans_parts.append(f"Hiện tại bạn đang ở {current_facts['location']}.")
            else:
                ans_parts.append("Mình chưa biết nơi ở của bạn.")
        if "style" in msg_lower or "trả lời" in msg_lower:
            if "response_style" in current_facts:
                ans_parts.append(f"Style trả lời của bạn là {current_facts['response_style']}.")
            else:
                ans_parts.append("Mình chưa biết style trả lời bạn thích.")
        if "món ăn" in msg_lower or "ăn gì" in msg_lower:
            if "favorite_food" in current_facts:
                ans_parts.append(f"Món ăn yêu thích của bạn là {current_facts['favorite_food']}.")
            else:
                ans_parts.append("Mình chưa biết món ăn yêu thích của bạn.")
        if "nuôi" in msg_lower or "con gì" in msg_lower:
            if "pet" in current_facts:
                ans_parts.append(f"Bạn nuôi {current_facts['pet']}.")
            else:
                ans_parts.append("Mình chưa biết bạn nuôi con gì.")
                
        if not ans_parts:
            reply_content = "Chào bạn! Mình có thể giúp gì cho bạn?"
        else:
            reply_content = " ".join(ans_parts)
            
        session.messages.append({"role": "assistant", "content": reply_content})
        gen_tokens = estimate_tokens(reply_content)
        session.token_usage += gen_tokens
        
        return {
            "content": reply_content,
            "agent_tokens": gen_tokens,
            "prompt_tokens": prompt_tokens
        }

    def _maybe_build_langchain_agent(self):
        if self.langchain_agent is None:
            self.langchain_agent = build_chat_model(self.config.model)
