from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


import json
import shutil
from tabulate import tabulate

def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear in the answer."""
    if not expected:
        return 1.0
    matched = sum(1 for word in expected if word.lower() in answer.lower())
    total = len(expected)
    
    if matched == total:
        return 1.0
    elif matched > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Add a lightweight quality score for offline mode based on recall and reply fluency."""
    recall = recall_points(answer, expected)
    ans_lower = answer.lower()
    
    # Penalize default/unknown answers
    if "chưa biết" in ans_lower or "không biết" in ans_lower or "chào bạn! mình" in ans_lower:
        return max(0.0, recall - 0.5)
        
    if len(answer.strip()) < 15:
        return 0.1
        
    return recall


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations and collect metrics."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    
    # Check initial memory file size
    user_ids = list(set(c["user_id"] for c in conversations))
    init_sizes = {}
    for uid in user_ids:
        if hasattr(agent, "memory_file_size"):
            init_sizes[uid] = agent.memory_file_size(uid)
        else:
            init_sizes[uid] = 0
            
    recalls = []
    qualities = []
    total_compactions = 0
    
    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        turns = conv["turns"]
        
        # Feed regular turns to agent
        for turn in turns:
            res = agent.reply(user_id, conv_id, turn)
            total_agent_tokens += res.get("agent_tokens", 0)
            total_prompt_tokens += res.get("prompt_tokens", 0)
            
        total_compactions += agent.compaction_count(conv_id)
        
        # Ask recall questions in a fresh thread
        recall_questions = conv.get("recall_questions", [])
        for q_idx, q in enumerate(recall_questions):
            recall_thread = f"{conv_id}_recall_{q_idx}"
            q_res = agent.reply(user_id, recall_thread, q["question"])
            
            total_agent_tokens += q_res.get("agent_tokens", 0)
            total_prompt_tokens += q_res.get("prompt_tokens", 0)
            
            ans = q_res["content"]
            expected = q["expected_contains"]
            
            recalls.append(recall_points(ans, expected))
            qualities.append(heuristic_quality(ans, expected))
            
    # Compute final memory size growth
    memory_growth = 0
    for uid in user_ids:
        if hasattr(agent, "memory_file_size"):
            final_size = agent.memory_file_size(uid)
            memory_growth += max(0, final_size - init_sizes[uid])
            
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_quality = sum(qualities) / len(qualities) if qualities else 0.0
    
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=total_compactions
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a markdown table with final metrics."""
    headers = [
        "Agent Name",
        "Agent Tokens Only",
        "Prompt Tokens Processed",
        "Cross-Session Recall",
        "Response Quality",
        "Memory Growth (bytes)",
        "Compactions"
    ]
    data = []
    for r in rows:
        data.append([
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions
        ])
    return tabulate(data, headers=headers, tablefmt="github")


def main() -> None:
    """Run standard and long-context stress benchmarks and output results."""
    config = load_config(Path(__file__).resolve().parent.parent)
    
    # Reset/clear state directory for profiles
    profiles_dir = config.state_dir / "profiles"
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"
    
    std_convs = load_conversations(standard_path)
    stress_convs = load_conversations(stress_path)
    
    # Auto-detect if a valid API key is present
    model_config = config.model
    api_key = model_config.api_key
    is_live = False
    if api_key and api_key.strip() != "" and "your_" not in api_key.lower():
        is_live = True
        
    print(f"=== BENCHMARK EXECUTION MODE: {'LIVE (using ' + model_config.provider + '/' + model_config.model_name + ')' if is_live else 'OFFLINE'} ===")
    print()
    
    print("=== RUNNING STANDARD BENCHMARK (conversations.json) ===")
    
    # Run Baseline (naive)
    baseline_std = BaselineAgent(config, force_offline=not is_live)
    baseline_std_row = run_agent_benchmark("Baseline Agent (Std)", baseline_std, std_convs, config)
    
    # Clear state dir to run Advanced Agent cleanly
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    # Run Advanced
    advanced_std = AdvancedAgent(config, force_offline=not is_live)
    advanced_std_row = run_agent_benchmark("Advanced Agent (Std)", advanced_std, std_convs, config)
    
    print(format_rows([baseline_std_row, advanced_std_row]))
    print()
    
    print("=== RUNNING LONG-CONTEXT STRESS BENCHMARK (advanced_long_context.json) ===")
    
    # Run Baseline (naive)
    baseline_stress = BaselineAgent(config, force_offline=not is_live)
    baseline_stress_row = run_agent_benchmark("Baseline Agent (Stress)", baseline_stress, stress_convs, config)
    
    # Clear state dir to run Advanced Agent cleanly
    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    
    # Run Advanced
    advanced_stress = AdvancedAgent(config, force_offline=not is_live)
    advanced_stress_row = run_agent_benchmark("Advanced Agent (Stress)", advanced_stress, stress_convs, config)
    
    print(format_rows([baseline_stress_row, advanced_stress_row]))


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
