from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


from config import LabConfig
from model_provider import ProviderConfig

def make_config(tmp_path: Path):
    """Build an isolated config for tests."""
    base_dir = Path(__file__).resolve().parent.parent
    (tmp_path / "profiles").mkdir(parents=True, exist_ok=True)
    
    # Low threshold and keep messages to trigger compaction quickly in tests
    return LabConfig(
        base_dir=base_dir,
        data_dir=base_dir / "data",
        state_dir=tmp_path,
        compact_threshold_tokens=40,
        compact_keep_messages=2,
        model=ProviderConfig("openai", "gpt-4o-mini", 0.0),
        judge_model=ProviderConfig("openai", "gpt-4o-mini", 0.0)
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    from memory_store import UserProfileStore
    store = UserProfileStore(tmp_path / "profiles")
    user_id = "test_user"
    
    # Test Write
    path = store.write_text(user_id, "# Profile\n- Tên: Nguyễn Văn A")
    assert path.exists()
    
    # Test Read
    content = store.read_text(user_id)
    assert "- Tên: Nguyễn Văn A" in content
    
    # Test Edit
    changed = store.edit_text(user_id, "Nguyễn Văn A", "Nguyễn Văn B")
    assert changed is True
    
    content_after = store.read_text(user_id)
    assert "Nguyễn Văn B" in content_after
    assert "Nguyễn Văn A" not in content_after
    
    # Test Size
    assert store.file_size(user_id) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    
    # Send enough turns to exceed threshold (40 tokens)
    agent.reply("user1", "thread1", "Chào bạn, mình đang tìm hiểu về hệ thống bộ nhớ cho AI agent.")
    agent.reply("user1", "thread1", "Mình cần xây dựng một agent có khả năng lưu trữ facts dài hạn.")
    agent.reply("user1", "thread1", "Đồng thời nén lịch sử cuộc trò chuyện khi context quá dài.")
    
    assert agent.compaction_count("thread1") > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions/threads and baseline does not."""
    config = make_config(tmp_path)
    
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    
    # Thread 1: Learn fact
    baseline.reply("user1", "thread1", "Chào bạn, mình tên là DũngCT.")
    advanced.reply("user1", "thread1", "Chào bạn, mình tên là DũngCT.")
    
    # Thread 2: Query fact
    baseline_res = baseline.reply("user1", "thread2", "Nhắc lại giúp mình xem mình tên là gì?")
    advanced_res = advanced.reply("user1", "thread2", "Nhắc lại giúp mình xem mình tên là gì?")
    
    # Baseline must NOT remember
    assert "DũngCT" not in baseline_res["content"]
    # Advanced MUST remember
    assert "DũngCT" in advanced_res["content"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    config = make_config(tmp_path)
    
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    
    # A genuinely long conversation with longer messages to show the advantage of compaction
    messages = [
        "Chào bạn, mình tên là DũngCT, hiện đang sinh sống và làm việc tại thành phố Huế mộng mơ.",
        "Đồ uống yêu thích nhất của mình vào mỗi buổi sáng là một ly cà phê sữa đá đậm đà.",
        "Nghề nghiệp chính của mình là MLOps engineer, chịu trách nhiệm vận hành các mô hình AI.",
        "Mình rất thích chạy bộ quanh sông Hương vào lúc 6 giờ sáng để bắt đầu ngày mới tỉnh táo.",
        "Style trả lời mình thích là ngắn gọn, có cấu trúc bullet point rõ ràng và lấy ví dụ thực tế.",
        "Hôm nay mình đang viết mã nguồn thử nghiệm cho hệ thống compact memory nhằm nén các cuộc hội thoại quá dài.",
        "Mình thấy việc lưu trữ thông tin lâu dài giúp ích rất nhiều cho việc chăm sóc khách hàng tự động.",
        "Tuy nhiên chi phí token cũng là một bài toán đau đầu cần giải quyết khi số lượng lượt chat tăng lên.",
        "Chúng ta nên có sự so sánh chi tiết và trực quan giữa recall và chi phí token để tối ưu hệ thống.",
        "Mình cũng nuôi một chú chó corgi rất đáng yêu tên là Bơ, nó hay phá phách mỗi khi mình làm việc.",
        "Bạn hãy tóm tắt lại các sở thích và phong cách phản hồi mà mình đã chia sẻ từ đầu đến giờ nhé."
    ]
    
    baseline_prompt_tokens = 0
    advanced_prompt_tokens = 0
    
    for msg in messages:
        b_res = baseline.reply("user1", "thread1", msg)
        a_res = advanced.reply("user1", "thread1", msg)
        
        baseline_prompt_tokens += b_res["prompt_tokens"]
        advanced_prompt_tokens += a_res["prompt_tokens"]
        
    assert advanced.compaction_count("thread1") > 0
    assert advanced_prompt_tokens < baseline_prompt_tokens
