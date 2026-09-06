"""BLOCKED is terminal even when the platform already delivered the stream."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

from gateway.run_goals import GatewayGoalsMixin


def test_gateway_durable_fence_prevents_goal_judge_and_loop_resend():
    class Runner(GatewayGoalsMixin):
        pass
    runner = Runner()
    runner.async_session_store = NS(get_or_create_session=AsyncMock(return_value=NS(session_id="blocked")))
    runner.session_store = NS(_db=NS(get_inference_terminal_marker=Mock(return_value={"status": "BLOCKED"})))
    runner._post_turn_goal_continuation = AsyncMock()
    runner._post_turn_loop_completion = AsyncMock()
    for result in ({"blocked": True}, None):
        asyncio.run(runner._run_post_turn_hooks(agent_result=result, source=None, is_internal=False,
                    event=NS(_streamed_final_response="AGENT_STALLED")))
    runner._post_turn_goal_continuation.assert_not_called()
    runner._post_turn_loop_completion.assert_not_called()


def test_tui_terminal_result_never_drains_followups():
    from tui_gateway import prompt_turn
    from tui_gateway.method_ctx import rebind
    drain = Mock(side_effect=AssertionError("terminal result drained a queued continuation"))
    followups = rebind(prompt_turn._run_post_turn_followups, {"_drain_queued_prompt": drain})
    followups("rid", "sid", {}, {"blocked": True, "pending_steer": "Continue"}, "goal")
    drain.assert_not_called()
