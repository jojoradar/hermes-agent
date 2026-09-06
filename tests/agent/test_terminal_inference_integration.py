"""Real AIAgent turn/stream/recovery integration with deterministic provider frames."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from tests.agent.test_terminal_inference_gate import Clock
from tests.run_agent.test_streaming import _make_stream_chunk as chunk, _make_tool_call_delta as tool


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    from run_agent import AIAgent
    from agent.inference_gate import TurnInferenceGate
    import hermes_cli.config as config
    import agent.inference_gate_stream as stream_gate
    clock = Clock()
    original = TurnInferenceGate.__init__
    monkeypatch.setattr(TurnInferenceGate, "__init__", lambda self, agent, limits, **kw:
                        original(self, agent, limits, clock=clock))
    monkeypatch.setattr(config, "load_config_readonly", lambda: {
        "agent": {"terminal_inference_gate": {"enabled": True}},
        "model": {"context_length": 71936}})
    monkeypatch.setattr(AIAgent, "_ensure_lmstudio_runtime_loaded", lambda *a: 71936)
    monkeypatch.setattr(stream_gate, "local_lane_idle", lambda *a: True)
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = AIAgent(api_key="test-key", provider="lmstudio", base_url="https://example.com/v1",
                    model="test/model", quiet_mode=True, skip_context_files=True, skip_memory=True,
                    skip_background_review=True, enabled_toolsets=[], max_iterations=3, session_db=db)
    agent._try_refresh_env_client_credentials = Mock()
    agent._try_activate_fallback = Mock(side_effect=AssertionError("fallback must not run"))
    agent._recover_primary_openai_client = Mock(side_effect=AssertionError("recovery must not run"))
    deltas = []
    agent.stream_delta_callback = deltas.append
    client = Mock()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda *a, **kw: None)
    abort = Mock()
    monkeypatch.setattr(agent, "_abort_request_openai_client", abort)
    yield agent, client, clock, deltas, abort
    db.close()


class Stream:
    response = NS(headers={})
    def __init__(self, frames):
        self.frames = frames
        self.closed = False
    def __iter__(self):
        return iter(self.frames())
    def close(self):
        self.closed = True


@pytest.mark.parametrize("mode", ["heartbeat", "reasoning", "partial_tool", "length", "empty", "disconnect"])
def test_full_turn_terminal_never_retries_or_executes_partial(runtime, mode, monkeypatch):
    import httpx
    agent, client, clock, deltas, abort = runtime
    def frames():
        if mode == "disconnect":
            raise httpx.RemoteProtocolError("controlled disconnect")
        if mode in {"length", "empty"}:
            yield chunk(content="partial" if mode == "length" else None)
            yield chunk(finish_reason="length" if mode == "length" else "stop")
            return
        deadline = 120 if mode == "heartbeat" else 300
        for second in range(deadline + 1):
            clock.now = second
            if mode == "reasoning":
                yield chunk(reasoning_content="thinking")
            elif mode == "partial_tool":
                yield chunk(tool_calls=[tool(index=0, tc_id="call-1" if second == 0 else None,
                    name="terminal" if second == 0 else None, arguments='{"command":')])
            else:
                yield chunk()
        yield chunk(content="LATE MUST NEVER APPEAR")
        yield chunk(finish_reason="stop")
    wire = Stream(frames)
    client.chat.completions.create.return_value = wire
    result = agent.run_conversation("Harmless deterministic test", system_message="Answer briefly.")
    assert result["blocked"] and not result["completed"]
    assert result["retryable"] is False
    assert client.chat.completions.create.call_count == 1
    assert "LATE" not in "".join(deltas)
    agent._try_activate_fallback.assert_not_called()
    assert result["inference_gate"]["fence_persisted"]
    if mode in {"heartbeat", "reasoning", "partial_tool", "disconnect"}:
        assert result["inference_gate"]["cancel_count"] == 1
    # Same cached agent cannot restart this logical action with a fresh turn id.
    repeated = agent.run_conversation("Continue", system_message="Answer briefly.")
    assert repeated["blocked"]
    assert client.chat.completions.create.call_count == 1


@pytest.mark.parametrize("platform", ["cli", "cron", "subagent"])
def test_normal_turn_and_second_normal_turn_complete(runtime, platform):
    import threading
    agent, client, clock, _, _ = runtime
    agent.platform = platform
    owners = []
    streams = iter([
        Stream(lambda: iter([chunk(reasoning_content="brief"), chunk(content="OK"), chunk(finish_reason="stop")])),
        Stream(lambda: iter([chunk(content="OK"), chunk(finish_reason="stop")]))])
    def create(**kw):
        owners.append(threading.get_ident())
        return next(streams)
    client.chat.completions.create.side_effect = create
    for _ in range(2):
        result = agent.run_conversation("Reply exactly OK", system_message="Answer briefly.")
        assert result["completed"] and result["final_response"] == "OK"
    assert client.chat.completions.create.call_count == 2
    assert all(c.kwargs["max_tokens"] == 8192 for c in client.chat.completions.create.call_args_list)
    if platform != "cli":
        assert owners == [threading.get_ident()] * 2  # preserve the native no-nested-worker contract


def test_late_finish_frame_rejected_after_terminal_fence(runtime):
    from agent.chat_completion_helpers import _StreamingCall
    from agent.inference_gate import admit_turn
    agent, client, clock, deltas, _ = runtime
    admit_turn(agent)
    call = _StreamingCall(agent, {}, None)
    attempt = call._start_stream_attempt()
    clock.now = 120
    assert not call._accept_chat_chunk(attempt, chunk(finish_reason="stop"))
    assert not call._accept_chat_chunk(attempt, chunk(content="late"))
    call._emit_text("late")
    assert not deltas


@pytest.mark.parametrize("idle", [True, False])
def test_cancel_confirmation_requires_matching_stream_end_and_idle(runtime, monkeypatch, idle):
    import threading
    from agent.chat_completion_helpers import _StreamingCall
    from agent.inference_gate import admit_turn, InferenceBlocked
    import agent.inference_gate_stream as sg
    agent, client, clock, _, _ = runtime
    admit_turn(agent)
    call = _StreamingCall(agent, {}, None)
    call._call_done = threading.Event()
    call._call_done.set()
    call.inference_stream_ended = True
    sg.trip(call, "INFERENCE_STALLED", "transport_stall")
    # Accelerate only the acknowledgement budget; inference clocks remain deterministic.
    call.inference_clock.cancel_deadline = sg.time.monotonic() + 0.02
    monkeypatch.setattr(sg, "local_lane_idle", lambda *a: idle)
    with pytest.raises(InferenceBlocked):
        sg.finish(call)
    assert call.inference_gate.terminal["cancel_result"] == (
        "CANCELLED_CONFIRMED" if idle else "CANCEL_UNKNOWN")
    assert call.inference_clock.cancel_count == 1
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("mode", ["503", "reasoning", "heartbeat", "partial_tool"])
def test_real_openai_sdk_has_zero_physical_resends(runtime, monkeypatch, mode):
    import httpx
    from run_agent import AIAgent
    agent, _, clock, deltas, _ = runtime
    posts, request_clients = [], []
    class Bytes(httpx.SyncByteStream):
        def __iter__(self):
            deadline = 120 if mode == "heartbeat" else 300
            for second in range(deadline + 1):
                clock.now = second
                delta = {"role": "assistant"}
                if mode == "reasoning":
                    delta["reasoning_content"] = "r"
                elif mode == "partial_tool":
                    delta["tool_calls"] = [{"index": 0, "id": "call-x" if second == 0 else None,
                        "type": "function", "function": {"name": "terminal" if second == 0 else None,
                        "arguments": '{"command":'}}]
                body = {"id": "simulated", "object": "chat.completion.chunk", "created": 0,
                        "model": agent.model, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                yield ("data: " + json.dumps(body) + "\n\n").encode()
            yield b'data: [DONE]\n\n'
    def wire(request):
        posts.append(json.loads(request.content))
        if mode == "503":
            return httpx.Response(503, json={"error": {"message": "controlled unavailable"}})
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=Bytes())
    http = httpx.Client(transport=httpx.MockTransport(wire))
    agent._client_kwargs["http_client"] = http
    def create(**kwargs):
        client = AIAgent._create_request_openai_client(agent, **kwargs)
        request_clients.append(client)
        return client
    monkeypatch.setattr(agent, "_create_request_openai_client", create)
    result = agent.run_conversation("Harmless SDK boundary test", system_message="Answer briefly.")
    assert result["blocked"]
    assert len(posts) == 1
    assert posts[0]["max_tokens"] == 8192
    assert request_clients[0].max_retries == 0
    assert agent.run_conversation("Continue")["blocked"]
    assert len(posts) == 1
    http.close()


def test_complete_tool_workflow_executes_once_then_completes(runtime, monkeypatch):
    import model_tools
    agent, client, clock, _, _ = runtime
    schema = {"type": "function", "function": {"name": "terminal", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
    agent.tools, agent.valid_tool_names = [schema], {"terminal"}
    def execute(*args, **kwargs):
        clock.now += 2000  # model clocks are disarmed while the tool owns execution
        return json.dumps({"output": "CANARY", "exit_code": 0})
    executor = Mock(side_effect=execute)
    monkeypatch.setattr(model_tools, "handle_function_call", executor)
    client.chat.completions.create.side_effect = [
        Stream(lambda: iter([chunk(tool_calls=[tool(index=0, tc_id="tool-1", name="terminal",
            arguments=json.dumps({"command": "printf CANARY"}))]), chunk(finish_reason="tool_calls")])),
        Stream(lambda: iter([chunk(content="OK"), chunk(finish_reason="stop")]))]
    result = agent.run_conversation("Use terminal once, then say OK", system_message="Answer briefly.")
    assert result["completed"] and result["final_response"] == "OK"
    assert client.chat.completions.create.call_count == 2
    executor.assert_called_once()
    assert any(m.get("role") == "tool" for m in result["messages"])


def test_all_known_send_entries_reject_a_terminal_gate(runtime):
    from agent import chat_completion_helpers as ch
    from agent.inference_gate import admit_turn, InferenceBlocked
    from run_agent import AIAgent
    agent, client, _, _, _ = runtime
    admit_turn(agent)
    agent._terminal_inference_gate.fence("AGENT_STALLED", "test")
    entries = [lambda: ch.direct_api_call(agent, {}), lambda: ch.interruptible_api_call(agent, {}),
               lambda: ch.interruptible_streaming_api_call(agent, {}),
               lambda: ch.try_activate_fallback(agent), lambda: ch.handle_max_iterations(agent, [], 500),
               lambda: AIAgent._create_request_openai_client(agent, reason="reconnect")]
    for entry in entries:
        with pytest.raises(InferenceBlocked):
            entry()
    client.chat.completions.create.assert_not_called()


def test_iteration_budget_cannot_start_an_unowned_summary(runtime, monkeypatch):
    import model_tools
    agent, client, _, _, _ = runtime
    agent.max_iterations = 1
    agent.tools = [{"type": "function", "function": {"name": "terminal", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
    agent.valid_tool_names = {"terminal"}
    monkeypatch.setattr(model_tools, "handle_function_call", lambda *a, **kw:
                        json.dumps({"output": "CANARY", "exit_code": 0}))
    client.chat.completions.create.side_effect = lambda **kw: Stream(lambda: iter([
        chunk(tool_calls=[tool(index=0, tc_id="budget-tool", name="terminal",
                              arguments='{"command":"printf CANARY"}')]),
        chunk(finish_reason="tool_calls")]))
    result = agent.run_conversation("Use terminal once", system_message="Answer briefly.")
    assert result["blocked"] and result["inference_gate"]["reason"] == "iteration_budget_exhausted"
    assert all(c.kwargs.get("stream") is True for c in client.chat.completions.create.call_args_list)
    assert all(c.kwargs.get("tools") for c in client.chat.completions.create.call_args_list)


@pytest.mark.parametrize("platform", ["cli", "cron"])
def test_real_socket_monitor_cancels_only_the_request_owner_once(runtime, monkeypatch, platform):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from run_agent import AIAgent
    agent, _, clock, _, _ = runtime
    agent.platform = platform
    accepted, release = threading.Event(), threading.Event()
    posts, aborted = [], []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            posts.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.flush()
            accepted.set()
            release.wait(5)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    agent.base_url = f"http://127.0.0.1:{server.server_port}/v1"
    agent._client_kwargs["base_url"] = agent.base_url
    monkeypatch.setattr(agent, "_create_request_openai_client",
                        lambda **kw: AIAgent._create_request_openai_client(agent, **kw))
    monkeypatch.setattr(agent, "_close_request_openai_client",
                        lambda *a, **kw: AIAgent._close_request_openai_client(agent, *a, **kw))
    def abort(client, **kwargs):
        aborted.append(client)
        return AIAgent._abort_request_openai_client(agent, client, **kwargs)
    monkeypatch.setattr(agent, "_abort_request_openai_client", abort)
    def advance():
        if accepted.wait(3):
            clock.now = 120
    advance_thread = threading.Thread(target=advance)
    advance_thread.start()
    try:
        result = agent.run_conversation("Controlled silent server test", system_message="Answer briefly.")
        assert result["blocked"]
        assert result["inference_gate"]["cancel_count"] == 1
        assert result["inference_gate"]["cancel_result"] == "CANCELLED_CONFIRMED"
        assert len(aborted) == len(posts) == 1
        assert aborted[0] is not agent.client
        assert not agent.client.is_closed()
    finally:
        release.set()
        advance_thread.join(3)
        server.shutdown()
        server.server_close()
        serving.join(3)
