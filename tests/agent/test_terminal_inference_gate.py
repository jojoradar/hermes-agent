"""Three clocks and terminal boundaries, with deterministic time and no model/network."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from agent.inference_gate import (InferenceBlocked, InferenceClock, Limits, TurnInferenceGate,
                                  admit_turn, validate_response)
from agent.inference_gate_stream import check_clock, before_send, trip


class Clock:
    now = 0.0
    def __call__(self):
        return self.now


def chunk(**delta):
    return NS(choices=[NS(delta=NS(**delta), finish_reason=None)])


def make_call(limits=Limits(first_token=120)):
    clock = Clock()
    agent = NS(_emit_status=Mock(), _strip_think_blocks=lambda t: t,
               _flush_messages_to_session_db=Mock(return_value=True))
    gate = agent._terminal_inference_gate = TurnInferenceGate(agent, limits, clock=clock)
    gate.messages = []
    request = gate.new_request()
    call = NS(agent=agent, inference_gate=gate, inference_clock=request,
              _request_cancelled={"value": False}, _cancel_current_stream_attempt=Mock(),
              clients=NS(close_once=Mock()))
    return call, clock


@pytest.mark.parametrize("kind,deadline,code", [
    ("empty", 120, "INFERENCE_STALLED"), ("role", 120, "INFERENCE_STALLED"),
    ("metadata", 120, "INFERENCE_STALLED"), ("reasoning", 300, "AGENT_STALLED"),
    ("partial_tool", 300, "AGENT_STALLED"), ("inline_think", 300, "AGENT_STALLED"),
    ("answer", 600, "AGENT_STALLED"),
])
def test_clocks_fence_once_and_never_resend(kind, deadline, code):
    call, clock = make_call()
    before_send(call, {"max_tokens": 65536})
    frames = {"empty": chunk(), "role": chunk(role="assistant"),
              "metadata": NS(choices=[]), "reasoning": chunk(reasoning_content="thinking"),
              "partial_tool": chunk(tool_calls=[NS(function=NS(arguments='{"x":'))]),
              "inline_think": chunk(content="thinking "), "answer": chunk(content="answer ")}
    if kind == "inline_think":
        call.inference_clock.observe(chunk(content="<thi"))
        call.inference_clock.observe(chunk(content="nk>"))
    for second in range(1, deadline):
        clock.now = second
        assert check_clock(call)
        call.inference_clock.observe(frames[kind])
    clock.now = deadline
    assert not check_clock(call)
    assert call.inference_gate.terminal["code"] == code
    assert call.inference_clock.cancel_count == 1
    trip(call, code, "late_second_trip")
    call.clients.close_once.assert_called_once()
    with pytest.raises(InferenceBlocked):
        before_send(call, {})
    with pytest.raises(InferenceBlocked):
        call.inference_gate.new_request()
    assert call.inference_clock.sent
    assert call.inference_gate.persisted


def test_transport_content_updates_and_absolute_survives_reasoning():
    clock = Clock()
    request = InferenceClock(Limits(useful=1000), clock)
    for second in range(1, 600):
        clock.now = second
        request.observe(chunk(reasoning_content="r"))
        assert request.expired() is None
    clock.now = 600
    assert request.expired() == ("AGENT_STALLED", "absolute_limit")
    request = InferenceClock(Limits(), clock)
    clock.now = 719
    request.observe(chunk(content="OK"))
    clock.now = 720
    assert request.expired() is None
    assert request.transport == request.useful == 719


@pytest.mark.parametrize("first_token", [180, 240, 300])
def test_cold_prefill_does_not_consume_generation_stall_budget(first_token):
    call, clock = make_call(Limits(first_token=first_token))
    before_send(call, {})
    for second in range(1, first_token):
        clock.now = second
        call.inference_clock.observe(chunk(role="assistant"))
        assert check_clock(call)
        assert call.inference_clock.first_delta_at is None
    call.inference_clock.observe(chunk(content="OK"))
    assert call.inference_clock.first_delta_at == first_token - 1
    clock.now += 119
    assert check_clock(call)
    clock.now += 1
    assert not check_clock(call)
    assert call.inference_gate.terminal["reason"] == "transport_stall"
    call.clients.close_once.assert_called_once()


@pytest.mark.parametrize("first_token", [180, 240, 300])
def test_first_token_timeout_keeps_terminal_cancel_and_zero_resend(first_token):
    call, clock = make_call(Limits(first_token=first_token))
    before_send(call, {})
    clock.now = first_token
    call.inference_clock.observe(chunk(role="assistant"))
    assert not check_clock(call)
    assert call.inference_gate.terminal["reason"] == "first_token_timeout"
    with pytest.raises(InferenceBlocked):
        before_send(call, {})
    assert call.inference_clock.cancel_count == 1
    call.clients.close_once.assert_called_once()


def test_first_reasoning_delta_does_not_restart_useful_progress_clock():
    call, clock = make_call(Limits(first_token=240))
    before_send(call, {})
    for second in range(200, 300):
        clock.now = second
        call.inference_clock.observe(chunk(reasoning_content="r"))
        assert check_clock(call)
    clock.now = 300
    assert not check_clock(call)
    assert call.inference_gate.terminal["reason"] == "useful_progress"
    assert call.inference_clock.first_delta_at == 200
    assert call.inference_clock.useful == 0


def test_socket_read_permits_prefill_without_widening_generation_monitor():
    import httpx
    call, clock = make_call(Limits(first_token=240))
    kwargs = before_send(call, {"timeout": httpx.Timeout(120)})
    assert kwargs["timeout"].read == 240
    clock.now = 200
    call.inference_clock.observe(chunk(content="OK"))
    clock.now = 320
    assert not check_clock(call)
    assert call.inference_gate.terminal["reason"] == "transport_stall"


SCHEMA = [{"type": "function", "function": {"name": "harmless", "parameters": {
    "type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"],
    "additionalProperties": False}}}]


@pytest.mark.parametrize("args,finish,accepted", [
    ('{"value":1}', "tool_calls", True), ('{"value":', "tool_calls", False),
    ('{"value":"bad"}', "tool_calls", False), ('{}', "tool_calls", False),
    ('{"value":1}', "length", False), ('{"value":1}', None, False),
])
def test_only_complete_schema_valid_calls_are_executable(args, finish, accepted):
    call, clock = make_call()
    response = NS(choices=[NS(finish_reason=finish, message=NS(content=None,
        tool_calls=[NS(id="call-1", function=NS(name="harmless", arguments=args))]))])
    clock.now = 200
    if accepted:
        validate_response(call.inference_gate, response, SCHEMA)
        assert call.inference_clock.useful == 200
    else:
        with pytest.raises(InferenceBlocked):
            validate_response(call.inference_gate, response, SCHEMA)
        assert call.inference_gate.result()["blocked"]


@pytest.mark.parametrize("finish,content", [("length", "partial"), ("stop", ""), (None, "text")])
def test_no_length_empty_or_partial_continuation(finish, content):
    call, _ = make_call()
    response = NS(choices=[NS(finish_reason=finish, message=NS(content=content, tool_calls=None))])
    with pytest.raises(InferenceBlocked):
        validate_response(call.inference_gate, response, [])
    assert call.inference_gate.result()["retryable"] is False


def test_output_cap_is_on_the_final_wire_kwargs():
    for cap in (8192, 16384):
        call, _ = make_call(Limits(output=cap))
        assert before_send(call, {"max_tokens": 65536})["max_tokens"] == cap
        with pytest.raises(InferenceBlocked):
            before_send(call, {})  # middleware/provider retry cannot create another POST


@pytest.mark.parametrize("provider,receipt", [
    ("lmstudio", True), ("openrouter", True), ("lmstudio", False)])
def test_durable_fence_blocks_new_agent_and_recovery(tmp_path, monkeypatch, provider, receipt):
    from hermes_state import SessionDB
    import hermes_cli.config as config
    monkeypatch.setattr(config, "load_config_readonly", lambda: {
        "agent": {"terminal_inference_gate": {"enabled": True}}})
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("fenced", source="cli")
    terminal = {"code": "AGENT_STALLED", "reason": "useful_progress", "status": "BLOCKED"}
    db.append_message("fenced", "assistant", "BLOCKED", finish_reason="inference_terminal",
                      display_metadata={"inference_terminal": terminal} if receipt else {})
    db.close()
    db = SessionDB(db_path=tmp_path / "state.db")
    agent = NS(provider=provider, model="model", session_id="fenced", _session_db=db)
    assert admit_turn(agent)["blocked"]
    assert admit_turn(agent)["retryable"] is False
    db.close()


@pytest.mark.parametrize("field", ["absolute_seconds", "first_token_seconds"])
@pytest.mark.parametrize("ceiling", [float("inf"), float("nan"), 0, -1])
def test_ceiling_cannot_be_disabled(monkeypatch, ceiling, field):
    import hermes_cli.config as config
    monkeypatch.setattr(config, "load_config_readonly", lambda: {"agent": {
        "terminal_inference_gate": {"enabled": True, field: ceiling}}})
    with pytest.raises(ValueError):
        admit_turn(NS(provider="lmstudio", model="model"))
