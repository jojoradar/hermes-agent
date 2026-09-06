"""Opt-in terminal inference policy at the existing request/turn boundaries.

The transcript is the durable fence. A fenced session requires operator reconciliation;
neither a new agent instance nor a gateway recovery may silently resume it.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass
from agent.think_scrubber import StreamingThinkScrubber

logger = logging.getLogger(__name__)
MARKER = "inference_terminal"


class InferenceBlocked(RuntimeError):
    """Terminal control flow, never a transient provider error."""


@dataclass(frozen=True)
class Limits:
    transport: float = 120.0
    useful: float = 300.0
    absolute: float = 600.0
    output: int = 8192


class InferenceClock:
    def __init__(self, limits: Limits, clock=time.monotonic):
        self.limits, self.clock = limits, clock
        self.started = self.transport = self.useful = clock()
        self.ended = False
        self.sent = False
        self.cancel_count = 0
        self.cancel_deadline = None
        self.tools_pending = False
        self.scrubber = StreamingThinkScrubber()

    def expired(self):
        if self.ended:
            return None
        now = self.clock()
        if now - self.started >= self.limits.absolute:
            return "AGENT_STALLED", "absolute_limit"
        if now - self.transport >= self.limits.transport:
            return "INFERENCE_STALLED", "transport_stall"
        if now - self.useful >= self.limits.useful:
            return "AGENT_STALLED", "useful_progress"
        return None

    def observe(self, chunk):
        from agent.message_content import flatten_message_text
        for choice in getattr(chunk, "choices", None) or []:
            delta = getattr(choice, "delta", None)
            content = flatten_message_text(getattr(delta, "content", None), sep="")
            reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            arguments = any(getattr(getattr(tc, "function", None), "arguments", None)
                            for tc in getattr(delta, "tool_calls", None) or [])
            self.tools_pending |= bool(getattr(delta, "tool_calls", None))
            if content or reasoning or arguments:
                self.transport = self.clock()
            # Split inline <think> tags must not turn reasoning into answer progress.
            if isinstance(content, str) and self.scrubber.feed(content).strip() and not self.tools_pending:
                self.useful = self.clock()


class TurnInferenceGate:
    def __init__(self, agent, limits: Limits, *, clock=time.monotonic):
        self.agent, self.limits, self.clock = agent, limits, clock
        self.lock = threading.RLock()
        self.terminal = None
        self.messages = None
        self.history = None
        self.request = None
        self.persisted = False
        self.turn_active = False

    def check(self):
        if self.terminal is not None:
            raise InferenceBlocked(self.terminal["code"])

    def new_request(self):
        with self.lock:
            self.check()
            self.request = InferenceClock(self.limits, self.clock)
            return self.request

    def fence(self, code, reason):
        with self.lock:
            if self.terminal is not None:
                return
            self.terminal = {"status": "BLOCKED", "code": code, "reason": reason,
                             "turn_id": str(getattr(self.agent, "_current_turn_id", "") or ""),
                             "api_request_id": str(getattr(self.agent, "_current_api_request_id", "") or ""),
                             "cancel_result": "CANCEL_UNKNOWN", "zero_resend": True}
        # Publish the in-memory fence BEFORE durable recording or cancellation.
        if self.messages is not None:
            marker = {"role": "assistant", "content": self.notice(), "finish_reason": MARKER,
                      "display_metadata": {MARKER: dict(self.terminal)}}
            self.messages.append(marker)
            try:
                self.persisted = self.agent._flush_messages_to_session_db(self.messages, self.history) is True
            except Exception:
                logger.exception("Terminal inference fence persistence failed")
        logger.warning("%s status=BLOCKED reason=%s zero_resend=true", code, reason)
        try:
            self.agent._emit_status(self.notice())
        except Exception:
            logger.exception("Terminal inference status delivery failed")

    def notice(self):
        return (f"{self.terminal['code']}: BLOCKED ({self.terminal['reason']}). "
                "Inference stopped. No automatic resend. Operator reconciliation is required.")

    def result(self, messages=None, api_calls=0):
        return {"final_response": self.notice(), "messages": messages or self.messages or [],
                "api_calls": api_calls, "completed": False, "failed": True, "blocked": True,
                "status": "BLOCKED", "error": self.terminal["code"],
                "failure_reason": self.terminal["code"], "retryable": False,
                "inference_gate": {**self.terminal, "fence_persisted": self.persisted,
                                   "cancel_count": self.request.cancel_count if self.request else 0}}

    def record_cancel_result(self):
        db = getattr(self.agent, "_session_db", None)
        if self.persisted and db is not None:
            try:
                db.update_inference_terminal_marker(self.agent.session_id, self.terminal)
            except Exception:
                self.persisted = False
                logger.exception("Terminal cancellation receipt persistence failed")


def get_gate(agent):
    value = getattr(agent, "_terminal_inference_gate", None)
    return value if isinstance(value, TurnInferenceGate) else None


def require_unfenced(agent):
    gate = get_gate(agent)
    if gate is not None:
        gate.check()
    return gate


def admit_turn(agent, history=None):
    """Called before preflight/auxiliary work, including recovery with a fresh agent."""
    old = get_gate(agent)
    if old is not None and old.terminal:
        return old.result(history)
    if old is not None and old.turn_active:
        return {"final_response": "Session already has an active guarded turn.",
                "messages": history or [], "completed": False, "failed": True,
                "blocked": True, "retryable": False, "error": "session_turn_busy"}
    from hermes_cli.config import load_config_readonly
    cfg = (load_config_readonly().get("agent") or {}).get("terminal_inference_gate") or {}
    db, sid = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
    # A different provider or a rebuilt agent must not bypass an existing fence.
    if db is not None and sid:
        marker = db.get_inference_terminal_marker(sid)
        if isinstance(marker, dict) and marker:
            gate = agent._terminal_inference_gate = TurnInferenceGate(agent, Limits())
            gate.terminal, gate.persisted = marker, True
            return gate.result(history)
    if cfg.get("enabled") is not True or getattr(agent, "provider", None) != "lmstudio":
        agent._terminal_inference_gate = None
        return None
    if cfg.get("model") and cfg["model"] != agent.model:
        agent._terminal_inference_gate = None
        return None
    limits = Limits(float(cfg.get("transport_seconds", 120)), float(cfg.get("useful_seconds", 300)),
                    float(cfg.get("absolute_seconds", 600)), int(cfg.get("max_output_tokens", 8192)))
    if any(not math.isfinite(v) or v <= 0 for v in
           (limits.transport, limits.useful, limits.absolute, limits.output)):
        raise ValueError("Terminal inference limits must be finite and positive")
    gate = agent._terminal_inference_gate = TurnInferenceGate(agent, limits)
    if db is None or not sid:
        gate.fence("AGENT_STALLED", "durable_fence_unavailable")
        return gate.result(history)
    return None


def validate_response(gate, response, schemas):
    """Require a complete response before any outer recovery or tool executor sees it."""
    gate.check()
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    finish = getattr(choice, "finish_reason", None)
    message = getattr(choice, "message", None)
    calls = getattr(message, "tool_calls", None) or []
    if finish == "length":
        gate.fence("AGENT_STALLED", "output_limit")
    elif finish not in {"stop", "tool_calls"}:
        gate.fence("AGENT_STALLED", "incomplete_stream")
    elif calls:
        from jsonschema.validators import validator_for
        from tools.tool_search_validation import _schema_for_local_validation, _schema_has_external_ref
        by_name = {s["function"]["name"]: s["function"].get("parameters", {})
                   for s in schemas or [] if s.get("type") == "function"}
        try:
            for call in calls:
                fn = call.function
                if not call.id or fn.name not in by_name or not isinstance(fn.arguments, str):
                    raise ValueError("incomplete tool identity")
                args = json.loads(fn.arguments)
                schema = _schema_for_local_validation(by_name[fn.name])
                if not isinstance(args, dict) or _schema_has_external_ref(schema):
                    raise ValueError("unvalidated tool arguments")
                validator = validator_for(schema)
                validator.check_schema(schema)
                validator(schema).validate(args)
        except Exception:
            gate.fence("AGENT_STALLED", "invalid_tool_call")
        else:
            gate.request.useful = gate.clock()
    else:
        content = getattr(message, "content", None) or ""
        visible = gate.agent._strip_think_blocks(content) if isinstance(content, str) else ""
        if not visible.strip():
            gate.fence("AGENT_STALLED", "empty_or_reasoning_only_completion")
    gate.check()


def blocked_turn_result(agent, state):
    gate = get_gate(agent)
    assert gate is not None and gate.terminal is not None
    from agent.turn_api_call import stop_thinking_spinner
    stop_thinking_spinner(agent, state.thinking_spinner)
    gate.record_cancel_result()
    agent._persist_session(state.messages, state.conversation_history)
    # Already-dispatched background tools retain their own lifecycle. A model
    # deadline must not clean up their VM/browser or kill their process.
    return gate.result(state.messages, state.api_call_count)
