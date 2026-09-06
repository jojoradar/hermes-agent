"""Terminal policy hooks for the existing streaming worker and monitor."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

from agent.inference_gate import InferenceBlocked, get_gate


def initialize(call):
    call.inference_gate = get_gate(call.agent)
    call.inference_clock = call.inference_gate.new_request() if call.inference_gate else None
    call.inference_stream_ended = False


def trip(call, code, reason):
    gate, request = call.inference_gate, call.inference_clock
    gate.fence(code, reason)
    with gate.lock:
        call._request_cancelled["value"] = True
        if request.cancel_count:
            return
        request.cancel_count = 1
        request.cancel_deadline = time.monotonic() + 10.0
    call._cancel_current_stream_attempt("terminal_inference_gate")
    # Existing registry aborts ONLY request-owned sockets from a stranger thread.
    # The worker's finally owns actual pool close; no shared client is closed.
    call.clients.close_once("terminal_inference_gate")


def check_clock(call):
    gate = call.inference_gate
    if gate is None:
        return True
    with gate.lock:
        expired = call.inference_clock.expired() if gate.terminal is None else None
        if expired:
            trip(call, *expired)
    return gate.terminal is None


def before_send(call, kwargs):
    gate = call.inference_gate
    if gate is None:
        return kwargs
    with gate.lock:
        gate.check()
        if call.inference_clock.sent:
            trip(call, "AGENT_STALLED", "resend_refused")
            gate.check()
        call.inference_clock.sent = True
    result = dict(kwargs)
    # Keep the native inline request owner for cron/delegated contexts. Its
    # monitor fences/aborts independently; also remove unbounded socket waits.
    # A failed abort remains UNKNOWN, never permission to replay the request.
    import httpx
    old_timeout = result.get("timeout")
    old_timeout = old_timeout if isinstance(old_timeout, httpx.Timeout) else httpx.Timeout(old_timeout)
    bound = gate.limits.transport
    timeouts = {name: min(getattr(old_timeout, name) or bound, bound)
                for name in ("connect", "write", "pool")}
    # The socket must permit cold prefill. After the first substantive delta,
    # the existing independent monitor still aborts a 120s generation stall.
    # A role-only frame cannot move either deadline.
    timeouts["read"] = min(max(gate.limits.first_token, bound), gate.limits.absolute)
    result["timeout"] = httpx.Timeout(**timeouts)
    key = "max_completion_tokens" if "max_completion_tokens" in result else "max_tokens"
    result[key] = min(int(result.get(key) or gate.limits.output), gate.limits.output)
    return result


def observe(call, chunk):
    if call.inference_gate is not None and check_clock(call):
        with call.inference_gate.lock:
            if call.inference_gate.terminal is None:
                call.inference_clock.observe(chunk)


def on_error(call, error):
    if call.inference_gate is None or (call._request_cancelled["value"] and call.inference_gate.terminal is None):
        return False
    # A disconnected/invalid stream is terminal too. No partial replay, retry, or
    # synthetic length response may turn an ambiguous request into another POST.
    if call.inference_gate.terminal is None:
        trip(call, "INFERENCE_STALLED", "stream_ended_without_completion")
    call.result["error"] = InferenceBlocked(call.inference_gate.terminal["code"])
    return True


def monitor(call):
    if call.inference_gate is None:
        return False
    if check_clock(call):
        return False
    # One bounded wait, never a restart or second abort.
    call._call_done.wait(timeout=max(0, call.inference_clock.cancel_deadline - time.monotonic()))
    return True


def local_lane_idle(agent, timeout):
    """Read the supported CLI status only for the local server it describes."""
    endpoint = urlsplit(str(agent.base_url))
    if endpoint.hostname not in {"localhost", "127.0.0.1", "::1"} or endpoint.port != 1234:
        return False
    command = shutil.which("lms") or str(Path.home() / ".lmstudio" / "bin" / "lms")
    try:
        proc = subprocess.run([command, "ps", "--json"], capture_output=True, text=True,
                              timeout=max(0.05, min(2.0, timeout)), check=True)
        models = json.loads(proc.stdout)
        matching = [m for m in models if m.get("identifier") == agent.model]
        return bool(matching) and all(m.get("status") == "idle" and m.get("queued") == 0 for m in models)
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def finish(call):
    gate = call.inference_gate
    if gate is None:
        return
    if gate.terminal is None and call.agent._interrupt_requested:
        return  # explicit user interrupt keeps its existing redirect semantics
    if gate.terminal is None:
        check_clock(call)
    if gate.terminal is not None:
        # Locally ended matching stream AND aggregate idle are both necessary.
        # No socket-close-only acknowledgement; unavailable status stays UNKNOWN.
        deadline = call.inference_clock.cancel_deadline or time.monotonic()
        while call._call_done.is_set() and call.inference_stream_ended and time.monotonic() < deadline:
            if local_lane_idle(call.agent, deadline - time.monotonic()):
                gate.terminal["cancel_result"] = "CANCELLED_CONFIRMED"
                break
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        raise InferenceBlocked(gate.terminal["code"])
    from agent.inference_gate import validate_response
    try:
        validate_response(gate, call.result["response"], call.api_kwargs.get("tools"))
    except InferenceBlocked:
        # Provider already closed a length/empty/invalid completion; there is no
        # in-flight request to abort. Keep the turn fenced without inventing a cancel.
        gate.terminal["cancel_result"] = "REQUEST_ALREADY_ENDED"
        raise
    call.inference_clock.ended = True
