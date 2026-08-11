"""
Recursive Language Model (RLM) Engine.

Implements the core orchestration loop from the Recursive Language Models
paper (Zhang, Kraska, Khattab).  The long input lives as a Python variable
in a sandboxed REPL; the orchestrating LLM writes arbitrary code against it,
including recursive sub-calls to fresh LLM instances on data slices.

This module is self-contained: it depends only on backend.ai for the shared
NIM client and never imports graph, memory, entity-resolution, or
conflict-detection code.
"""

import json
import re
import sys
import time
import subprocess
import threading
from pydantic import BaseModel
from backend.ai import client


# ---------------------------------------------------------------------------
# Configuration & stats models
# ---------------------------------------------------------------------------

class RLMConfig(BaseModel):
    """Tunable safety / cost caps for a single RLM query."""
    max_recursion_depth: int = 5
    max_total_llm_subcalls: int = 15
    max_wall_clock_seconds: float = 60.0
    max_total_tokens: int = 100_000


class RLMStats(BaseModel):
    """Usage telemetry returned alongside every answer."""
    subcalls_made: int = 0
    recursion_depth_reached: int = 0
    tokens_used: int = 0
    elapsed_seconds: float = 0.0
    capped: bool = False


# ---------------------------------------------------------------------------
# Sandbox bootstrap — injected before every user-generated code snippet
# ---------------------------------------------------------------------------

_SANDBOX_BOOTSTRAP = '''\
import sys as _sys
import json as _json
import builtins as _builtins

# ── Read context payload from parent via stdin ──────────────────────────
_raw_line = _sys.stdin.readline()
try:
    CONTEXT_DATA = _json.loads(_raw_line)["context"]
except Exception:
    CONTEXT_DATA = ""
del _raw_line

# ── IPC helpers for llm_ask ─────────────────────────────────────────────
_ipc_print = _builtins.print
_ipc_input = _builtins.input

def llm_ask(data, question):
    """Ask a fresh LLM to analyse *data* and answer *question*."""
    payload = _json.dumps({
        "__type__": "llm_ask",
        "context": str(data),
        "query": str(question),
    })
    _ipc_print(payload, flush=True)
    try:
        resp = _ipc_input()
        return _json.loads(resp).get("answer", "")
    except Exception:
        return "[llm_ask: IPC error]"

# ── DEFENSE IN DEPTH — restrict dangerous operations ───────────────────

# 1. Remove filesystem access from builtins
if hasattr(_builtins, "open"):
    del _builtins.open

# 2. Whitelist-based import restriction
_ALLOWED_MODULES = frozenset({
    "json", "re", "math", "collections", "itertools", "functools",
    "string", "textwrap", "unicodedata", "datetime", "statistics",
    "decimal", "fractions", "random", "bisect", "heapq",
    "copy", "pprint", "enum", "dataclasses", "typing",
    "hashlib", "base64", "struct", "operator", "numbers",
    "io", "abc", "contextlib", "types", "warnings",
})

_original_import = _builtins.__import__

def _safe_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top not in _ALLOWED_MODULES:
        raise ImportError(f"Module \\'{name}\\' is blocked by sandbox policy")
    return _original_import(name, *args, **kwargs)

_builtins.__import__ = _safe_import

# 3. Nullify dangerous modules already cached in sys.modules
for _blocked in (
    "os", "subprocess", "socket", "shutil", "pathlib",
    "urllib", "urllib.request", "http", "http.client",
    "ftplib", "smtplib", "ctypes", "multiprocessing",
    "signal", "webbrowser",
):
    _sys.modules[_blocked] = None
del _blocked

# ── User code follows ──────────────────────────────────────────────────
'''


# ---------------------------------------------------------------------------
# Orchestrator prompts
# ---------------------------------------------------------------------------

_ORCHESTRATOR_SYSTEM = """\
You are an advanced Recursive Language Model (RLM) orchestrator.
You must answer the user's query by exploring a large dataset.
The dataset is loaded into a Python REPL as a string variable called \
`CONTEXT_DATA`.

On each turn you MUST do exactly ONE of two things:

1. **Run code** — write a Python script inside a fenced block:
   ```python
   # your code here — use print() to see results
   ```
   The script runs in a sandbox with CONTEXT_DATA available.
   You may also call  llm_ask(slice, question) -> str  to send a
   data slice to a fresh LLM for analysis; its answer is returned
   as a string you can print or store.

2. **Return the final answer** — when you have enough information:
   <ANSWER>your answer here</ANSWER>

Rules:
- Always print() what you want to see; unprinted values are lost.
- Keep code short and focused; one exploration step per turn.
- You have limited turns — be efficient.
"""

_ORCHESTRATOR_USER = "Query: {query}"

_SUBCALL_PROMPT = """\
Answer the following question based ONLY on the provided context.

Context:
{context}

Question: {query}

Answer:"""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RLMEngine:
    """Runs a recursive-decomposition query loop over long context."""

    def __init__(self, config: RLMConfig | None = None):
        self.config = config or RLMConfig()
        self.stats = RLMStats()
        self.start_time = time.time()
        self._lock = threading.Lock()

    # ── Cap checking ────────────────────────────────────────────────────

    def _check_caps(self) -> bool:
        with self._lock:
            if time.time() - self.start_time > self.config.max_wall_clock_seconds:
                self.stats.capped = True
            if self.stats.subcalls_made >= self.config.max_total_llm_subcalls:
                self.stats.capped = True
            if self.stats.tokens_used >= self.config.max_total_tokens:
                self.stats.capped = True
        return self.stats.capped

    def _estimate_tokens(self, text: str) -> int:
        """Best-effort token count (~4 chars / token)."""
        return max(1, len(text) // 4)

    # ── LLM subcall ────────────────────────────────────────────────────

    def _do_subcall(self, context_slice: str, query: str, depth: int) -> str:
        with self._lock:
            self.stats.recursion_depth_reached = max(
                self.stats.recursion_depth_reached, depth
            )
            self.stats.subcalls_made += 1

        if depth > self.config.max_recursion_depth:
            self.stats.capped = True
            return "[Error: max recursion depth reached]"

        if self._check_caps():
            return "[Error: safety caps reached]"

        prompt = _SUBCALL_PROMPT.format(context=context_slice, query=query)
        self.stats.tokens_used += self._estimate_tokens(prompt)

        try:
            resp = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            ans = resp.choices[0].message.content or ""
            self.stats.tokens_used += self._estimate_tokens(ans)
            return ans
        except Exception as exc:
            return f"[LLM API error: {exc}]"

    # ── Sandbox execution ───────────────────────────────────────────────

    def _run_sandbox(self, code: str, context_data: str, current_depth: int) -> str:
        """
        Execute *code* in an isolated subprocess with CONTEXT_DATA injected.

        Returns the captured stdout (with IPC lines filtered out).
        Appends a ``[Capped: ...]`` marker if the wall-clock or other cap
        fires before the subprocess exits naturally.
        """
        full_script = _SANDBOX_BOOTSTRAP + code

        proc = subprocess.Popen(
            [sys.executable, "-c", full_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Write context via stdin in a daemon thread (avoids pipe dead-lock
        # when context_data is larger than the OS pipe buffer).
        def _feed_context():
            try:
                proc.stdin.write(json.dumps({"context": context_data}) + "\n")
                proc.stdin.flush()
            except Exception:
                pass

        feeder = threading.Thread(target=_feed_context, daemon=True)
        feeder.start()

        output_parts: list[str] = []
        deadline = self.start_time + self.config.max_wall_clock_seconds

        while True:
            remaining = deadline - time.time()
            if remaining <= 0 or self._check_caps():
                try:
                    proc.kill()
                except OSError:
                    pass
                self.stats.capped = True
                output_parts.append("[Capped: Safety limits exceeded]\n")
                break

            # Threaded readline so we can enforce the wall-clock cap even
            # when the child is stuck (e.g. infinite loop without output).
            line_box: list[str | None] = [None]
            read_done = threading.Event()

            def _readline():
                try:
                    line_box[0] = proc.stdout.readline()
                except Exception:
                    line_box[0] = ""
                read_done.set()

            rt = threading.Thread(target=_readline, daemon=True)
            rt.start()
            read_done.wait(timeout=min(remaining, 2.0))

            if not read_done.is_set():
                # readline still blocking — loop back to re-check caps
                continue

            line = line_box[0]
            if not line:
                break  # subprocess exited or closed stdout

            # ── IPC: llm_ask request from sandbox ───────────────────────
            stripped = line.strip()
            try:
                msg = json.loads(stripped)
                if isinstance(msg, dict) and msg.get("__type__") == "llm_ask":
                    ans = self._do_subcall(
                        msg.get("context", ""),
                        msg.get("query", ""),
                        current_depth + 1,
                    )
                    try:
                        proc.stdin.write(json.dumps({"answer": ans}) + "\n")
                        proc.stdin.flush()
                    except Exception:
                        pass
                    continue  # don't include IPC line in visible output
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

            output_parts.append(line)

        # Ensure the subprocess is fully reaped
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

        return "".join(output_parts)

    # ── Main query loop ─────────────────────────────────────────────────

    def query(self, context_data: str, query_text: str) -> tuple[str, RLMStats]:
        """
        Run the full RLM orchestration loop.

        Returns (answer_string, stats).
        """
        self.start_time = time.time()

        messages = [
            {"role": "system", "content": _ORCHESTRATOR_SYSTEM},
            {"role": "user", "content": _ORCHESTRATOR_USER.format(query=query_text)},
        ]
        self.stats.tokens_used += self._estimate_tokens(
            _ORCHESTRATOR_SYSTEM + query_text
        )

        final_answer = ""

        while not self._check_caps():
            try:
                resp = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1024,
                )
                content = resp.choices[0].message.content or ""
                self.stats.tokens_used += self._estimate_tokens(content)
                messages.append({"role": "assistant", "content": content})

                with self._lock:
                    self.stats.subcalls_made += 1

            except Exception as exc:
                final_answer = f"Error during LLM call: {exc}"
                break

            # ── Check for final answer ──────────────────────────────────
            ans_match = re.search(
                r"<ANSWER>(.*?)</ANSWER>", content, re.DOTALL
            )
            if ans_match:
                final_answer = ans_match.group(1).strip()
                break

            # ── Check for executable code ───────────────────────────────
            code_match = re.search(
                r"```python\s*\n(.*?)```", content, re.DOTALL
            )
            if code_match:
                code = code_match.group(1)
                stdout = self._run_sandbox(code, context_data, current_depth=1)
                observation = (
                    f"Execution output:\n{stdout.strip() or '[no output]'}\n"
                )
                messages.append({"role": "user", "content": observation})
                self.stats.tokens_used += self._estimate_tokens(observation)
            else:
                nudge = (
                    "You must output either a ```python code block or "
                    "an <ANSWER>...</ANSWER> block. Try again."
                )
                messages.append({"role": "user", "content": nudge})
                self.stats.tokens_used += self._estimate_tokens(nudge)

        self.stats.elapsed_seconds = time.time() - self.start_time

        if not final_answer:
            self.stats.capped = True
            final_answer = (
                "Processing stopped: safety/cost cap reached before a "
                "final answer could be produced."
            )

        return final_answer, self.stats
