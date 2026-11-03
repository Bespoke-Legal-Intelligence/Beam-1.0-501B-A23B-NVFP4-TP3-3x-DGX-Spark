# beam_sched1_patcher_v1_0_1.py (v1.0.1: tolerant of a scheduler without scheduler_config; prints one
# 'LOCAL PATCH sched1 active' line to the engine log the first time it throttles) - make vLLM's existing prefill throttle work on a single (non-data-parallel) engine.
# Edits ONE function in vllm/v1/engine/core.py: EngineCore._should_throttle_prefills, which in this build always
# returns False unless the data-parallel engine overrides it. After the edit it honours --prefill-schedule-interval N:
# on N-1 of every N steps it asks the scheduler to defer in-progress prompt chunks. The scheduler itself only defers
# when at least one request is decoding (scheduler.py: defer_prefills ... and any(not r.is_prefill_chunk ...)),
# so a prompt read on its own is never slowed. Usage: python3 -I beam_sched1_patcher_v1_0_1.py <core.py in> <core.py out>
import ast, hashlib, sys
src, dst = sys.argv[1], sys.argv[2]
OLD = (
    "    def _should_throttle_prefills(self) -> bool:\n"
    "        \"\"\"Whether to defer new prefills this step (DP prefill balancing).\n"
    "        Overridden by the DP engine core; never throttles otherwise.\"\"\"\n"
    "        return False\n"
)
NEW = (
    "    def _should_throttle_prefills(self) -> bool:\n"
    "        \"\"\"Whether to defer new prefills this step (DP prefill balancing).\n"
    "        Overridden by the DP engine core; never throttles otherwise.\"\"\"\n"
    "        # LOCAL PATCH sched1 (2026-10-30, Beam TP=3): also honour --prefill-schedule-interval\n"
    "        # on a single (non-DP) engine, so running chats get decode-only steps between chunks\n"
    "        # of a long prompt. The scheduler only defers when some request is decoding.\n"
    "        sched_cfg = getattr(self.scheduler, \"scheduler_config\", None)\n"
    "        n = getattr(sched_cfg, \"prefill_schedule_interval\", 1) or 1\n"
    "        if n <= 1:\n"
    "            return False\n"
    "        if not getattr(self, \"_sched1_logged\", False):\n"
    "            self._sched1_logged = True\n"
    "            print(\"LOCAL PATCH sched1 active: prefill_schedule_interval=%d\" % n, flush=True)\n"
    "        self._sched1_step = getattr(self, \"_sched1_step\", 0) + 1\n"
    "        return self._sched1_step % n != 0\n"
)
text = open(src).read()
print("ORIGINAL_SHA %s" % hashlib.sha256(text.encode()).hexdigest())
c = text.count(OLD)
if c != 1:
    print("PATCH=FAIL expected the original function exactly once, found %d; nothing written" % c); sys.exit(1)
out = text.replace(OLD, NEW)
ast.parse(out)
open(dst, "w").write(out)
print("PATCHED_SHA %s" % hashlib.sha256(out.encode()).hexdigest())
print("PATCH=OK lines %d -> %d" % (text.count("\n"), out.count("\n")))
