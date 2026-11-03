<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Changelog

## v2.0.0 (Nov 2, 2026)

The padded checkpoint, its `SHA256SUMS` (`9f57e87b…`) and the base image are unchanged. Everything new is in how the
model is served. Measured on the same three nodes; see `results/v2_20261031/README.md`.

**Added**
- `scripts/launch_node_v0_2_1.sh`: v0.1.4 plus optional `RETENTION`, `PREFILL_INTERVAL`, `LONG_PREFILL`, `LONG_PREFILL_ADAPTIVE` and `PROF_DIR`. With none set it launches exactly as v0.1.4. The Oct 28–31 runs used a private v0.2.0, identical on the success path; the Nov 1 v1.0.1 runs used v0.2.1 as published. v0.2.1 changes only failure handling: its "container already exists" hint gives the graceful stop instead of `docker rm -f`, a `PROF_DIR` that cannot be created stops the launch, and `STARTED` is printed only if the container is still running after 10 s.
- `docker/Dockerfile.sched1` and `docker/beam_sched1_patcher_v1_0_1.py`: node0's scheduler image. Edits one function, `EngineCore._should_throttle_prefills`, so a single-replica engine honours `--prefill-schedule-interval` (in this build only the data-parallel engine did). The patcher is byte-identical to the one used for the measurements; the image there was built by applying it to a copy of the same file, here it is applied inside the build.
- `scripts/start_all_v1_0_1.sh`, `scripts/stop_all_v1_0_1.sh`, `scripts/cluster_v2.conf.example`: one-command start and graceful stop from node0, generalised from the scripts we ran (node list and paths moved into the config file), with a preflight on all nodes, a check for the launcher's `STARTED` line, and stop results that flag unreachable nodes and timed-out stops. v1.0.1 over the v1.0.0 we ran on Nov 1: `start_all` derives `GPU_UTIL` from the memory total when the config does not set it, and clears its last-run file before each node; `stop_all` reports a container that had already exited as `ALREADY_EXITED` (ending in `STOP_ALL_DONE_WITH_NOTES`), and a failed page-cache drop as `DROP_FAILED` (ending in `STOP_ALL_INCOMPLETE`).
- `scripts/bench_v1_3_0.py`: the before/after benchmark (v1.1.0 and v1.2.0 as run, with the node list read from `BENCH_NODES`; v1.3.0 stops with `BENCH=FAIL` instead of reporting a time when a T3 or T4 request fails). Tests and timing unchanged.
- README: "What's new in v2", Step 4b (run on node0 after Step 5), Step 6 rewritten, Step 8, "Concurrency and long prompts", "Choosing the KV-cache size"; Hugging Face card updated to match.
- `results/v2_20261031/`: KV pool, benchmark and cache measurements, and the fitted step-time model.
- README Step 0b and `scripts/gb10_clock_cap_v1_0_1.service` (v1.0.1 waits for the GPU and still applies the clock lock if persistence mode fails, which v1.0.0, the version we measured with, skipped; its start at boot is untested), after node0 switched off with no log entry during the Oct 31 soak: the same SoC and Embedded Controller firmware on every node (SoC `0x02009e0f`, EC `0x03000508`) and a 2,300 MHz GPU clock cap, with the mixed evidence from other owners; unproven as a fix. Also Display Reserved Memory 0.5 GB (about 1.5 GiB more memory per node; a memory change, not a power-off fix), which needs `GPU_UTIL=0.874`.
- Validated (Nov 1): `docker/Dockerfile.sched1` reproduces the measured patched `core.py` (`675dc58a…`); `start_all_v1_0_0.sh` started our cluster (after the `GPU_UTIL` fix); `stop_all_v1_0_0.sh` cleaned up after the failed start and stopped a running server gracefully (results file). Then the fixed versions on the same cluster: `stop_all_v1_0_1.sh` reported a container that had exited on its own as `ALREADY_EXITED` and ended `STOP_ALL_DONE_WITH_NOTES` with the memory back; `start_all_v1_0_1.sh` with launcher v0.2.1 derived `GPU_UTIL` 0.874 and started the cluster at `MAX_MODEL_LEN=262144`; `bench_v1_3_0.py` completed all four tests (results file).
- `results/v2_20261031/`: also the Oct 31 soak and power-off, the Nov 1 changes, and the measurements after them.
- `results/legal_agent_eval_20261102/`: Beam served with these settings as a litigation agent on Harvey's Legal Agent Benchmark (LAB v1.3.0), against Beam in Harvey's own harness, plus a fusion step (Beam merges two agent drafts) on 6 probe and 4 held-out tasks; a matching section on the Hugging Face card.
- README: Step 0b notes a 2,200 MHz reproduction by another owner; Limitations notes Reflection's DFlash2 drafter, untested at TP=3.

**v2 settings of record:** `KV_BYTES=4831838208` (4.5 GiB), `MAX_BATCHED=4096`, `RETENTION=4096`, `LONG_PREFILL=3584`, `PREFILL_INTERVAL=9`, node0 on `beam-tp3:v1-sched1`; `GPU_UTIL=0.874` with Display Reserved Memory at 0.5 GB (0.885 at the default 2 GB).

**Corrected**
- "Stopping does not give the memory back … reboot every node before relaunching." That was after `docker rm -f`, which kills the server. A graceful stop returned the memory on every stop we recorded, Oct 28–31 (`results/v2_20261031/README.md`).
- "running without a desktop session, about 2–3 GB." Measured: 0.22–0.25 GiB per node.
- The launcher's start-up message said 65–70 minutes; that was our first-ever launch. All seven timed launches since took 17–18 minutes.

**Unchanged:** the padder, the proof, the checkpoint, the vocabulary-padding image (`docker/Dockerfile`), the smoke test, the v1 results.

## v1.0.0 (Oct 21, 2026; tagged Oct 22)

First release.
