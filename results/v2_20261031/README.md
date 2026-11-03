<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# v2 measurements (Oct 28 – Nov 2, 2026)

Same three DGX Sparks, OS, driver, base image and padded checkpoint as v1 (see the main README, "Tested on").
node0 ran `beam-tp3:v1-sched1` (docker/Dockerfile.sched1) for the "after" runs; node1 and node2 ran `beam-tp3:v1`.
All runs used `scripts/launch_node_v0_2_1.sh`'s settings as listed (the Oct 28–31 runs used a private v0.2.0, identical
on the success path; the Nov 1 runs with `start_all_v1_0_1.sh` used v0.2.1 as published). Figures below are from the tool output; node names are generalised.

## KV-cache pool

| Pinned KV per node (`KV_BYTES`) | `MAX_BATCHED` | Pool reported by vLLM | Full 131,072-token requests (pool ÷ 131,072) |
|---|---|---|---|
| automatic (v1, Oct 21) | 2048 (default) | 178,253 / 180,048 tokens | 1.36–1.37 |
| 3.0 GiB (Oct 28) | 2048 | 396,153 tokens | 3.02 |
| 4.0 GiB (Oct 28–30) | 2048 | 528,205 tokens | 4.03 |
| 4.0 GiB (Oct 30) | 4096 | 466,273 tokens | 3.56 |
| **4.5 GiB (Oct 31, v2 settings)** | **4096** | **524,552 tokens** | **4.00** |

The step size matters because 43 of Beam's 52 layers use a 512-token sliding window, and vLLM reserves window-plus-step
blocks for each running request out of the same pinned pool (our reading of the drop; the 466,273 figure is measured).

## Benchmark (scripts/bench_v1_3_0.py; the runs used v1.1.0, identical tests)

| Run | Settings | T1 lone read, 100,481 tokens | T2 single chat | T3 chats while a read arrives (per chat before / during; read time) | T4 short question behind a read (short / long) | Lowest free memory node0 / node1 / node2 |
|---|---|---|---|---|---|---|
| `before`, Oct 30 19:51 | KV 4.0 GiB, step 2048, retention 4096, no cap, no interval | 50.9 s (1,974 tok/s) | 18.2 tok/s | 13.1 / **1.08** tok/s; 51.9 s | **49.4 s** / 51.4 s | 3.84 / 5.28 / 2.18 GiB |
| `after`, Oct 30 23:31 | KV 4.0 GiB, step 4096, cap 3584, interval 5 | 46.3 s (2,170 tok/s) | 20.1 tok/s | 12.8 / 2.63 tok/s; 55.8 s | 4.6 s / 45.8 s | 2.39 / 4.30 / 3.86 GiB |
| `after_n9`, Oct 31 00:12 | **KV 4.5 GiB, step 4096, cap 3584, interval 9** | **46.4 s (2,165 tok/s)** | **19.2 tok/s** | 12.9 / **3.97** tok/s; 65.0 s | **4.6 s** / 44.8 s | 2.07 / 4.33 / 3.83 GiB |

## Same document, new question (prefix cache)

A ~100,400-token document; request 1 writes a 700-token answer; requests 2 and 3 ask new questions about the same
document (`max_tokens` 1, so time ≈ time to first token). Both runs: KV pinned at 4.0 GiB, 2,048-token steps, no cap,
no interval, `beam-tp3:v1` on all nodes; only the retention setting differs.

| Oct 30 | Request 1 (read + 700 tokens) | Request 2 (new question) | Request 3 (another question) |
|---|---|---|---|
| default (`--prefix-cache-retention-interval` 0), 14:58 | 99.8 s | **51.2 s** (full re-read) | 0.5 s |
| `RETENTION=4096`, 19:28 | 89.9 s | **0.5 s** | 0.4 s |

One document, one run each. With retention 4096, a question that branches mid-document can re-read up to about
4,096 tokens (about 2 s, calculated).

With request 1 limited to a single token (so the answer never moved past the 512-token window), request 2 already hit
the cache at the default setting (0.5 s): the miss only appears once an answer has run past the window.

## Long-prompt cap

In this build `--long-prefill-token-threshold` is applied only when more than one request is running or waiting
(`vllm/v1/core/sched/scheduler.py`: `long_prefill_token_threshold = … if num_eligible_reqs > 1 else 0`), so T1, a
prompt read on its own, used the full 4,096-token steps.

## Starts and graceful stops

| Oct 30 launch | Settings | Launch → `/health` 200 |
|---|---|---|
| 15:23 | KV 4.0 GiB, 2,048-token steps, retention 4096 | 17 min (15:23 → 15:40) |
| 22:32 | KV 4.0 GiB, 4,096, cap 3584, interval 5 | 17 min (22:32 → 22:49) |
| 23:53 | KV 4.5 GiB, 4,096, cap 3584, interval 9 | 18 min (23:53 → 00:11) |

| Oct 30 stop | Method | Exit code node0 / node1 / node2 | `MemAvailable` after dropping the page cache (GiB) |
|---|---|---|---|
| 15:19 | `docker stop -t 120`, `docker rm`, drop caches, on each node | 0 / 0 / 0 | 116 / 116 / 116 |
| 22:30 | same | 0 / 0 / 0 | 116 / 116 / 116 |
| 23:45 | the stop-all script (same steps) | 0 / 0 / 0 | 116 / 116 / 117 |

Earlier graceful stops on Oct 28–29 (at least five in a row on Oct 28) also returned the memory with no reboot; their
per-node values are in our session notes, not reproduced here. A desktop session was measured on Oct 28 at
0.22–0.25 GiB per node.

## Step-time model (fitted, used for the interval table in the main README)

From T1 at 2048 tokens per step (≈1.0 s per step) and 4096 (≈1.85 s): one prompt-reading step ≈ 0.15 s + 0.41 ms per
token. A decode-only step with two requests writing ≈ 0.078 s (from T3's "before" rate of ~12.9 tok/s per chat).
It is a two-point fit, and its write-only step comes from the same runs. It agrees with T3 at interval 5 (calculated
2.55 tok/s and 55 s; measured 2.63 and 55.8) and at interval 9 (calculated ~4.0 tok/s and ~63 s; measured 3.97 and 65.0).

## Soak, Oct 31: stopped by a power-off on node0

Started 11:05 (local time) at the v2 settings. Each cycle: four ~120K-token prompts at once, each of which must return
a code hidden in its own document; then the four-test benchmark; then a container and GPU-error (Xid) check on all
three nodes.

| Cycle | Burst answers correct | T1 lone read | T3 per chat during a read | T4 short behind a read | Lowest free memory node0 / node1 / node2 | Xid |
|---|---|---|---|---|---|---|
| 1 (11:12) | yes | 44.5 s | 3.98 tok/s | 4.5 s | 2.22 / 2.66 / 3.81 GiB | 0 / 0 / 0 |
| 2 (11:20) | yes | 44.9 s | 3.88 tok/s | 4.5 s | 2.21 / 4.12 / 3.81 GiB | 0 / 0 / 0 |

During cycle 3, with three requests running, node0 switched off. Its last server log line was at 11:23:20 and its
system journal ends at 11:23:06 with routine entries; there was no kernel, GPU, watchdog or thermal message and no
saved crash record. At 11:23:32 node1 and node2 logged NCCL errors on both of their links to node0. node0's
system-activity samples to 11:20 show about 2.2 GiB available, no swapping out (some swapping in, 1–2 pages/s) and a load of about 2.7. It restarted by itself about four hours later; no one pressed the power button or unplugged it. The two other nodes stopped gracefully and returned their memory. Cause not established;
see the main README, Step 0b.

## Nov 1: firmware, display memory and clock-cap changes (Step 0b), and re-validation

Kernel 6.17.0-1026-nvidia, driver 595.71.05, images and checkpoint unchanged on all three nodes.

| | node0 | node1 | node2 |
|---|---|---|---|
| SoC firmware before → after | `0x0200980f` → `0x02009e0f` | `0x0200980f` → `0x02009e0f` | `0x02009b0b` → `0x02009e0f` |
| Embedded Controller before → after | `0x03000302` → `0x03000508` | `0x03000302` → `0x03000508` | `0x03000508` (unchanged) |
| USB-C power-delivery firmware (not changed) | `0x00000001` | `0x00000516` | `0x00000001` |
| Firmware reboot (silent) | not timed | about 7 min | about 6 min |
| `MemTotal` before → after Display Reserved Memory 0.5 GB | 127,600,524 → 129,170,308 kB | 127,600,748 → 129,170,308 kB | 127,600,524 → 129,170,308 kB |
| 25-s matrix-multiply test with the 2,300 MHz cap (unit v1.0.0): clock, power, GPU °C | 2,210–2,268 MHz, 90.6–92.1 W, 58→69 | 2,236–2,275 MHz, 92.1–93.3 W, 54→66 | 2,268–2,275 MHz, 86.7–89.5 W, 56→67 |
| Multiplications completed in that test | 3,221 | 3,225 | 3,233 |

After each firmware reboot: all four ring ports up at 200 Gb/s, all four RDMA devices active and no Xid on all three
nodes; full-size (9000-byte) pings to both neighbours with 0% loss and no failed services, checked on node0 and node1. node0 and node2 each had an old USB-C
power-delivery update file (`fwupd-dd1a238a-….cap`) waiting in `/boot/efi`; we set it aside before rebooting.

**Public scripts and image, as run on Nov 1:**
- `docker/Dockerfile.sched1` built on node0: both hash checks passed and the patched `core.py` is `675dc58ac841fc3a…`,
  identical to the image the Oct 28–31 measurements used.
- `stop_all_v1_0_0.sh` (the version before `scripts/stop_all_v1_0_1.sh`), run against the three containers left by the failed start below (the server had
  already exited, `exit_code=1`): `STOPPED_AND_REMOVED` on all three, `MEM_BACK` 116 / 117 / 117 GiB, `STOP_ALL_DONE`.
  A graceful stop of a running server with this script (Nov 1, before the 262K restart): `STOPPED_AND_REMOVED exit_code=0`
  on all three nodes (19 s, 25 s and 30 s), `MEM_BACK` 117 / 117 / 117 GiB with `GPU_PROCS=0`, `STOP_ALL_DONE` about 2 minutes
  after it started (including typing three sudo passwords).
- `start_all_v1_0_0.sh` (the version before `scripts/start_all_v1_0_1.sh`): the first start after the memory change failed in vLLM's start-up check
  (`Free memory on device cuda:0 (108.87/123.19 GiB) on startup is less than desired GPU memory utilization (0.885,
  109.02 GiB)`); with `GPU_UTIL=0.874`: `PREFLIGHT_OK` on all three, `START_ALL_READY` after 18 minutes
  (21:08:58 → 21:27:16), KV pool 524,552 tokens, all five v2 settings confirmed in the engine log.
- The fixed scripts, Nov 1 22:01, at `MAX_MODEL_LEN=262144`. `stop_all_v1_0_1.sh`: node0's server had exited on its own
  (`exit_code=1`) after waiting about 20 minutes for node1, whose `sudo` prompt we had left unanswered; the script reported
  `ALREADY_EXITED`, `NO_CONTAINER` on the other two, `MEM_BACK` 117 / 116 / 117 GiB, `STOP_ALL_DONE_WITH_NOTES`.
  `start_all_v1_0_1.sh` with `GPU_UTIL` unset: `using 0.874 (107.7 GiB of the largest MemTotal, 129170308 kB)`, `PREFLIGHT_OK`,
  `STARTED` and `CONTAINER_RUNNING` on all three ranks, `START_ALL_READY` after 18 minutes (22:02:12 → 22:20:33), vLLM's
  `GPU KV cache size: 599,354 tokens, Maximum concurrency for 262,144 tokens per request: 2.29x` (same 4.5 GiB; README,
  Longer requests). `bench_v1_3_0.py`: table below.

## Benchmark after the Nov 1 changes

`v2pub_cap_1101`: `bench_v1_2_0.py` (the version before `scripts/bench_v1_3_0.py`; same tests), Nov 1 21:29 (cluster started 21:08 with `start_all_v1_0_0.sh`, above), v2 settings with `GPU_UTIL=0.874`, after the firmware, display-memory
and clock-cap changes; one run. Oct 31's `after_n9` run (same settings, before the changes) for comparison.

| Run | T1 lone read | T2 single chat | T3 per chat before / during; read time | T4 short / long | Lowest free memory node0 / node1 / node2 |
|---|---|---|---|---|---|
| `after_n9`, Oct 31 | 46.4 s (100,481 tokens) | 19.2 tok/s | 12.9 / 3.97 tok/s; 65.0 s | 4.6 s / 44.8 s | 2.07 / 4.33 / 3.83 GiB |
| `v2pub_cap_1101`, Nov 1 | 47.6 s (100,489 tokens; 2,110 tok/s) | 18.4 tok/s | 12.7 / 3.92 tok/s; 66.1 s | 4.8 s / 46.0 s | 2.96 / 4.78 / 4.33 GiB |
| `v101_262k`, Nov 1 22:21 (`bench_v1_3_0.py`, `MAX_MODEL_LEN=262144`) | 47.6 s (100,489 tokens; 2,112 tok/s) | 18.5 tok/s | 12.7 / 3.93 tok/s; 65.9 s | 4.7 s / 45.6 s | 3.09 / 5.00 / 4.34 GiB |

- Every test is within 1–5% of the Oct 31 run, inside the spread we saw between earlier runs (T2: 18.2–20.1 tok/s).
  One run each, so we cannot separate the clock cap's cost from that spread.
- Lowest free memory rose by 0.89 / 0.45 / 0.50 GiB, less than the 1.5 GiB the display change added to each node's
  total. We have not explained the difference (one run each).
- The engine log showed `LOCAL PATCH sched1 active: prefill_schedule_interval=9`.

## Soak after the Nov 1 changes

Our soak script (`beam_soak_v1_0_0.sh`, not published) with `HOURS=2`, Nov 1 22:34:44 to Nov 2 00:41:46 (2 h 7 min), after the firmware, display-memory and
clock-cap changes, at `MAX_MODEL_LEN=262144` and `GPU_UTIL=0.874`, started with `start_all_v1_0_1.sh`. Each cycle: four ~121K-token
prompts sent at the same instant (each with a short answer), then the four benchmark tests, then health checks on all three nodes.

| | Result |
|---|---|
| Cycles | 16 of 16 completed, `burst_failures=0` (`SOAK_DONE ... cycles=16 burst_failures=0`) |
| Bursts | every cycle `burst_answers_ok=True`; prompts are read one after another (first tokens about 67 / 130 / 189 / 248 s in), at most 3 running and at most 2 waiting (maxima at different moments), KV cache at most 31.8 % |
| Benchmark, last cycle | T1 45.7 s, T2 17.7 tok/s, T3 12.9 / 3.88 tok/s with a 66.2 s read, T4 4.7 s / 45.7 s |
| Lowest free memory, last cycle | 2.71 / 4.58 / 4.21 GiB (node0 / node1 / node2) |
| Health | containers running on all three nodes and zero Xid faults at every cycle |
| Temperatures | in the live view (sampled every 30 s for the first minutes, not the whole run), the hottest ACPI zone on node0 reached 93 °C during a burst, node1 and node2 up to 84 °C; the zones' critical trip point is 104 °C |
| Power-offs | none |

This soak ran about 7 times longer than the Oct 31 soak lasted before node0 switched off (18 minutes), without a power-off.
That is consistent with the Nov 1 changes helping, but one 2-hour run does not show it: the Oct 31 power-off came after
18 minutes, and owners describe the fault as intermittent.

