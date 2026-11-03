---
license: apache-2.0
base_model: reflection/Beam-1.0-501B-A23B
base_model_relation: quantized
library_name: vllm
pipeline_tag: text-generation
tags:
  - nvfp4
  - tensor-parallel
  - dgx-spark
  - gb10
  - beam
  - reflection
  - multi-node
language:
  - en
---
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark (unofficial)

Reflection's [Beam-1.0-501B-A23B-NVFP4](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4), zero-padded so vLLM can split it evenly across **three** GPUs (tensor parallelism 3). It was built for three NVIDIA DGX Spark (GB10) nodes, where it runs on a stock vLLM image with one constant changed (the v2 recipe also edits one engine function on the node serving the API).

**This is a modified version of Reflection's model, not an official release.** It is not affiliated with or endorsed by Reflection AI or NVIDIA. Full instructions, the padder, the equivalence proof and the launch scripts: **https://github.com/Bespoke-Legal-Intelligence/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark**

## What's new in v2 (Nov 2, 2026)

**Same weights, same `SHA256SUMS`: nothing to re-download.** Only how the model is served changed. Measured on our three Sparks:

| | v1 | v2 |
|---|---|---|
| KV-cache pool | about 180,000 tokens (1.4 full 131K requests) | **524,552 tokens** (4 full 131K requests); a 262,144-token context also runs |
| New question about the same 100K-token document | 51.2 s (full re-read) | **0.5 s** (measured on its own) |
| One-line question sent behind a 100K-token prompt | 49.4 s | **4.6 s** |
| Requests already writing while a 100K-token prompt is read | 1.08 tokens/s each | **3.97 tokens/s** each |
| Start and stop | three terminals; reboot after stopping | **one command each**; graceful stop, no reboot |

How: the KV cache is pinned instead of estimated; sliding-window cache checkpoints keep a document cached after a long answer; a long prompt can no longer block new requests; and decode-only steps between chunks of a long prompt keep writing requests moving (one engine function patched on the node serving the API). The flags are under [Use](#use); the full recipe and measurements are on GitHub.

**Also new:** Beam on this cluster as a private legal agent on Harvey's Legal Agent Benchmark; see [Real-world use](#real-world-use-beam-as-a-private-legal-agent).

## At a glance

| | |
|---|---|
| Hardware | 3 × NVIDIA DGX Spark (GB10, 128 GB unified memory each), linked in a switchless ring of three QSFP cables |
| Serving | vLLM, tensor parallelism 3, one rank per node; pinned image plus a one-line patch (v2: plus one engine function on node0) |
| Context | 131,072 tokens (262,144 also runs with the v2 settings); KV-cache pool 524,552 tokens with the v2 settings |
| Speed | about 18 tokens/s for one request at any context length; about 32 tokens/s in total with 4 short requests at once |
| Start-up | 17–18 minutes (7 timed launches) |
| Size | about 309 GB (53 shards); each node needs the full copy and about 103 GB of it in memory |
| Weights | Reflection's NVFP4 values, unchanged; only zeros added |

## Quick start

The GitHub repo has the full, tested walkthrough with a pass/fail check after every step: **https://github.com/Bespoke-Legal-Intelligence/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark**. In outline:

1. **Network:** cable and address the three-node ring and check every link (GitHub Step 1).
2. **Download this checkpoint** to one node and verify it:

   ```bash
   DST=/srv/models/tp3/Beam-1.0-501B-A23B-NVFP4-TP3
   mkdir -p "$DST"
   docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_DISABLE_TELEMETRY=1 -v "$DST:/out" \
     python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 sh -c \
     "pip install -q --target /tmp/pkgs huggingface_hub==1.27.0 && PYTHONPATH=/tmp/pkgs python -c \
     \"from huggingface_hub import snapshot_download as s; s(repo_id='Bespoke-Intelligence/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark', local_dir='/out', max_workers=8)\""
   cd "$DST" && sha256sum SHA256SUMS && sha256sum --quiet -c SHA256SUMS && echo SUMS_MATCH
   ```

   The download runs in a throwaway container (DGX OS's system Python refuses a host `pip install`); it is GitHub Step 2's shortcut.

   `sha256sum SHA256SUMS` must print `9f57e87b492a27c3fa3cea11b0e45e1f4784069a35ee05ea2b3e0f1c100cd5dd`, the value published on GitHub, so a changed checksum list cannot pass unnoticed. Then `SUMS_MATCH`.

3. **Build the serving image** from the repo's `docker/Dockerfile` and **copy it and the checkpoint to the other two nodes** (GitHub Steps 4–5).
4. **Then build the scheduler image on node0** (`docker/Dockerfile.sched1`, GitHub Step 4b; it builds on the serving image).
5. **Launch** from node0 with `scripts/start_all_v1_0_1.sh`, or with `scripts/launch_node_v0_2_1.sh` on each node, node0 first (GitHub Step 6), then run `scripts/smoke_test_v0_1_1.sh` on node0 (Step 7). Stop with `scripts/stop_all_v1_0_1.sh` (Step 8).

## What was changed

Zeros were appended so every split divides by 3. No original value was altered.

| | Original | Padded |
|---|---|---|
| Query heads | 48 | 54 |
| Key-value heads | 8 | 9 |
| Dense feed-forward width (layers 0–2) | 20,480 | 20,736 |
| Expert feed-forward width (256 experts × 49 layers) | 2,560 | 2,688 |
| Vocabulary | 151,936 | unchanged; vLLM's padding multiple is set to 192 at serve time |

The attention output gate and its per-head bias, and the NVFP4 expert weights with their FP8 block scales, were padded to match. `config.json` changes in four fields only. `PADDING_MANIFEST.json` records the padder version, its arguments, every shard's SHA-256 and the config changes. See `NOTICE` for the Apache-2.0 modification notice.

## How it was checked

- **Source:** revision `3bd1dccf13c8f7aa3d6732c5aaa48ba107a2c086` of the NVFP4 repo; all 63 files matched the Hub's hashes before padding.
- **Byte-level:** all 151,262 tensors compared with the source. Original regions identical, padded regions zero, untouched tensors identical: 0 failures.
- **Numeric:** float64 forward passes on sampled layers. Exact unsharded, and within 5.3×10⁻¹⁶ relative error when split three ways.
- **Served on 3 × DGX Spark:** about 18 tokens/s single-stream decode; a correct tool call (one probe); exact retrieval of a planted fact from a 41,522-token prompt. In Reflection's Beam Verifier, 131 of 133 checks passed, including every structured-output and tool-calling check; the 2 failures are one request-validation check (a message with an invalid role is accepted instead of refused), which happens before the model runs.
- **Public benchmarks (one pass each, temperature 0.7, top_p 0.99, reasoning effort left at Beam's default, the maximum):**

| Benchmark | This build | Reflection's NVFP4 figure |
|---|---|---|
| GPQA Diamond | 89.9% at a 32,768-token cap; 90.9% with the 2 cut-off answers re-asked at 65,536 | 89.3% |
| IFBench, strict / lenient | 75.0% / 77.7% at 16,384; 76.7% / 79.3% with the 13 cut-off answers re-asked at 32,768 (unchanged after re-asking the last 3 at 65,536) | 79.2% |
| GSM8K (first 200) / HumanEval | 97.5% / 98.2% | — (a community 3-bit build reports 97.5% / 97.6%) |

All are consistent with the reference figures by a rule fixed before scoring. Method, intervals and caveats are in the GitHub repo.
- **Not yet done:** a full-model comparison of output probabilities against an unpadded reference.

For context, diffbot's [EXL3 build](https://huggingface.co/diffbot/Beam-1.0-501B-A23B-EXL3-3bpw-2x-RTX-PRO-6000) reports a KL divergence from the full-precision model of 0.0116 for NVFP4 as served on Blackwell. Padding adds only exact zeros, so this build is expected to match served NVFP4, not to improve on it. That is expected, not yet measured.

## Performance (one request unless stated; eager mode)

| Prompt length | Time to first token (1 request) | Decode speed (1 request) | Total output with 4 requests at once |
|---|---|---|---|
| 1,024 tokens | 0.7 s | 18.2 tokens/s | 31.9 tokens/s |
| 8,192 tokens | 3.5 s | 18.2 tokens/s | 26.5 tokens/s |
| 32,768 tokens | 12.5 s | 18.1 tokens/s | 16.4 tokens/s |
| 130,000 tokens | 71.7 s | 18.0 tokens/s | not run (the v1 KV pool held about 1.4 full-length requests) |

v1 settings, `vllm bench serve`: medians from one run on Oct 20 with random-token prompts, 256-token answers and small samples (2–16 requests per cell). The last column counts the whole run time, prompt reading included, so it falls as prompts get longer. Full tables, 99th percentiles and raw files are on GitHub.

**v2 settings, the benchmark in `scripts/bench_v1_3_0.py` (run as v1.1.0, same tests; the last row is a separate test), one run each (Oct 30–31, 100K-token prompts of synthetic text):**

| | Before (2,048-token steps, no cap, no interval) | After (v2 settings) |
|---|---|---|
| Read one 100K-token prompt | 50.9 s | 46.4 s |
| One request writing | 18.2 tokens/s | 19.2 tokens/s (within the spread of our three runs, 18.2–20.1, at different settings; 17.7–18.5 after the Nov 1 clock cap) |
| Two requests writing while a 100K-token prompt is read | 1.08 tokens/s each | 3.97 tokens/s each (that read: 65.0 s) |
| One-line question sent behind a 100K-token prompt | 49.4 s | 4.6 s |
| Same 100K document, new question after a 700-token answer | 51.2 s | 0.5 s (with `--prefix-cache-retention-interval 4096`; a separate test at 2,048-token steps, other settings off) |

## Real-world use: Beam as a private legal agent

We ran Beam on this cluster as an autonomous litigation agent on Harvey's open [Legal Agent Benchmark](https://github.com/harveyai/harvey-labs) (LAB v1.3.0): litigation tasks with 6–15 case documents each and a Word deliverable, graded against Harvey's rubric criteria by Claude Sonnet 5.5. No case material left the three Sparks.

| Criteria passed, majority of 3 gradings | 10 tasks (512 criteria) | of which 4 held-out tasks (226) |
|---|---|---|
| Beam in Harvey's own harness, one run | 455 (0.889) | 194 (0.858) |
| Beam in a self-hosted agent UI (v2 settings, 262,144 context), runs A / B | 458 / 470 | 203 / 209 |
| **Fusion: Beam merges the two agent drafts into one deliverable** | **476 (0.930)** | **209 (0.925)** |

Fusion matched or beat the better of its two drafts on all 10 tasks, without seeing any score, and beat bare Beam on 6 tasks, tied 3 and lost 1. It uses more compute than one bare run (two agent runs plus a fusion pass), and is reported that way; it saw the task, the case documents and the two drafts, never the rubric. Beam is the agent, the baseline and the fuser; the judge is the only other model. Bare Beam run twice and fused the same way, the fairest comparator, has not been run. The 4 held-out tasks were drawn at random before their runs and never used to build the agent or the fusion step; on them, a single agent run already beat bare Beam.

Getting there: as first deployed, the agent scored 0.614 against 0.904 for bare Beam on 16 tasks. The gap was integration, not Beam. Four fixes closed it: sending Beam's earlier reasoning back between tool calls (one task went from 1/34 to 26/34), a plain `bash` tool, a terminal that waits for commands, and temperature 0 with 32,768 tokens per call. Then the v2 settings gave enough context that the agent's own work never had to be compressed.

The judge flips on 0.6% of criteria when re-grading the same memo, and Beam's runs differ on 13.1%. These are criterion-level scores on public tasks, not comparable to task-level leaderboards. Method, every number and the limits: [results/legal_agent_eval_20261102](https://github.com/Bespoke-Legal-Intelligence/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark/tree/main/results/legal_agent_eval_20261102). This is a benchmark study, not legal advice.

## Use

**What you need**
- Three DGX Spark (GB10, 128 GB) nodes, each with the full checkpoint (about 309 GB on disk) and at least 109 GiB of memory available at launch.
- Tensor parallelism 3 only. The padded checkpoint has 9 key-value heads, which don't split 2, 4 or 8 ways; for those, use Reflection's original checkpoint.

**Image**
- Every node: `docker/Dockerfile` from the GitHub repo. It is `vllm/vllm-openai@sha256:c1a6be6ad08016f8cebcb449146b3d992791fe1868f9a26f3e48a51040517b77` with vLLM's `DEFAULT_VOCAB_PADDING_SIZE` set to 192.
- The node serving the API (node0), v2 only: also `docker/Dockerfile.sched1`, which edits one engine function so `--prefill-schedule-interval` works.

**Start and stop.** From node0: `scripts/start_all_v1_0_1.sh` and `scripts/stop_all_v1_0_1.sh`, with your node list and paths in `scripts/cluster.conf`. They pass all the flags below. GitHub Steps 6 and 8 cover the details.

**Server flags**

| Flag | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | `3` | one rank per node |
| `--moe-backend` | `b12x` | FP4 expert kernels for GB10 |
| `--kv-cache-dtype` | `fp8` | half the KV-cache memory |
| `--gpu-memory-utilization` | `0.885`, or `0.874` with Display Reserved Memory at 0.5 GB | 0.88 left too little KV cache on 2 of 3 launches |
| `--max-model-len`, `--max-num-seqs` | `131072`, `4` | our context and concurrency targets; 262,144 also runs |
| `--enforce-eager` | | CUDA graphs are untried |
| `--kv-cache-memory-bytes` (v2) | `4831838208` (4.5 GiB) | pins the KV-cache pool; choose yours with the GitHub README's "Memory margin" |
| `--max-num-batched-tokens` (v2) | `4096` | reads 4,096 prompt tokens per step |
| `--prefix-cache-retention-interval` (v2) | `4096` | keeps a document cached for the next question |
| `--long-prefill-token-threshold` (v2) | `3584` | a long prompt leaves room for new requests |
| `--prefill-schedule-interval` (v2) | `9` | writing requests keep moving during a long read; needs the node0 image above |

**Verify the download.** Run `sha256sum -c SHA256SUMS`. It lists the 64 files the padding step produced, including Reflection's original model card under its new name, `README_reflection.md`. This card (`README.md`) and `NOTICE` are not listed.

## Known limitations

- **Memory margin is thin.** With v1's automatic sizing the KV pool swung between launches; the v2 settings pin it at 4.5 GiB per node, which bypasses vLLM's own memory check (lowest free memory we measured: 2.07 GiB). Choose the size for your nodes with the GitHub README's "Memory margin".
- **Stop gracefully.** v1 said to reboot after stopping; that was after killing the server (`docker rm -f`). A graceful stop (`docker stop -t 120`, `docker rm`, drop the page cache) returned the memory on every stop we recorded (Oct 28–31).
- **Eager mode only;** CUDA graphs and speculative decoding are untried. During another prompt's read, writing requests get about 4 tokens/s with the v2 settings, not full speed; with this build's step costs that would need a second copy of the model.
- **Equivalence is shown analytically and on sampled blocks,** not yet by a full-model comparison against an unpadded reference.
- **One cluster.** A 24-hour soak passed at the original settings (0.88); see `results/v2_20261031/` on GitHub for the v2 soaks.
- **One unexplained hard power-off.** One node switched off with no log entry during the first v2 soak (Oct 31), as other DGX Spark owners have reported under sustained load. The GitHub README's Step 0b records what we changed afterwards, unproven as a fix (the same SoC and Embedded Controller firmware on every node and a 2,300 MHz GPU clock cap; reports from other owners are mixed), plus display memory 2 GB → 0.5 GB for about 1.5 GiB more memory per node, which needs `--gpu-memory-utilization 0.874`.

## Support

Questions and problems: open an issue on the GitHub repo. Help is best effort, with no support promised.

## Licence and citation

Apache-2.0, as the original. Please cite Reflection:

```bibtex
@misc{reflection2026introducingbeam,
  author = {{Reflection}},
  title  = {Introducing {Beam}: {Reflection}'s 501B open-weight model},
  year   = {2026},
  month  = oct,
  url    = {https://reflection.ai/blog/introducing-beam}
}
```

To cite this build, use the "Cite this repository" button on the GitHub repo.

Padded and validated by Josh Pearlman on his own three-node DGX Spark cluster. Code and documentation drafted with Claude (Anthropic).
