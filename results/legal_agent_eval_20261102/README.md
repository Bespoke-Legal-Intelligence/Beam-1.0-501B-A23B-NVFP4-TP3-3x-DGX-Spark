<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Beam as a legal agent on this cluster: Harvey LAB results (Oct 24 – Nov 2, 2026)

Beam served by this recipe was run as an autonomous litigation agent on Harvey's open **Legal Agent Benchmark (LAB)**, both in Harvey's own minimal harness ("bare") and inside a self-hosted agent stack, and then with a fusion step that merges two agent drafts into one deliverable. Everything ran on the three Sparks; no case material left the machines. This is a benchmark study, not legal advice and not a production validation.

## Setup

- **Benchmark:** Harvey LAB v1.3.0, a 16-task litigation subset fixed before any run (oppositions to motions to dismiss and for summary judgment, analysis of an opponent's motion, review of proposed jury instructions, complaints, a multistate veil-piercing research memo, and others). Each task supplies 6–15 case documents and asks for a finished Word deliverable. Each deliverable is graded against Harvey's per-task rubric, 841 criteria in total.
- **Judge:** Claude Sonnet 5.5 as a single judge, one call per criterion. Nothing the agent or the fusion step saw contained rubric text. Comparisons below use the majority verdict of 3 gradings of each deliverable.
- **Bare:** Harvey's harness, unchanged: Beam at temperature 0, 32,768 tokens per call, one run.
- **Agent:** a self-hosted, open-source chat UI with native tool calling, a sandboxed terminal, document tools and offline research tools, with Beam behind it at temperature 0. Runs A and B use the same prompt and tools; B also has sub-agents available (it never used one). The agent stack is not published; this page reports what we changed and what we measured.
- **Fusion:** Beam alone (no tools) receives the task's instructions, the full text of every case document, and the finished deliverables of runs A and B, labelled only "Draft 1" and "Draft 2" in alternating order by task, and writes one final deliverable. Temperature 0; never the rubric, scores or run names.
- **Task sets:** 6 probe tasks, used while building the agent and the fusion step, and 4 held-out tasks never used for either, drawn before their runs from the 10 unused tasks with `random.Random(sha256("bespoke-ali ny16 fusion holdout 2026-11-02").hexdigest()).sample(sorted(tasks), 4)`: `draft-complaint`, `draft-federal-complaint-drafting`, `draft-motion-for-preliminary-injunction`, `identify-government-subpoena-issues`.

## Results

| Arm | Tasks | Criteria passed | Notes |
|---|---|---|---|
| Bare Beam (Harvey's harness) | 16 | **760 / 841 (0.904)** | single grading; 1 task with every criterion passed |
| Agent as first deployed | 16 | 0.614 | collapsed on 4 tasks: integration defects, below |
| Agent with the defects fixed, 131K context | 16 | 0.647 | matched bare on the 12 tasks it finished (0.863 vs 0.879); 4 overflowed the 131,072-token window |
| Agent with context compaction, 131K | 6 probe tasks, 2 runs | 0.876 | bare on the same 6: 0.913 |

At 262,144-token context (v2 serving), compaction off, majority of 3 gradings:

| Task set | Bare Beam | Agent run A | Agent run B | Better of A/B per task | **Fusion of A + B** |
|---|---|---|---|---|---|
| **All 10 tasks (512 criteria)** | **455 (0.889)** | **458 (0.895)** | **470 (0.918)** | **473 (0.924)** | **476 (0.930)** |
| of which 6 probe tasks (286) | 261 (0.913) | 255 (0.892) | 261 (0.913) | 264 (0.923) | 267 (0.934) |
| of which 4 held-out tasks (226) | 194 (0.858) | 203 (0.898) | 209 (0.925) | 209 (0.925) | 209 (0.925) |

Per task (majority of 3 gradings):

| Task | Set | Bare | A | B | Fusion |
|---|---|---|---|---|---|
| Opposition to motion to dismiss | probe | 61/63 | 60/63 | 61/63 | 61/63 |
| Opposition to motion for summary judgment | probe | 54/59 | 53/59 | 55/59 | 56/59 |
| Analysis of opponent's summary judgment motion | probe | 31/32 | 30/32 | 31/32 | 31/32 |
| Review of proposed jury instructions | probe | 27/33 | 27/33 | 24/33 | 28/33 |
| Analysis of opponent's motion to dismiss | probe | 24/34 | 24/34 | 28/34 | 29/34 |
| Multistate veil-piercing research memo | probe | 64/65 | 61/65 | 62/65 | 62/65 |
| Draft complaint (complaint + drafting notes) | held-out | 70/71 | 68/71 | 70/71 | 70/71 |
| Federal complaint (complaint + filing-issues memo) | held-out | 44/50 | 47/50 | 48/50 | 48/50 |
| Motion for preliminary injunction | held-out | 48/53 | 49/53 | 50/53 | 50/53 |
| Government subpoena issues memo | held-out | 32/52 | 39/52 | 41/52 | 41/52 |

On the probe tasks the self-hosted agent matches Beam in Harvey's own harness, and fusing two of its drafts beats both. On the held-out tasks a single agent run already beats bare Beam (203 and 209 against 194 of 226), and fusion matched the better draft on every task (209). Across all 10 tasks fusion matched or beat the better of its two drafts on every task, without seeing any score, and beat bare Beam on 6 tasks, tied 3 and lost 1: 476 of 512 (0.930) against 455 (0.889). Fusion uses more compute than one bare run (two agent runs plus a fusion pass of 22–70 minutes per deliverable); this is Beam plus a harness against Beam run once. With 10 tasks, a 6–1 split with 3 ties is suggestive rather than conclusive (exact sign test, two-sided, p ≈ 0.13); the criterion totals are the stronger evidence. Beam is the agent, the baseline and the fuser; the judge is the only other model.

## What it took to get there

1. **Beam's earlier reasoning must be sent back between tool calls.** The chat UI dropped the model's reasoning from earlier turns when it called the model through an OpenAI-compatible connection. Beam's chat template expects it, and one task fell from 26/34 to 1/34 without it.
2. **Give Beam the tools it was trained with.** A plain `bash(command)` tool, and a terminal that waits for the command to finish instead of polling it.
3. **Match the sampling.** Temperature 0 and 32,768 tokens per call, as in Harvey's harness. The agent had been deployed at 0.7 and 15,000.
4. **Don't compress the agent's own work.** At 131,072 tokens, 4 of 16 tasks overflowed. Compaction that blanked the agent's earlier drafts and reasoning let them finish but cost criteria: facts were stated in the memo but never argued. The fix was capacity: the v2 serving settings in this repo give about 3× the KV-cache pool (about 180,000 → 524,552 tokens; 599,354 at 262,144) and 2× the context per request (131,072 → 262,144). With compaction off, peak prompts stayed under about 134K tokens, 2–3 tasks ran at once, and the agent reached parity.
5. **Fuse, don't self-review.** Most of Beam's misses are points it makes in some runs and omits in others (next section). Merging two independent drafts recovers them; on the probe tasks fusion matched or beat the better of its two drafts on every task.

## How reliable these numbers are

- **Not comparable to leaderboard figures.** These are criterion pass rates on public LAB tasks. Vals AI's independent run of Beam on Harvey's 120-task held-out set ([HLAB-Vals-1.1](https://www.vals.ai/benchmarks/hlab)) reports 4.58%, a task pass rate that counts a task only if every criterion passes, with two judges and its own harness. By that all-criteria rule, bare Beam passed 1 of our 16 tasks.
- **The judge is consistent.** Grading the same deliverable three times changes the verdict on 38 of 6,144 criterion pairs (0.6%) across the 10 tasks; majority and first grading differ on 7 of 2,048 criterion verdicts.
- **Beam's runs differ.** Two runs of the same harness on the same task differ on 46 of 351 criteria (13.1%), and runs A and B differ by 6 criteria on the probe tasks. Almost all run-to-run variation is the model writing a different memo, not the judge. So no single task's difference means much on its own; the comparison rests on the totals and on how many tasks go each way. Bare Beam is one run per task, and so is each agent run.
- **Beam's misses are mostly reachable.** Across every graded Beam run on the 16 tasks, 0.958 of criteria were passed by at least one run, against 0.904 for a single bare run. Only 35 of 841 were never passed.
- **Fusion was run once per task.** Its own run-to-run spread is not measured yet.
- **The fairest comparator was not run.** Bare Beam run twice and fused the same way would separate the agent's contribution from fusion's. Bare Beam here is one run per task; the agent's two runs are also what fusion consumes.

## Limits

- One judge model, small numbers of runs, one fusion run per task, 10 of 16 tasks, and one cluster.
- The 6 probe tasks are where the agent and bare differed most and were used to build the agent and the fusion step's loop guard and retry; the 4 held-out tasks are the clean test. On the held-out tasks fusion equalled run B on every task, so the held-out gain over bare Beam is the agent's, not fusion's.
- Timings are on this cluster: 37–109 minutes per agent task with 2–3 running at once; fusion 22–70 minutes per deliverable with 2–3 at once. On one probe task the first fusion attempt looped in its reasoning and was restarted; the built-in retry (repetition penalty 1.05) produced the deliverable on 3 of the 12 fusion calls (veil-piercing memo, federal complaint, subpoena memo). Fusion produced every deliverable. In run B, the draft complaint's first agent turn ended with an empty reply ("Response payload is not completed"); the runner's follow-up recovered it (108 minutes instead of about 65).
- The agent stack and the fusion script are not part of this repo. This page documents the serving settings they ran on, the method and the measurements.
