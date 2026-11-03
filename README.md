<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Beam-1.0 501B on three DGX Sparks at TP=3

Run Reflection's [Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4) (NVFP4) across three NVIDIA DGX Spark (GB10) nodes with tensor parallelism 3, on a stock vLLM image with one constant changed (v2 adds one edited engine function on node0; Step 4b).

Beam has 8 key-value heads, and three other widths that don't divide by 3, so it cannot start at TP=3 as shipped. This repo pads the checkpoint offline with zeros so every split is even, shows analytically (under stated kernel assumptions) that the padded model computes the same function, and verifies the padded files byte by byte against the original. Then it walks through serving it on a switchless three-node ring, with a pass/fail check after every step. The only change to vLLM in v1 is one constant, patched into the pinned upstream image; v2 also edits one engine function on node0 (Step 4b).

**Measured on our cluster:**
- about 18 tokens/s single-stream decode;
- a correct tool call (one probe);
- exact retrieval of a planted fact from a 41,522-token prompt;
- 131 of 133 checks passed in Reflection's own [Beam Verifier](https://github.com/reflection-oss/beam-verifier), including every structured-output and tool-calling check (the 2 failures are one request-validation check; see [Results](#results));
- GPQA Diamond and IFBench scores consistent with Reflection's published NVFP4 figures ([Results](#results));
- about 3 hours from starting the download to the first token.

To the best of our knowledge, this is the first public recipe for running Reflection's own NVFP4 (4-bit) weights on DGX Spark, the first Beam setup across several Sparks, and the first at TP=3. That rests on a search of Hugging Face, GitHub, vendor documentation and the web as of Oct 21, 2026, which was not exhaustive. A different approach, a single-Spark build that re-quantizes the experts to about 1.5 bits and runs on its own engine, was published on Oct 20, 2026 (see [Related work](#related-work-and-other-builds)). TP=3 on Sparks itself is established for other models.

> Not affiliated with or endorsed by NVIDIA or Reflection AI. NVIDIA, DGX Spark, GB10 and ConnectX are trademarks of NVIDIA Corporation; Beam is Reflection AI's model. Names are used only to describe what this was run on.

## What's new in v2 (Nov 2, 2026)

Same padded checkpoint (no re-download; `SHA256SUMS` unchanged), same base image. What changed is how it is run:

| Measured on our cluster (Oct 28–31, same nodes; one run each) | Before (2,048-token steps, no cap, no interval; pool row: v1's automatic sizing) | After (v2 settings) |
|---|---|---|
| KV-cache pool | 178,253–180,048 tokens on Oct 21 (about 1.4 full 131K requests) | **524,552 tokens (4.0 full 131K requests)** |
| Same document, new question after a long answer | full re-read (51.2 s for 100K tokens) | **0.5 s** with `RETENTION=4096` (measured at 2,048-token steps with the other v2 settings off; a branch can re-read up to about 4K tokens, about 2 s, calculated) |
| A one-line question sent 2 s after a 100K-token prompt | waited for the whole read (49.4 s) | **4.6 s** |
| Two requests already writing while a 100K-token prompt is read | 1.08 tokens/s each | **3.97 tokens/s each** (that read: 65.0 s, against 46.4 s for the same read with nothing else running) |
| Reading one 100K-token prompt | 50.9 s | **46.4 s** |
| One request writing | 18.2 tokens/s | 19.2 tokens/s (unchanged within the spread of our three runs, 18.2–20.1, at different settings; 17.7–18.5 after the Nov 1 clock cap) |
| Stopping | `docker rm -f`, then reboot every node | **graceful stop returned the memory on every stop we recorded (Oct 28–31); no reboot** |
| Start / stop | three terminals | **one command each from node0** (scripts: Steps 6 and 8) |

How:
- **The KV cache is pinned** (`--kv-cache-memory-bytes`) instead of estimated, so its size no longer swings between launches.
- **Sliding-window cache checkpoints** every 4,096 tokens (`--prefix-cache-retention-interval`), so a document stays cached after a long answer.
- **4,096-token steps** (`--max-num-batched-tokens`) with a **cap on any one long prompt** while other requests are present (`--long-prefill-token-threshold 3584`), so a new request can start in the next step instead of waiting for the whole read.
- **Decode-only steps between chunks of a long prompt** (`--prefill-schedule-interval 9`). vLLM already has this mechanism, but in this build only its data-parallel engine turns it on; `docker/Dockerfile.sched1` edits one function so the single-replica engine honours the setting too.
- **Graceful stop and one-command start/stop** (`scripts/start_all_v1_0_1.sh`, `scripts/stop_all_v1_0_1.sh`).
- **After a power-off (Nov 1, unproven as a fix):** the same SoC and Embedded Controller firmware on every node and a GPU clock cap, after one node switched off under load during a soak; while in the firmware setup we also cut display memory from 2 GB to 0.5 GB (about 1.5 GiB more memory per node; unrelated to the power-off). See [Step 0b](#step-0b-firmware-display-memory-and-a-gpu-clock-cap-what-we-changed-on-nov-1).

**Beam as a legal agent on this cluster (Oct 24 – Nov 2):** on 16 litigation tasks of Harvey's Legal Agent Benchmark, Beam in Harvey's own harness passed 760 of 841 rubric criteria (0.904). A self-hosted agent served with the v2 settings at a 262,144-token context matched it on 6 probe tasks (261 of 286 criteria for both, majority of 3 gradings; a second agent run, 255), and fusing the two agent drafts into one deliverable scored 267 of 286. On 4 held-out tasks drawn at random: fusion 209 of 226, bare Beam 194; on all 10 tasks: fusion 476 of 512 (0.930), bare Beam 455 (0.889). Method, every number and the limits: [results/legal_agent_eval_20261102/README.md](results/legal_agent_eval_20261102/README.md).

Details: [Concurrency and long prompts](#concurrency-and-long-prompts), [Memory margin](#memory-margin), [v2 results](results/v2_20261031/README.md), [legal agent results](results/legal_agent_eval_20261102/README.md), [CHANGELOG](CHANGELOG.md). Upgrading from v1:
1. `git pull` this repo on all three nodes (the workers run `scripts/launch_node_v0_2_1.sh` from their own copy).
2. Create `scripts/cluster.conf` on node0 and choose `KV_BYTES` ([Choosing the KV-cache size](#choosing-the-kv-cache-size)). Leave `GPU_UTIL` unset: `start_all` derives it from the memory total, which matters if you lower Display Reserved Memory ([Step 0b](#step-0b-firmware-display-memory-and-a-gpu-clock-cap-what-we-changed-on-nov-1)).
3. Stop the v1 server gracefully with `scripts/stop_all_v1_0_1.sh` (Step 8), not `docker rm -f`. Reboot a node only if the script reports `MEM_HELD` there, or if the server was already killed with `docker rm -f`.
4. Build the scheduler image on node0 (Step 4b).
5. Start (Step 6).

## Contents

1. [What's new in v2](#whats-new-in-v2-nov-2-2026)
2. [Tested on](#tested-on)
3. [What you need](#what-you-need)
4. [Step 0: prepare every node](#step-0-prepare-every-node) (and [Step 0b: firmware, display memory and a GPU clock cap](#step-0b-firmware-display-memory-and-a-gpu-clock-cap-what-we-changed-on-nov-1))
5. [Step 1: ring network](#step-1-ring-network)
6. [Step 2: download and verify the checkpoint](#step-2-download-and-verify-the-checkpoint)
7. [Step 3: pad, verify and test](#step-3-pad-verify-and-test)
8. [Step 4: build the serving image](#step-4-build-the-serving-image) (and [Step 4b: build the scheduler image](#step-4b-build-the-scheduler-image-node0-only-v2))
9. [Step 5: copy to the other nodes](#step-5-copy-to-the-other-nodes)
10. [Step 6: launch](#step-6-launch)
11. [Step 7: smoke tests](#step-7-smoke-tests)
12. [Step 8: stop and restart](#step-8-stop-and-restart)
13. [Results](#results)
14. [Concurrency and long prompts](#concurrency-and-long-prompts)
15. [Why it is built this way](#why-it-is-built-this-way)
16. [Troubleshooting](#troubleshooting)
17. [Limitations](#limitations)
18. [Related work and other builds](#related-work-and-other-builds)
19. [Credits, citation and licence](#credits-citation-and-licence)

## Tested on

All three nodes ran the same kernel, driver and software; their SoC and Embedded Controller firmware differed until Nov 1 (below). Recorded Oct 20, 2026, from the hosts and the running containers; nothing was changed before the last measurement on Oct 21.

| Component | Version |
|---|---|
| Hardware | 3 × DGX Spark (GB10, 128 GB unified memory, ConnectX-7 with two 200 Gb/s QSFP ports) |
| OS | DGX OS 7.5.0, kernel 6.17.0-1026-nvidia, aarch64 |
| NVIDIA driver | 595.71.05 |
| ConnectX-7 firmware | 28.45.4028 |
| Host test tools | NCCL 2.30.7 (`libnccl2 2.30.7-1+cuda13.3`), Open MPI 4.1.6 (`4.1.6-7ubuntu2`), perftest 24.01 (`24.01.0+0.38-1build2`, `ib_write_bw` 6.20), nccl-tests commit `358ca293`, built against them |
| Base image | `vllm/vllm-openai@sha256:c1a6be6ad08016f8cebcb449146b3d992791fe1868f9a26f3e48a51040517b77` (arm64, Beam day-0 support build) |
| Inside the image | vLLM 0.1.dev23042+g8e0922e88, PyTorch 2.13.0+cu130, FlashInfer 0.7.2.dev20261014 (includes the b12x MoE kernels), NCCL 2.30.7 (`nvidia-nccl-cu13`), transformers 5.19.0 |
| Model | `reflection/Beam-1.0-501B-A23B-NVFP4`, revision `3bd1dccf13c8f7aa3d6732c5aaa48ba107a2c086`, Apache-2.0; 63 files, 293,654,927,078 bytes |

Other driver, kernel or vLLM versions are untested. The v2 measurements (Oct 28–31) used the same hardware and software, plus the scheduler image of Step 4b on node0. One node's `/etc/dgx-release` reads 7.6.0: it shipped with DGX OS 7.6.0 and was brought to the kernel and driver above before any measurement.

**Changed on Nov 1 (Step 0b), on all three nodes:** SoC firmware `0x02009e0f`, Embedded Controller `0x03000508` (before: two nodes on `0x0200980f` / `0x03000302`, one on `0x02009b0b` / `0x03000508`), Display Reserved Memory 0.5 GB (was 2 GB) with `GPU_UTIL=0.874`, GPU clock capped at 2,300 MHz. Kernel, driver, ConnectX-7 firmware, USB-C power-delivery firmware, images and checkpoint unchanged. Measurements after these changes are labelled as such.

## What you need

- **Three DGX Sparks and three QSFP cables** (200 Gb/s QSFP112 direct-attach) for a switchless ring.
- **Docker and the NVIDIA Container Toolkit** on every node; your user in the `docker` group and able to use `sudo` (netplan, dropping the page cache, creating `/srv/models`).
- **Passwordless SSH** from node0 to the other two nodes (for the NCCL test), and from node0 and node2 to the build node (for copying).
- **Disk:**
  - about 650 GB free on the build node (original, padded copy and images);
  - about 360 GB on each other node.
  - Models live under `/srv/models`, owned by your user: `sudo install -d -o "$USER" -g "$USER" /srv/models`.
- **Memory at launch:** at least 109 GiB `MemAvailable` on every node, after dropping the page cache. Stop every other GPU workload first. GB10 memory is shared between CPU and GPU, so anything else resident competes with the weights.
- **A build node with a fast internet link.** We used node1. One of our nodes had negotiated a 100 Mb/s management link; downloading on it would have taken over 6 hours.

Node roles in this guide:

| Node | TP rank | Role |
|---|---|---|
| node0 | 0 | Serves the API; rendezvous master |
| node1 | 1 | Worker; build node (download and padding) |
| node2 | 2 | Worker |

## Step 0: prepare every node

Run every command in this guide from the repo root unless a step says otherwise. Each step re-declares the variables it needs, so a new shell is fine.

On **every** node:

```bash
git clone https://github.com/Bespoke-Legal-Intelligence/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark && cd Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark
sudo apt install -y perftest=24.01.0+0.38-1build2 openmpi-bin=4.1.6-7ubuntu2 libopenmpi-dev=4.1.6-7ubuntu2 \
  libnccl2=2.30.7-1+cuda13.3 libnccl-dev=2.30.7-1+cuda13.3   # the versions we ran; NCCL from NVIDIA's CUDA apt repository
sudo install -d -o "$USER" -g "$USER" /srv/models /opt/nccl-tests
git clone https://github.com/NVIDIA/nccl-tests /opt/nccl-tests && git -C /opt/nccl-tests checkout 358ca293f16b1248fead83353520d5768fee07c8
make -C /opt/nccl-tests MPI=1 MPI_HOME=/usr/lib/aarch64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda
```

- These are the exact package versions and nccl-tests commit we ran (see [Tested on](#tested-on)); other versions are untested. If apt no longer offers a pinned version, install the nearest one and expect to re-check Step 1.
- nccl-tests is needed at the same path on all three nodes, because `mpirun` starts it on each.

**SSH.** From node0, `ssh-copy-id <user>@<node1-address>` and `ssh-copy-id <user>@<node2-address>` (for the NCCL test). From node0 and node2, also copy a key to node1, the build node (for Step 5).

**Pass:** `ls /opt/nccl-tests/build/all_gather_perf` exists on every node, and `ssh -o BatchMode=yes <user>@<address> true` succeeds from node0 to both workers.

### Step 0b: firmware, display memory and a GPU clock cap (what we changed on Nov 1)

**Why.** On Oct 31, during a soak at the v2 settings, node0 switched itself off about 18 minutes in, with three requests running. It left nothing behind: no kernel error, no GPU error, no thermal message and no crash record; the other two nodes only saw their links to it drop. Other DGX Spark owners report the same log-less power-off under sustained GPU load, and what they report about fixes is mixed:
- One owner's unit, starting from exactly our two older nodes' firmware, stopped powering off after a firmware update, but still failed NVIDIA's field diagnostic and an RMA was approved ([NVIDIA forum](https://forums.developer.nvidia.com/t/dgx-spark-gb10-partnerdiag-powerstress-reproducibly-hard-powers-off-the-box-acpitz-88-97-8-c-in-5s-all-other-field-tests-pass/377365)).
- Another owner's ten power-offs in one day happened *on* the Embedded Controller version we moved to (`0x03000508`); they attribute them to a faulty thermal sensor (RMA requested) ([NVIDIA forum](https://forums.developer.nvidia.com/t/dgx-spark-log-less-hard-power-off-under-load-thermal-sensor-fault-firmware-up-to-date-matches-rmad-case-377365/385262)).
- An eight-unit write-up reports that all its units were already on the latest firmware and still powered off; spacing out their taped-together power bricks reduced the power-offs but did not end them, and locking the GPU clock is the change after which they saw none ([write-up](https://petronellatech.com/blog/dgx-spark-shutdown-under-load-our-8-unit-gb10-diagnosis/)).

On Nov 1 we made the changes below on all three nodes, one node at a time. **We have not shown that any of them prevents the power-off;** the soak after them is in [results/v2_20261031/README.md](results/v2_20261031/README.md). Parts 1 and 3 are aimed at the power-off; part 2 is only a memory gain we took while in the firmware setup. If power-offs continue, run NVIDIA's field diagnostic and contact NVIDIA support: the owners above with repeated power-offs were granted or requested RMAs.

**1. Firmware: the same SoC and Embedded Controller versions on every node.** Firmware is the one change here you cannot undo from this repo. Keep the node on reliable power. NVIDIA's own [update guide](https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html) installs the newest firmware (`fwupdmgr upgrade` or the DGX Dashboard), which would include Embedded Controller `0x03000d07`; we named versions instead, for the reason below.
- Ours were mixed: two nodes on Embedded Controller `0x03000302` and SoC `0x0200980f`, one on `0x03000508` and `0x02009b0b`. We put all three on SoC `0x02009e0f` (October 2026; it adds the setting in part 2) and Embedded Controller `0x03000508`. We did not touch the USB-C power-delivery firmware (still `0x00000516` on one node, `0x00000001` on the other two) or the ConnectX-7 firmware.
- **Why not the newest Embedded Controller, `0x03000d07`:** one owner reports, from one unit under a 20-core CPU load, that it ignores the host's fan requests and ran at 92 °C against 73 °C on `0x03000508`; another poster in the same thread says the newer version raises the minimum fan speed. It is unconfirmed by NVIDIA, so this is a judgement call, not a settled fact ([NVIDIA forum](https://forums.developer.nvidia.com/t/new-dgx-spark-fans-barely-speed-up-at-high-temps-and-cant-be-adjusted-critical-please-open-fan-control/386805?page=2)). The same owner's best guess is that the October SoC firmware and the older EC do not line up on one EC mailbox (it timed out for them); the October bundle pairs SoC `0x02009e0f` with EC `0x03000d07` ([kelchm/home-lab](https://github.com/kelchm/home-lab/issues/1030)). We saw no problem with our pair, though we did not test fan control.
- **If you are already on `0x03000d07`,** staying there is reasonable. Going back to `0x03000508` is a downgrade (`fwupdmgr` needs `--allow-older`), which we have not done.
- `fwupdmgr update` always takes the newest versions, so we named each one. Use the version string exactly as `fwupdmgr get-updates` prints it; on our nodes it printed, and accepted, the hex form:

```bash
sudo fwupdmgr refresh --force
sudo fwupdmgr get-updates                     # each device's ID, current version and the versions on offer
sudo fwupdmgr install <Embedded Controller device ID> 0x03000508 -y --no-reboot-check   # skip if already 0x03000508
sudo fwupdmgr install <SoC "UEFI Device Firmware" device ID> 0x02009e0f -y --no-reboot-check
sudo find /boot/efi -iname '*.cap' -printf '%TY-%Tm-%Td %TH:%TM %p\n'   # the update files waiting for the reboot
sudo reboot
```

- **Before rebooting, read the list of waiting files.** On two of our nodes an old power-delivery update (`fwupd-dd1a238a-….cap`; on one of them dated May, in `/boot/efi/EFI/UpdateCapsule/`) was also waiting, never applied; it could have been applied together with ours. To change one thing at a time we copied it out of `/boot/efi` to our home folder and then removed it from `/boot/efi`. The files you just staged (one or two, depending on which versions the node already had; the UX capsule below may also be new) should carry today's date in the listing; an older-dated `.cap` was left by an earlier update. A `fwupd-3b8c8162-….cap` file, if present, is fwupd's update-progress screen (the UEFI "UX capsule"), not firmware; leave it. If you are unsure what a file is, leave all of them: the cost is that an older pending update may be applied in the same reboot.
- **The firmware reboot is silent for several minutes** (6–7 on the two we timed). Do not remove power.
- **Pass:** after the reboot `sudo fwupdmgr get-devices` shows the new versions, and Step 1's link checks pass.
- **Later updates:** a later `fwupdmgr update` or DGX OS update can move the Embedded Controller to `0x03000d07`. Check `fwupdmgr get-devices` after any update.
- If `fwupdmgr` fails with `libfwupd version … does not match daemon …`, the `fwupd` and `libfwupd` packages are out of step. On one of our nodes, installing Ubuntu's current `fwupd` (2.0.20) fixed it and changed nothing else (check with `apt-get -s install fwupd` first).

**2. Display memory: 2 GB → 0.5 GB (memory gain, not a power-off fix).** This needs a monitor and keyboard on the node: don't do it over SSH on a node you cannot reach in person. With SoC `0x02009e0f` the firmware setup screen (hold Esc during boot; `sudo systemctl reboot --firmware-setup` should also work, but we have not confirmed it on a Spark) has a **Display Reserved Memory** setting under **Advanced**, default 2 GB. It may be missing on partner GB10 systems ([NVIDIA forum](https://forums.developer.nvidia.com/t/reclaim-2gib-of-ram-on-headless-sparks/384621)).
- We chose 0.5 GB on each node and saved with **Save & Exit → Save Changes and Exit**. `MemTotal` rose by about 1.5 GiB (1,532–1,533 MiB) on every node (from 127,600,524–127,600,748 kB to 129,170,308 kB).
- **`GPU_UTIL` must then drop to 0.874.** `scripts/start_all_v1_0_1.sh` does this for you when `cluster.conf` does not set `GPU_UTIL`; if you launch by hand, set it yourself. vLLM claims `GPU_UTIL` × the memory total at start-up, so the larger total raised its claim from 107.70 GiB (calculated) to 109.02 GiB, and our first start afterwards failed with `Free memory on device cuda:0 (108.87/123.19 GiB) on startup is less than desired GPU memory utilization (0.885, 109.02 GiB)`. With 0.874 (107.67 GiB) it started normally. In general, `GPU_UTIL` ≈ 107.7 ÷ (`MemTotal` in kB ÷ 1,048,576), rounded down to three decimals; vLLM's error message prints the total it sees (the figure after the slash). The KV cache is pinned separately, so nothing else changes.
- "Auto" sizes the reserve from the displays detected at boot (another owner reports 0.5 GB with no display or one 1080p/1440p display, [kelchm/home-lab](https://github.com/kelchm/home-lab/issues/1030)); we chose the fixed 0.5 GB so the figure does not depend on what is plugged in.
- Do not unplug or switch a monitor in the first few minutes after boot: one owner hung the GPU that way in 6 of 14 tries, at 2 GB as well as 0.5 GB, so the setting is not the cause ([kelchm/home-lab](https://github.com/kelchm/home-lab/issues/1030)).
- **Undo:** set it back to 2 GB the same way. `start_all` re-derives `GPU_UTIL` (0.885); if you set 0.874 by hand, remove it.

**3. GPU clock cap at 2,300 MHz,** installed as a start-up service from `scripts/gb10_clock_cap_v1_0_1.service`:

```bash
sudo cp scripts/gb10_clock_cap_v1_0_1.service /etc/systemd/system/gb10-clock-cap.service
sudo systemctl daemon-reload && sudo systemctl enable --now gb10-clock-cap.service
systemctl is-active gb10-clock-cap.service   # active
nvidia-smi --query-gpu=clocks.gr,power.draw,temperature.gpu --format=csv -l 2   # watch it while the server is busy; Ctrl+C to stop
```

- **Why 2,300 MHz:** the eight-unit write-up (four DGX Sparks, four other GB10 systems) reports 0 power-offs in 448 long-prompt starts at 2,300 MHz, against 1 in 233 at default clocks once its power supplies had been separated (earlier, 2 in 22 on one unit). The write-up itself says the two rates are not a precise comparison and that counting alone cannot prove the cap is safer at a loss rate this low; it chose 2,300 MHz partly for thermal margin. That is their result, not ours.
- **A lower cap has stronger single-unit evidence.** One owner reproduced the power-off on purpose on an ASUS GX10 (also GB10): uncapped, a heavy run switched off at 156 s; capped at 2,200 MHz (`nvidia-smi -lgc 300,2200`), the same run finished in 251 s, time at or above 95 °C fell from 12% to 0% and peak power from 93.2 to 74.0 W, at about 9% longer run time. The author calls it a workaround, not a fix ([ai-muninn, Sep 14](https://ai-muninn.com/en/blog/gx10-thermal-hard-poweroff); see also [this NVIDIA forum thread](https://forums.developer.nvidia.com/t/gb10-abruptly-powers-off-under-heavy-gpu-load-nvidia-smi-lgc-300-2200-prevents-it/384492)). We kept 2,300 MHz and have not measured 2,200 on our nodes. If a node still powers off at 2,300, change `2300` to `2200` in the service file's `ExecStart` line and reinstall it.
- **What it costs in clock:** without the cap `nvidia-smi` reported a maximum of 3,003 MHz on our nodes, so this cuts the ceiling by 23%. Measured effect on serving: [results/v2_20261031/README.md](results/v2_20261031/README.md).
- **Pass:** `systemctl is-active gb10-clock-cap.service` prints `active`, and under load the clock stays at or below 2,300 MHz. In a 25-second matrix-multiply test (our own, not published) our three GPUs ran at 2,210–2,275 MHz and 87–93 W (as `nvidia-smi` reports it) and idled at 305 MHz afterwards. We did not run the same test without the cap, so this does not by itself show the cap binding. The write-up notes that on its driver `nvidia-smi -q` does not show the lock.
- **A GPU that sits near 611 MHz under load after a power loss or a reboot is stuck** (reported in the same write-up, on its non-NVIDIA GB10 units); a reboot does not clear it, removing AC power for about 15 s does.
- **What we have and have not tested:** the measurements above were taken with the unit's first version (v1.0.0: no wait for the GPU, and the clock lock skipped if persistence mode failed), started with `enable --now`. v1.0.1 adds the wait and the fallback; it is installed on our nodes, but none has rebooted with it yet, so its start at boot is untested.
- **Undo:** `sudo systemctl disable --now gb10-clock-cap.service && sudo nvidia-smi -rgc`.

## Step 1: ring network

Cable the triangle exactly as NVIDIA's [three-Spark ring playbook](https://build.nvidia.com/spark/connect-three-sparks/three-sparks-ring) specifies: **every cable joins a port 0 on one node to a port 1 on the next.**
- Port 0 is the QSFP port next to the Ethernet port.
- We first cabled one link port 1 to port 1, and pair tests over it failed until we re-plugged.

| Cable | End A | End B |
|---|---|---|
| C1 | node0 port 1 | node1 port 0 |
| C2 | node1 port 1 | node2 port 0 |
| C3 | node2 port 1 | node0 port 0 |

Each QSFP port appears to Linux as two network interfaces and two RDMA devices, one per PCIe half:
- port 0: `enp1s0f0np0` / `rocep1s0f0` and `enP2p1s0f0np0` / `roceP2p1s0f0`;
- port 1: the same names with `f1np1` / `f1`.

We give each cable two /24 subnets (one per half), jumbo frames, and no DHCP or IPv6 link-local addressing. This addressing is ours. NVIDIA's playbook uses a different scheme; either works as long as both ends of each cable share subnets.

Write `/etc/netplan/40-ring.yaml` on each node (`sudo chmod 600` it), then `sudo netplan apply`.

**node0**
```yaml
network:
  version: 2
  ethernets:
    enp1s0f1np1:   {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.100.1/24]}
    enP2p1s0f1np1: {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.101.1/24]}
    enp1s0f0np0:   {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.102.1/24]}
    enP2p1s0f0np0: {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.103.1/24]}
```

**node1**
```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:   {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.100.2/24]}
    enP2p1s0f0np0: {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.101.2/24]}
    enp1s0f1np1:   {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.104.1/24]}
    enP2p1s0f1np1: {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.105.1/24]}
```

**node2**
```yaml
network:
  version: 2
  ethernets:
    enp1s0f0np0:   {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.104.2/24]}
    enP2p1s0f0np0: {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.105.2/24]}
    enp1s0f1np1:   {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.102.2/24]}
    enP2p1s0f1np1: {dhcp4: no, dhcp6: no, link-local: [], mtu: 9000, addresses: [192.168.103.2/24]}
```

**Check every cable: six interface pairs, three cables × two halves.** For each row, start the server, then run the client:
- on the server node: `ib_write_bw -d <server device> -x 3 -F --report_gbits -s 1048576 -D 5`
- on the client node: `ping -M do -s 8972 -c 3 <server address>` and `ib_write_bw -d <client device> -x 3 -F --report_gbits -s 1048576 -D 5 <server address>`

| Cable, half | Server node, device | Server address | Client node, device |
|---|---|---|---|
| C1, 1 | node1 `rocep1s0f0` | 192.168.100.2 | node0 `rocep1s0f1` |
| C1, 2 | node1 `roceP2p1s0f0` | 192.168.101.2 | node0 `roceP2p1s0f1` |
| C2, 1 | node2 `rocep1s0f0` | 192.168.104.2 | node1 `rocep1s0f1` |
| C2, 2 | node2 `roceP2p1s0f0` | 192.168.105.2 | node1 `roceP2p1s0f1` |
| C3, 1 | node2 `rocep1s0f1` | 192.168.102.2 | node0 `rocep1s0f0` |
| C3, 2 | node2 `roceP2p1s0f1` | 192.168.103.2 | node0 `roceP2p1s0f0` |

`-x 3` selects the RoCE v2 IPv4 address entry (GID index 3) on our nodes. Confirm yours and change it if needed: `cat /sys/class/infiniband/<device>/ports/1/gids/3` should show your IPv4 address in IPv6-mapped form, and `cat /sys/class/infiniband/<device>/ports/1/gid_attrs/types/3` should print `RoCE v2`.

**Then a three-node NCCL test from node0.** nccl-tests must be built against MPI. `-H` takes any address each node is reachable on; `enP7s7` is the Spark's built-in Ethernet port, used here only for setup traffic.

```bash
export NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1 NCCL_IB_SUBNET_AWARE_ROUTING=1 NCCL_IB_MERGE_NICS=0 NCCL_NET_PLUGIN=none
mpirun -np 3 -H localhost:1,<node1-address>:1,<node2-address>:1 --mca plm_rsh_agent "ssh -o BatchMode=yes" \
  --mca btl_tcp_if_include enP7s7 --mca oob_tcp_if_include enP7s7 -x LD_LIBRARY_PATH -x NCCL_SOCKET_IFNAME=enP7s7 \
  -x NCCL_IB_HCA -x NCCL_IB_SUBNET_AWARE_ROUTING -x NCCL_IB_MERGE_NICS -x NCCL_NET_PLUGIN -x NCCL_DEBUG=INFO \
  /opt/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

**Pass:**
- all 12 ring interfaces up at 200,000 Mb/s (`cat /sys/class/net/<if>/speed`);
- jumbo pings with 0% loss;
- every interface pair at about 109 Gb/s (we measured 109.27–109.34);
- the all-gather completes. We measured 23.9 GB/s average bus bandwidth.

| NCCL variable | Value | Why |
|---|---|---|
| `NCCL_IB_HCA` | `rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1` | All four RDMA devices are eligible |
| `NCCL_IB_SUBNET_AWARE_ROUTING` | `1` | Pick the local device on the peer's subnet |
| `NCCL_IB_MERGE_NICS` | `0` | Keep the two halves of a port as separate NICs |
| `NCCL_NET_PLUGIN` | `none` | Use NCCL's built-in RoCE transport |
| `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` | `enP7s7` | Bootstrap over the LAN |

## Step 2: download and verify the checkpoint

**Shortcut: download the padded checkpoint instead (skips Steps 2 and 3).** The output of Step 3 is published at [Bespoke-Intelligence/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark](https://huggingface.co/Bespoke-Intelligence/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark). On the build node:

```bash
REPO=Bespoke-Intelligence/Beam-1.0-501B-A23B-NVFP4-TP3-3x-DGX-Spark
REV=main      # or pin a commit from the repo's history
DST=/srv/models/tp3/Beam-1.0-501B-A23B-NVFP4-TP3
mkdir -p "$DST"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_DISABLE_TELEMETRY=1 -v "$DST:/out" \
  python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 sh -c \
  "pip install -q --target /tmp/pkgs huggingface_hub==1.27.0 && PYTHONPATH=/tmp/pkgs python -c \
  \"from huggingface_hub import snapshot_download as s; s(repo_id='$REPO', revision='$REV', local_dir='/out', max_workers=8)\""
(cd "$DST" && sha256sum SHA256SUMS && sha256sum --quiet -c SHA256SUMS && echo SUMS_MATCH)
python3 scripts/check_sums_v0_1_0.py "$DST"
```

- **Pass:** the first line printed is `9f57e87b492a27c3fa3cea11b0e45e1f4784069a35ee05ea2b3e0f1c100cd5dd  SHA256SUMS`, then `SUMS_MATCH` and `SUMS_VS_MANIFEST=OK`. The fingerprint is published here on GitHub, separately from the Hugging Face repo, so a changed checksum list cannot pass unnoticed.
- `SHA256SUMS` lists the 64 files the padding step produced. Reflection's original model card is listed under its new name, `README_reflection.md`. The Hugging Face model card (`README.md`) and `NOTICE` are not listed.
- If the Hugging Face page still shows an "upload in progress" notice, the files are incomplete; build it yourself with Steps 2 and 3 instead.
- Then continue at Step 4.

**Or build it yourself (Steps 2 and 3).** Run on the build node. The revision is pinned. The download runs in a throwaway container, so no Hugging Face client is installed on the host.

```bash
REPO=reflection/Beam-1.0-501B-A23B-NVFP4
REV=3bd1dccf13c8f7aa3d6732c5aaa48ba107a2c086
SRC=/srv/models/Beam-1.0-501B-A23B-NVFP4
W=/srv/models/_manifests
mkdir -p "$SRC" "$W"
curl -fsS "https://huggingface.co/api/models/$REPO" -o "$W/api_model.json"
curl -fsS "https://huggingface.co/api/models/$REPO/tree/$REV?recursive=true" -o "$W/tree_$REV.json"
echo "$REV" > "$W/REVISION"
python3 -c "import json;t=[e for e in json.load(open('$W/tree_$REV.json')) if e['type']=='file'];print(len(t),'files',sum(e['size'] for e in t),'bytes')"   # want: 63 files 293654927078 bytes
# download in a throwaway python:3.12-slim container (pinned by digest)
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_DISABLE_TELEMETRY=1 -v "$SRC:/out" \
  python@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 sh -c \
  "pip install -q --target /tmp/pkgs huggingface_hub==1.27.0 && PYTHONPATH=/tmp/pkgs python -c \
  \"from huggingface_hub import snapshot_download as s; s(repo_id='$REPO', revision='$REV', local_dir='/out', max_workers=8)\""
python3 scripts/verify_hub_v0_1_0.py --dir "$SRC" --manifests "$W"
```

- **Before the download starts, the listing check must print** `63 files 293654927078 bytes`. The listing should hold 53 safetensors shards and no `.py`, `.bin`, `.pt` or pickle files; check with `python3 -c "import json,collections;print(collections.Counter(e['path'].rsplit('.',1)[-1] for e in json.load(open('$W/tree_$REV.json')) if e['type']=='file'))"`.
- **Progress looks frozen.** `snapshot_download` writes each file only when it finishes, so watch bytes received instead (e.g. `docker stats`).
- **Pass:** `VERIFY_RESULT=PASS`. The verifier checks every file's size, the SHA-256 of the 54 large files, and the git SHA-1 of the small ones.
- **Measured:** 42 min 54 s at about 114 MB/s; verification took 73 s.

## Step 3: pad, verify and test

The padder (`scripts/pad_beam_tp3_v0_1_0.py`, SHA-256 `8a66067ee51f25b106be593245ddd1e4ce721b9e7ef54841a97dc0553c53929c`) needs only Python and NumPy.
- **Run it in the pinned vLLM image,** which supplies NumPy, with networking off and the source mounted read-only.
- **It refuses to write if it meets any tensor it does not recognise.**
- **This is the exact file that produced our checkpoint.** It is published unchanged so its hash, and the version it stamps into the output, match the checkpoint we serve. (Its header names the author's own design notes, "Beam v1.1 review (Oct 5, 2026)": a planning document written from Beam's announced architecture before the weights were released. It is not a Beam version and is not published.)
- **To re-run it, delete `$DST` first;** the padder refuses to write into an existing folder.

```bash
SRC=/srv/models/Beam-1.0-501B-A23B-NVFP4
DST=/srv/models/tp3/Beam-1.0-501B-A23B-NVFP4-TP3
VIMG=vllm/vllm-openai@sha256:c1a6be6ad08016f8cebcb449146b3d992791fe1868f9a26f3e48a51040517b77
mkdir -p /srv/models/tp3
A="-u -I /w/pad_beam_tp3_v0_1_0.py --src $SRC --dst $DST --moe-to 2688"
docker run --rm --init --network none --user "$(id -u):$(id -g)" \
  -v "$SRC:$SRC:ro" -v /srv/models/tp3:/srv/models/tp3 -v "$PWD/scripts:/w:ro" --entrypoint sh "$VIMG" -c \
  "python3 $A --dry-run && python3 $A && python3 $A --verify && \
   python3 $A --numeric-test --numeric-layers 0,3,0 && echo ALL_GATES_PASS"
(cd "$DST" && find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%P\n' | sort | xargs -P 8 -n 1 sha256sum > SHA256SUMS)
python3 scripts/check_sums_v0_1_0.py "$DST"
```

| Phase | Pass line | Our time |
|---|---|---|
| Dry run | `PLAN_OK tensors=151262 shards=53` and `TP_GQA_MAPPING OK` | seconds |
| Write | `WRITE_DONE shards=53` | 541 s |
| Byte-level verify | `VERIFY tensors=151262 bad=0`, `VERIFY_RESULT=PASS` | 409 s |
| Numeric test | `NUMERIC_RESULT=PASS` | under a minute |
| Hash cross-check | 64 lines in `SHA256SUMS`, `SUMS_VS_MANIFEST=OK` | under a minute |

What changes, per layer (zeros appended at the end of each axis):

| Tensor | Original → padded | Zeros added to |
|---|---|---|
| `self_attn.q_proj.weight` (BF16) | [6144, 5120] → [6912, 5120] | rows (query heads 48–53) |
| `self_attn.k_proj.weight`, `v_proj.weight` (BF16) | [1024, 5120] → [1152, 5120] | rows (KV head 8) |
| `self_attn.gate_proj.weight` (BF16) | [6144, 5120] → [6912, 5120] | rows |
| `self_attn.gate_bias` (BF16) | [48] → [54] | entries |
| `self_attn.o_proj.weight` (BF16) | [5120, 6144] → [5120, 6912] | columns |
| `self_attn.q_norm`, `k_norm` | [128] | unchanged (shared across heads) |
| Dense `mlp.gate_proj`, `up_proj` (layers 0–2) | [20480, 5120] → [20736, 5120] | rows |
| Dense `mlp.down_proj` | [5120, 20480] → [5120, 20736] | columns |
| Expert `gate_proj`/`up_proj.weight` (U8, two FP4 per byte) | [2560, 2560] → [2688, 2560] | rows |
| Expert `gate_proj`/`up_proj.weight_scale` (E4M3) | [2560, 320] → [2688, 320] | rows |
| Expert `down_proj.weight` (U8) | [5120, 1280] → [5120, 1344] | byte columns |
| Expert `down_proj.weight_scale` (E4M3) | [5120, 160] → [5120, 168] | columns |
| `weight_scale_2`, `input_scale`, routers, norms, embeddings, `lm_head` | — | unchanged |

`config.json` changes in four fields only:
- `num_attention_heads` 54;
- `num_key_value_heads` 9;
- `intermediate_size` 20736;
- `moe_intermediate_size` 2688.

The padder also writes `PADDING_MANIFEST.json`, recording its version, arguments, per-shard hashes and the config changes. The proof that the padded model computes the original function is in [docs/PROOF.md](docs/PROOF.md).

## Step 4: build the serving image

The only source change is vLLM's vocabulary padding multiple, 64 → 192, so the 151,936-token vocabulary pads to 152,064 = 3 × 50,688. No fake tokens are added; vLLM masks its own padded rows.

```bash
docker build -t beam-tp3:v1 docker/
docker run --rm --network none --entrypoint python3 beam-tp3:v1 -c 'import importlib.util as u; p=u.find_spec("vllm").submodule_search_locations[0]+"/model_executor/layers/vocab_parallel_embedding.py"; print([l.strip() for l in open(p) if l.startswith("DEFAULT_VOCAB_PADDING_SIZE")])'
```

- **Pass:** prints `['DEFAULT_VOCAB_PADDING_SIZE = 192']`.
- **The build fails on purpose** if the upstream line ever changes.
- **Image IDs embed a build timestamp,** so yours will differ from ours. Compare the file content, not the ID.

### Step 4b: build the scheduler image (node0 only, v2)

**Run this on node0 after Step 5.** It builds on top of `beam-tp3:v1`, which node0 only receives in Step 5; before that, Docker tries to download a `beam-tp3:v1` from Docker Hub and the build fails. (Upgrading from v1, node0 already has it.)

```bash
docker build --progress=plain -t beam-tp3:v1-sched1 -f docker/Dockerfile.sched1 docker/
docker run --rm --network none --entrypoint sh beam-tp3:v1-sched1 -c 'F=$(python3 -c "import importlib.util as u; print(u.find_spec(\"vllm\").submodule_search_locations[0])")/v1/engine/core.py; sha256sum "$F"; grep -c "LOCAL PATCH sched1" "$F"'
```

- **Pass:** the build prints `ORIGINAL_SHA 20e44418e3db7c12d6333a0a227eeeb606cb5bcd93bd9dc44326a1c480099611`, `PATCHED_SHA 675dc58ac841fc3ac8f3811385e26a2def8d71f77a5085c531897493f9d61b23` and `PATCH=OK lines 2717 -> 2728`; the check prints that same patched hash and `2`.
- **What it changes:** one function, `EngineCore._should_throttle_prefills` in `vllm/v1/engine/core.py` (11 added lines). It returns the same `False` as before unless `--prefill-schedule-interval` is above 1. See [Concurrency and long prompts](#concurrency-and-long-prompts).
- **The build fails on purpose** if `core.py` is not byte-identical to the pinned image's (a SHA-256 check runs first and prints `core.py: FAILED`), and the patcher also refuses unless it finds the function exactly once.
- Only node0 needs it; node1 and node2 run model shards only and keep `beam-tp3:v1`.
- **Reproduced Nov 1:** building with exactly this Dockerfile gave the same patched `core.py` (`675dc58a…`) as the image the v2 measurements were taken with.

## Step 5: copy to the other nodes

Both receiving nodes pull from node1 over their own ring cable at the same time: node0 over C1, node2 over C2.

```bash
NODE1=192.168.100.2        # on node0 (cable C1); on node2 use NODE1=192.168.104.1 (cable C2)
DST=/srv/models/tp3/Beam-1.0-501B-A23B-NVFP4-TP3
sudo install -d -o "$USER" -g "$USER" /srv/models/tp3
rsync -a -e "ssh -c aes128-gcm@openssh.com" <user>@$NODE1:$DST /srv/models/tp3/
ssh <user>@$NODE1 'docker save beam-tp3:v1' | docker load
(cd "$DST" && sha256sum --quiet -c SHA256SUMS && echo SUMS_MATCH)
```

- **Pass:** `SUMS_MATCH` on both receiving nodes, and the same image ID on all three (`docker image inspect -f '{{.Id}}' beam-tp3:v1`).
- **Measured on our setup, which differed:** we relayed node2's copy through node0 (only node0 held keys for both workers). The two 309 GB copies, running at once, took about 8 and 10.5 minutes. The direct copies shown above avoid sharing a cable but are untested.

## Step 6: launch

The launcher refuses to start a node unless every check passes: image present, checkpoint present, ring ports up, no other GPU process, and at least 109 GiB free after dropping the page cache. Confirm no GPU processes are running first (`nvidia-smi`). If it says a `beam-tp3` container already exists, stop it gracefully with Step 8; the message says the same.

**One command from node0 (v2).** Copy the example settings once, fill in the first block (SSH targets, node0's LAN address, repo paths) and set `KV_BYTES` (the example holds our 4.5 GiB; start at 4.0 GiB, [Choosing the KV-cache size](#choosing-the-kv-cache-size)):

```bash
cp scripts/cluster_v2.conf.example scripts/cluster.conf && nano scripts/cluster.conf
tmux new -s beam          # recommended: the script waits about 17 minutes for the server
bash scripts/start_all_v1_0_1.sh
```

- It starts node0, then node1, then node2, asking each node's `sudo` password once, and stops before the next node if a container is not running.
- Type each password as soon as it is asked. node0's server waits for the other two to join; on Nov 1 we left node1's prompt unanswered for about 20 minutes and node0's server had exited (code 1) by the time we came back. We did not measure the exact limit. If that happens, run `scripts/stop_all_v1_0_1.sh` (it reports `ALREADY_EXITED`) and start again.
- It first checks all three nodes (reachable, launcher present, image present, no container left over) and starts nothing if any check fails.
- **Pass:** `PREFLIGHT_OK` for ranks 0–2, each node's launcher output containing `STARTED beam-tp3`, `CONTAINER_RUNNING` for all three ranks, `PAGECACHE_HELPER_RUNNING` on all three nodes, one line a minute until `health=200`, then `GPU KV cache size: …` and `START_ALL_READY`.
- Its settings come from `scripts/cluster.conf`; the example holds our v2 settings of record. Choose `KV_BYTES` for your nodes with [Memory margin](#memory-margin) first.
- **If `cluster.conf` does not set `GPU_UTIL`,** the script derives it from the largest `MemTotal` on the three nodes (107.7 GiB ÷ `MemTotal`, rounded down to three decimals: 0.885 at the default 2 GB display memory, 0.874 at 0.5 GB) and prints the value it used.
- **What we ran:** on Nov 1 we started our cluster with `start_all_v1_0_0.sh` (the version before this one) and a `cluster.conf` made from the example. The first start after Step 0b failed in vLLM's start-up memory check at `GPU_UTIL` 0.885 (Step 0b part 2); with `GPU_UTIL=0.874`: `PREFLIGHT_OK` on all three ranks, `CONTAINER_RUNNING` and `PAGECACHE_HELPER_RUNNING` on all three nodes, then `START_ALL_READY` after 18 minutes, with the 524,552-token pool. v1.0.1 adds the `GPU_UTIL` derivation and clearer failure messages (CHANGELOG). Its own run, Nov 1 at `MAX_MODEL_LEN=262144` with `GPU_UTIL` left unset: it derived 0.874 from the nodes' memory, `PREFLIGHT_OK` / `STARTED` / `CONTAINER_RUNNING` / `PAGECACHE_HELPER_RUNNING` on all three, `START_ALL_READY` after 18 minutes (22:02:12 → 22:20:33).

**Or by hand on each node**, node0 first (the v1 way, now with the v2 settings):

```bash
S="MASTER_ADDR=<node0 LAN address> KV_BYTES=4831838208 MAX_BATCHED=4096 RETENTION=4096 LONG_PREFILL=3584 PREFILL_INTERVAL=9"   # set this on each node
S="$S GPU_UTIL=0.874"   # only if Display Reserved Memory is 0.5 GB (Step 0b); omit at the default 2 GB
env $S IMG=beam-tp3:v1-sched1 NODE_RANK=0 bash scripts/launch_node_v0_2_1.sh   # on node0
env $S NODE_RANK=1 bash scripts/launch_node_v0_2_1.sh                         # on node1
env $S NODE_RANK=2 bash scripts/launch_node_v0_2_1.sh                         # on node2
```

With none of the v2 settings, `launch_node_v0_2_1.sh` launches exactly as v1's `launch_node_v0_1_4.sh` did.

The settings that matter:

| Setting | Why |
|---|---|
| `--nnodes 3 --node-rank R --master-addr … --distributed-executor-backend mp`, `--headless` on workers | vLLM's native multi-node mode; no Ray cluster. Rank 0 serves the API. |
| `--network host --ipc host --device /dev/infiniband --ulimit memlock=-1` | NCCL needs the host's RDMA devices and pinned memory |
| `--moe-backend b12x` | FP4 mixture-of-experts kernels for this GPU generation, shipped inside FlashInfer |
| `--kv-cache-dtype fp8` | Halves KV-cache memory; 9 KV heads cost 12.5% more than 8 would |
| `--gpu-memory-utilization 0.885` | vLLM's share of unified memory; the rest stays with the OS and host processes. 0.88 left too little KV-cache memory on two of three launches; see [Memory margin](#memory-margin). **0.874 with Display Reserved Memory at 0.5 GB** (Step 0b), which keeps the same claim in GiB |
| `--max-model-len 131072 --max-num-seqs 4` | Our context and concurrency targets. Only 9 of 52 layers keep full-length KV; sliding-window layers keep 512 tokens. |
| `--enforce-eager` | No CUDA-graph capture: shorter start-up and less memory, slower decode. Graphs are future work. |
| `--kv-cache-memory-bytes 4831838208` (`KV_BYTES`, v2) | Pins the KV cache at 4.5 GiB per node instead of letting vLLM estimate it; see [Memory margin](#memory-margin) |
| `--max-num-batched-tokens 4096` (`MAX_BATCHED`, v2) | Reads up to 4,096 prompt tokens per step instead of 2,048 |
| `--prefix-cache-retention-interval 4096` (`RETENTION`, v2) | Keeps sliding-window cache checkpoints every 4,096 tokens, so a document stays cached after a long answer |
| `--long-prefill-token-threshold 3584` (`LONG_PREFILL`, v2) | While other requests are active, a long prompt takes at most 3,584 of each step's 4,096 tokens, leaving room for new requests |
| `--prefill-schedule-interval 9` (`PREFILL_INTERVAL`, v2; node0 needs `beam-tp3:v1-sched1`) | While requests are writing, a long prompt is read on 1 step in 9 and the other 8 are decode-only |
| `VLLM_ENGINE_READY_TIMEOUT_S=3600` | Stops rank 0 giving up during the long, silent repack phase |
| `--host 127.0.0.1` | The HTTP API listens on loopback only. Put an authenticated proxy in front of it if anything else needs access. The distributed control plane (master port 29510 and the worker sockets) still listens on the LAN interface without authentication, so firewall it to the three nodes. |

**Firewall the LAN side.** An example with `ufw` (not what we ran; adapt it, and keep a console open in case you lock yourself out). On each node, allow the other two nodes' LAN addresses, the four ring interfaces and SSH from your own machine, then deny everything else:

```bash
sudo ufw allow from <other-node-A LAN address>; sudo ufw allow from <other-node-B LAN address>
for i in enp1s0f0np0 enP2p1s0f0np0 enp1s0f1np1 enP2p1s0f1np1; do sudo ufw allow in on $i; done
sudo ufw allow from <your admin machine> to any port 22 proto tcp
sudo ufw default deny incoming; sudo ufw enable
```

**Start-up time varies a lot.** With `launch_node_v0_1_4.sh` on Oct 21 (freshly rebooted nodes, settings as published), it took **17 minutes** from launch to `/health` returning 200, on both launches: the workers loaded their weights in about 4 minutes and node0 in about 15. Our first-ever launch (Oct 18) took an estimated 65–70 minutes, including a long phase with no log output after the weights loaded:

| First launch, Oct 18 | node0 | node1 | node2 |
|---|---|---|---|
| Weight load | 886 s | about 1,838 s | 224 s |
| One-time b12x expert repack (no log output) | about 25–30 min | about 25–30 min | about 60 min (that node was swapping) |

We have not isolated why the first launch was so much slower.

- **Do not stop the containers during a silent phase.** Allow up to about 70 minutes before suspecting a hang.
- **The script's page-cache helper:** after the weights load, it clears the Linux page cache every 10 s until vLLM has sized its KV cache, then stops. On GB10 the page cache sits in the same memory, and leftover cache from reading the weights shrinks vLLM's KV-cache estimate.
- **Ignore `NVRM` out-of-memory lines** in the kernel log during loading; they appear when `nvidia-smi` runs and do not mean loading failed. Watch `MemAvailable` falling and the container state instead.
- **Done** when `curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:8600/health` on node0 prints `200`.

## Step 7: smoke tests

On node0: `bash scripts/smoke_test_v0_1_1.sh`

| Check | Pass |
|---|---|
| Health | `200` |
| Tool call | `finish_reason=tool_calls`, one `get_weather` call with city `Paris` |
| Arithmetic canary | `391` |
| Long-context needle (synthetic service manual, 41,522 tokens; the script prints the count) | `TANGERINE-7X4Q` |

Beam reasons before it answers, and reasoning tokens count against `max_tokens`. Our first long-context probe used `max_tokens: 60` and came back empty. The script allows 600.

## Step 8: stop and restart

From node0: `bash scripts/stop_all_v1_0_1.sh`

- It stops all three containers at once with a 120-second grace period (`docker stop -t 120`), saves each node's container log to `~/beam_tp3_log_<time>.txt`, removes the containers, then drops each node's page cache, asking each node's `sudo` password once.
- **Pass:** `STOPPED_AND_REMOVED exit_code=0` for all three nodes, then `MEM_BACK MEM_AVAILABLE_GIB=11x` for each, then `STOP_ALL_DONE`. `ALREADY_EXITED exit_code=N` means that server had stopped on its own before you ran the script (a crash or failed start); its log is still saved, and the run ends in `STOP_ALL_DONE_WITH_NOTES`. `DROP_FAILED` (usually a mistyped `sudo` password) ends it in `STOP_ALL_INCOMPLETE`; drop that node's cache by hand (`sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`) before starting again. You can start again straight away (Step 6).
- `KILLED_AFTER_TIMEOUT` (exit code 137: the server did not stop within 120 s and was killed) or `MEM_HELD` on a node: reboot that node before relaunching. `UNREACHABLE`: that node was not stopped; stop it on the node itself.
- **v1 said to reboot every node after stopping. That was for `docker rm -f`, which kills the server.** A graceful stop returned the memory on every stop we ran from Oct 28 to Oct 31. If a node still prints `MEM_HELD`, reboot that node before relaunching.
- By hand on each node: `docker stop -t 120 beam-tp3 && docker rm beam-tp3 && sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`.

## Results

**v2 (Oct 28–31, 2026):** see [What's new in v2](#whats-new-in-v2-nov-2-2026), [Concurrency and long prompts](#concurrency-and-long-prompts) and the full tables in [results/v2_20261031/README.md](results/v2_20261031/README.md). **Legal agent (Oct 24 – Nov 2):** [results/legal_agent_eval_20261102/README.md](results/legal_agent_eval_20261102/README.md). Everything below this line is the v1 measurement set (Oct 18–21, 2026), unchanged, except the v2 part of [Memory margin](#memory-margin).

Measured Oct 18–21, 2026 on the hardware above.

**Pipeline time:** about 3 h 3 min from starting the download to the first token. The timed phases add up to about 2 h 20 min; the rest was the image build, checks and pauses between steps. The padding itself (write, verify and test) took about 16–17 minutes.

**Serving:**

| Probe | Result |
|---|---|
| Single-stream decode | About 18 tokens/s: 18.0–18.2 in the throughput sweep below, 18.1–18.4 in the server's own log in interactive use |
| Tool call (`smoke_test_v0_1_1.sh`, Oct 21) | `finish_reason=tool_calls`, one `get_weather` call with arguments `{"city": "Paris", "unit": "celsius"}` |
| Long-context needle (`smoke_test_v0_1_1.sh`, Oct 21) | `TANGERINE-7X4Q` returned exactly from a 41,522-token prompt |
| Arithmetic canary | `391`, 57 completion tokens (about 3.25 s in the soak's canary runs) |
| KV cache (Oct 21, published settings, two launches) | 2.44 / 1.35 / 1.82 and 2.35 / 1.36 / 1.56 GiB available on node0 / node1 / node2; the smallest sets the pool: 178,253 and 180,048 tokens, enough for 1.36–1.37 concurrent requests at the full 131,072-token context. (The first launch, at 0.88, reported 2.02 GiB per node and 266,659 tokens.) |

**Reflection's Beam Verifier (Oct 21):** `beam-verifier verify` (commit `dfa85fe771b3bcf014f5af5913f2ed363d31dfdf`), run on node0 against the loopback API. **131 of 133 checks passed** in 29 minutes, streamed and not: all response-contract (structured output) and tool-protocol checks (tool-call shape, parallel calls, every `tool_choice` mode, tool-result round trips), and every request-contract check except one. The 2 failures are the same check in both modes: a message with an invalid role (`fake_role`) should be refused with HTTP 400, but this server accepts it and starts answering (the non-streamed request then hit the verifier's time limit). vLLM's chat API accepts custom roles by design (`CustomChatCompletionMessageParam`, documented as "Enables custom roles in the Chat Completion API"), and the check runs before the model does, so we believe the padding cannot cause or fix it. We did not run the verifier against an unpadded checkpoint to confirm.
- **To run it yourself:** the verifier's documented install did not pass its options to its test runner when installed as a package, so we added one pytest setting:
  `beam-verifier verify --base-url http://127.0.0.1:8600/v1 --model-name beam -- --confcutdir="$(python -c 'import os, beam_verifier.functional_tests as f; print(os.path.dirname(f.__file__))')"` (use `python3` if `python` is not on your path)
- In that setup the verifier's summary table also shows the API suite as "SKIP — no tests ran", although the test log shows all 133 ran; the counts above are from the test log and its JSON report.

**Equivalence:** all 151,262 tensors verified byte for byte. On three sampled blocks (attention layer 0, dense layer 0, expert 0 of layer 3), numeric agreement is exact when unsharded, and within 5.3×10⁻¹⁶ relative error when split three ways ([docs/PROOF.md](docs/PROOF.md)).

**Public benchmarks (Oct 20–21, one pass each):** run with [scripts/pubbench_v1_0_5.py](scripts/pubbench_v1_0_5.py). The main runs used v1.0.0 and the re-asked questions v1.0.1 and v1.0.2 (earlier versions are not published; they hold private settings). Their question prompts and sampling settings are identical to v1.0.5's (checked by comparing the code); later versions add streaming, cap rounds, dataset sources pinned to exact commits and checked by hash, a pinned IFBench scorer, blanking of cut-off answers before scoring, and a record of the scoring containers and packages. All scores below were recomputed from the saved answers with v1.0.5; the container digests and scorer package versions it used are in each file under `results/benchmarks_20261021/`. The benchmarks ran from another machine through an authenticated, non-streaming reverse proxy, not from node0's loopback. Settings: temperature 0.7 and top_p 0.99 (Reflection's stated evaluation settings); reasoning effort not set, so Beam's chat template used its default, the maximum (Beam's chat template takes reasoning effort as a number from 8, `minimal`, to 512, `max`; higher allows longer reasoning); 4 requests at once; only the final answer scored (Beam's reasoning comes back separately and is never scored). An answer that hits its token cap is scored as wrong. The rule was fixed before scoring: a result is **consistent** if the published figure lies inside our 95% (Wilson) interval.

| Benchmark | Items | Token cap | Score | 95% interval | Published figure | Result |
|---|---|---|---|---|---|---|
| GPQA Diamond | 198 | 32,768 | **89.9%** (178) | 84.9–93.4 | 89.3% (Reflection, NVFP4) | consistent |
| GPQA Diamond, cut-off answers re-run at 65,536 | 198 | 32,768 / 65,536 | **90.9%** (180) | 86.1–94.2 | 89.3% (Reflection, NVFP4) | consistent |
| IFBench, strict | 300 | 16,384 | **75.0%** (225) | 69.8–79.6 | 79.2% (Reflection, NVFP4) | consistent |
| IFBench, lenient | 300 | 16,384 | **77.7%** (233) | 72.6–82.0 | 79.2% (Reflection, NVFP4) | consistent |
| IFBench, strict, cut-off answers re-run at 32,768 | 300 | 16,384 / 32,768 | **76.7%** (230) | 71.6–81.1 | 79.2% (Reflection, NVFP4) | consistent |
| IFBench, lenient, cut-off answers re-run at 32,768 | 300 | 16,384 / 32,768 | **79.3%** (238) | 74.4–83.5 | 79.2% (Reflection, NVFP4) | consistent |
| IFBench, strict, 3 still cut off re-run at 65,536 | 300 | 16,384 / 32,768 / 65,536 | **76.7%** (230) | 71.6–81.1 | 79.2% (Reflection, NVFP4) | consistent |
| IFBench, lenient, 3 still cut off re-run at 65,536 | 300 | 16,384 / 32,768 / 65,536 | **79.3%** (238) | 74.4–83.5 | 79.2% (Reflection, NVFP4) | consistent |
| GSM8K, first 200 | 200 | 16,384 | **97.5%** (195) | 94.3–98.9 | 97.5% (diffbot EXL3 3-bit) | consistent |
| HumanEval | 164 | 16,384 | **98.2%** (161) | 94.8–99.4 | 97.6% (diffbot EXL3 3-bit) | consistent |

- **The token cap matters.** On IFBench, 13 of 300 answers (4.3%) ran out of room at 16,384 tokens; given 32,768, 10 of those 13 finished and 3 still ran out, and 5 more prompts passed, raising both the strict and lenient scores by 1.7 points. Given 65,536, 2 of the last 3 finished but neither passed, and 1 still ran out, so the scores did not change. On GPQA, 2 of 198 ran out at 32,768; given 65,536, both finished and both were correct, raising the score by 1.0 point. In the primary rows, cut-off answers count as wrong. Reflection does not state its caps. Each re-run row keeps every other answer as it was and asks only the cut-off questions again, with a higher cap. Each re-run is a fresh sample rather than a continuation of the cut-off answer, so the difference between a benchmark's two rows is the effect of the cap plus sampling variation on those few questions. A fresh attempt can also be shorter than the one that ran out, which slightly favours the re-run row. An answer that runs out of room always scores zero, so in practice the change comes from the cut-off answers that now finish. Each further round with a higher cap re-asks only the questions still cut off, and is reported as its own row.
- **GPQA's two cut-off answers first failed** with a gateway time-out from the non-streaming proxy, because they were the two longest answers in the set. They were retried individually with the same settings and both ran to the cap. v1.0.1 streams answers, which avoids the time-out.
- Reflection does not say whether its IFBench figure is strict or lenient; we report both. "Lenient" is the scorer's `loose` mode.
- **IFBench's own guidance** is temperature 0, with the token cap chosen per model and more room for reasoning models; it sets no fixed cap. We used Reflection's stated evaluation settings instead, so at least the sampling matches theirs; Reflection does not state its caps, prompt format or reasoning effort.
- **Scorers:** IFBench with AllenAI's official code (commit `1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d`, pinned in v1.0.3); GPQA with OpenAI simple-evals' prompt and answer pattern, answer choices shuffled with seed 0; HumanEval by running each program against its tests in a container with no network and a read-only filesystem, 15 s limit; GSM8K by exact match on the last `Answer:` number, or on the last number in the answer if there is no `Answer:` line.
- **The GSM8K and HumanEval comparisons are rough:** they are against a different (3-bit) build whose settings are not published.
- **What these numbers do not show:** one pass each, so ordinary run-to-run variation is not measured; public benchmarks may appear in any model's training data; and matching scores are not proof of identical outputs (see [docs/PROOF.md](docs/PROOF.md) and the planned PP=3 comparison).
- **Data:** GPQA's authors ask that its questions not be posted online, so `results/benchmarks_20261021/` holds score summaries only, with no prompts or answers.

**Stability (24-hour soak, Oct 19 07:00 UTC to Oct 20 07:00 UTC):** a one-minute logger recorded `/health`, free memory, swap, NVIDIA Xid faults on every node, and server throughput, while the cluster also served other long-running requests. Result: 1,403 samples (one every 61.6 s on average) with no gap over 90 s, every sample healthy (HTTP 200), and zero Xid faults on all three nodes. A separate arithmetic canary ran every 10 minutes for 31 hours spanning the soak and passed 187 of 187, about 3.25 s each. Lowest free memory: 1 GiB, 6 GiB and 7 GiB on the three nodes; swap never exceeded 4 GiB.

**Throughput sweep (Oct 20, one run, server otherwise idle):** `vllm bench serve` from inside the serving container, random-token prompts, exactly 256 output tokens per request (`--ignore-eos`), `--seed 1`, after two unrecorded warm-up requests. 4 prompts per cell at 1 request at a time, 8 at 2, 16 at 4 (the server's `--max-num-seqs` limit), and 2 prompts of 130,000 tokens (close to the 131,072-token limit, leaving room for 256 output tokens). Median (p50) and 99th-percentile (p99) values:

| Prompt | Requests at once | Output tokens/s, all requests | Time to first token, p50 / p99 | Time per output token, p50 / p99 | Whole request, p50 |
|---|---|---|---|---|---|
| 1,024 | 1 | 17.4 | 0.72 s / 0.79 s | 54.8 ms / 54.9 ms | 14.7 s |
| 1,024 | 2 | 24.7 | 0.58 s / 1.53 s | 78.9 ms / 81.4 ms | 20.8 s |
| 1,024 | 4 | 31.9 | 0.66 s / 2.31 s | 121.5 ms / 126.1 ms | 31.8 s |
| 8,192 | 1 | 14.6 | 3.54 s / 3.66 s | 55.0 ms / 55.2 ms | 17.6 s |
| 8,192 | 2 | 20.5 | 3.66 s / 4.86 s | 90.0 ms / 93.6 ms | 25.8 s |
| 8,192 | 4 | 26.5 | 3.81 s / 10.86 s | 132.1 ms / 156.0 ms | 37.4 s |
| 32,768 | 1 | 9.6 | 12.51 s / 13.26 s | 55.1 ms / 55.3 ms | 26.6 s |
| 32,768 | 2 | 14.0 | 8.64 s / 31.72 s | 81.8 ms / 137.5 ms | 30.4 s |
| 32,768 | 4 | 16.4 | 8.96 s / 61.32 s | 125.6 ms / 288.0 ms | 40.6 s |
| 130,000 | 1 | 3.0 | 71.69 s / 71.76 s | 55.7 ms / 55.9 ms | 85.9 s |

What this means (derived from the table, not measured separately):

- **One request at a time decodes at about 18 tokens/s** (1000 ÷ 54.8 ms), and that barely changes with context: 18.0 tokens/s at 130,000 tokens.
- **Reading the prompt (prefill) runs at about 2,300 tokens/s at 8K, 2,600 at 32K and 1,800 at 130K** (prompt length ÷ time to first token at 1 request). A full-length prompt takes a little over a minute before the first token.
- **Four requests at once give about 1.8× the total output of one** (31.9 vs 17.4 tokens/s at 1K), but each request slows to about 8 tokens/s.
- **A long prompt arriving stalls the others.** With long prompts and more than one request, the slowest 1% of gaps between tokens reached about 1 second (raw `ITL` p99 in the result files: 0.93–1.11 s), because the new prompt is read while the others wait.
- The output column counts the whole run time, including prefill, so it falls as prompts get longer even though decoding speed does not.

Limits of these numbers: small samples (2–16 requests per cell, so p99 is essentially the slowest request), random tokens rather than real text, fixed 256-token answers, and a single run. Raw per-cell JSON is in `results/sweep_20261020/`.

**Cost of padding** (from tensor shapes, not measured):

| Component | Original | Padded | Growth |
|---|---|---|---|
| Attention weights, BF16, 52 layers | 10.90 GB | 12.27 GB | +12.5% |
| Expert weights incl. block scales | 277.45 GB | 291.33 GB | +5.0% |
| Dense feed-forward, 3 layers | 1.887 GB | 1.911 GB | +1.25% |
| Checkpoint total | 293.65 GB | ≈309 GB | +5.2% |
| Weight bytes read per decoded token | ≈23.0 GB | ≈24.8 GB | ≈+8% |
| KV cache per token | 8 heads | 9 heads | +12.5% |

Each node holds about 103 GB of weights.

### Memory margin

vLLM gives itself `GPU_UTIL` × 121.69 GiB on each node (107.7 GiB at 0.885). Weights, start-up working memory and other non-KV memory took 105.26–105.83 GiB per node on Oct 21; the KV cache gets the rest, and the smallest node's figure decides. One 131,072-token request needs 0.99 GiB.

| Launch | `GPU_UTIL` | KV-cache memory, node0 / node1 / node2 | Result |
|---|---|---|---|
| Oct 18 (first) | 0.88 | 2.02 GiB per node | started |
| Oct 21, 17:16 | 0.88 | 0.8 GiB on node0 (others not recorded) | failed: too little KV memory |
| Oct 21, 20:46 (page-cache helper added) | 0.88 | 1.42 / 0.77 / 0.91 GiB | failed: too little KV memory on node1 and node2 |
| Oct 21, 21:15 (published settings) | **0.885** | **2.44 / 1.35 / 1.82 GiB** | **started**; smallest node 36% above what 131K needs |
| Oct 21, 21:47 (published settings, repeat) | **0.885** | **2.35 / 1.36 / 1.56 GiB** | **started**; 180,048-token pool |

- The figure swings by up to about 1.2 GiB between launches of identical settings. Leftover page cache is part of it (the helper raised node0 from 0.8 to 1.42 GiB at the same setting); the rest we have not pinned down. Other processes on a node also reduce it.
- **If you hit the error (v1 settings, default 2 GB display memory):** raise `GPU_UTIL` to 0.89 (+0.6 GiB per node, taken from the OS; vLLM refuses to start if free memory at launch is below `GPU_UTIL` × 121.69 GiB, about 108.3 GiB at 0.89; at 0.5 GB display memory the same +0.005 steps apply from 0.874), or lower `MAX_MODEL_LEN` to the estimate vLLM prints. Better: pin the KV cache (below).
- **Correction to v1:** v1 said a desktop session costs about 2–3 GB. We measured 0.22–0.25 GiB per node (Oct 28). Running headless is not needed.
- **Correction to v1:** v1 said stopping does not give the memory back and every node must be rebooted. That happened after `docker rm -f`, which kills the server. A graceful stop (`docker stop -t 120`, then `docker rm`, then dropping the page cache; [Step 8](#step-8-stop-and-restart)) returned the memory on every stop we recorded (the stop table in [results/v2_20261031/README.md](results/v2_20261031/README.md)), with no reboot.

**v2: pin the KV cache instead.** `KV_BYTES` sets `--kv-cache-memory-bytes`, so vLLM uses exactly that much per node for the KV cache and skips its estimate (the log says it "skipped memory profiling"). The pool no longer swings between launches.

| Pinned per node | Step size (`MAX_BATCHED`) | Pool | Full 131K requests |
|---|---|---|---|
| 3.0 GiB | 2048 | 396,153 tokens | 3.02 |
| 4.0 GiB | 2048 | 528,205 tokens | 4.03 |
| 4.0 GiB | 4096 | 466,273 tokens | 3.56 |
| **4.5 GiB (v2)** | **4096** | **524,552 tokens** | **4.00** |

- **About 132,000 tokens per GiB at 2,048-token steps and about 117,000 at 4,096.** Larger steps cost pool because 43 of Beam's 52 layers use a 512-token sliding window, and vLLM reserves window-plus-step blocks for every running request from the same pool (our reading; the figures are measured).
- **Lowest free memory at 4.5 GiB** while reading 100K-token prompts with two other requests writing: 2.07 / 4.33 / 3.83 GiB on node0 / node1 / node2 (Oct 31).
- **Display Reserved Memory at 0.5 GB** ([Step 0b](#step-0b-firmware-display-memory-and-a-gpu-clock-cap-what-we-changed-on-nov-1)) adds about 1.5 GiB of `MemTotal` per node (121.69 → 123.19 GiB). With it, set `GPU_UTIL=0.874` so vLLM's start-up claim stays at about 107.7 GiB. We kept `KV_BYTES` at 4.5 GiB and left the extra memory as margin. Lowest free memory after the change, one run: 2.96 / 4.78 / 4.33 GiB on node0 / node1 / node2, against 2.07 / 4.33 / 3.83 before (less than the 1.5 GiB added; not explained, one run each).

#### Choosing the KV-cache size

1. Start at `KV_BYTES=4294967296` (4.0 GiB) with the other v2 settings.
2. Run `BENCH_NODES="node0=local node1=<user>@<node1> node2=<user>@<node2>" python3 -I scripts/bench_v1_3_0.py try1` on node0 (about 4 minutes, server otherwise idle) three times, and take each node's lowest `MIN_MEMAVAILABLE_GIB`.
3. If every node stays at or above 2.5 GiB, you can add 0.5 GiB (536,870,912 bytes), restart and measure again. Keep at least 1 GiB free on every node in the worst run.
- **Our 4.5 GiB is one step past rule 3:** at 4.0 GiB node0's lowest was 2.39 GiB (one run). We went to 4.5 GiB anyway, measured 2.07 GiB lowest on node0 (Oct 31), and, after Step 0b, 2.71 GiB lowest in the last cycle of a 2-hour soak ([results/v2_20261031/README.md](results/v2_20261031/README.md)). The rule is the cautious version for nodes you have not measured.
- **Why so much margin:** between two runs at the same `KV_BYTES` (4.0 GiB) but different step settings, the lowest free memory moved by up to about 1.7 GiB on a node: node0 fell 1.4 GiB (3.84 → 2.39) while node2 rose 1.7 GiB (2.18 → 3.86). One run each, so we cannot separate the step size's effect from run-to-run variation.
- **Why a too-large value is dangerous:** a pinned size bypasses vLLM's own check, so a value that is too large fails while serving, not at start-up. On unified memory, running short tends to stall a node rather than fail cleanly.

## Concurrency and long prompts

**What v1 did with several requests at once.** vLLM runs one step at a time across all three nodes. Each step serves every request that is writing (one token each) plus up to 2,048 tokens of any prompt being read. On these nodes a step that reads part of a prompt takes about 1 s, against about 0.08 s for a step that only writes, so while one request's 100K-token prompt was being read:
- every request already writing got one token per second (measured 13.1 → 1.08 tokens/s per request);
- a new request, even a one-line question, waited until the whole prompt was read (measured 49.4 s);
- asking the same document a new question after a long answer re-read it from scratch (measured 51.2 s).

**What v2 changes, and the trade-offs:**

| Problem | v2 setting | Measured effect | Cost |
|---|---|---|---|
| Same document, new question, re-read in full | `--prefix-cache-retention-interval 4096` | 51.2 s → 0.5 s (one document; a branch can re-read up to about 4K tokens, about 2 s, calculated) | A cached long document keeps more sliding-window blocks in each node's pool (not measured) |
| New requests wait behind a long read | `--long-prefill-token-threshold 3584` with `--max-num-batched-tokens 4096` | 49.4 s → 4.6 s for a one-line question (one run, nothing else writing) | None measured |
| Requests already writing crawl during a long read | `--prefill-schedule-interval 9` (node0: `beam-tp3:v1-sched1`) | 1.08 → 3.97 tokens/s per request | That read takes 65.0 s while two requests write, against about 45 s with the interval off (calculated); alone it is not slowed (46.4 s measured) |
| Reading speed | `--max-num-batched-tokens 4096` | 50.9 s → 46.4 s for 100K tokens | Pool needs 4.5 GiB instead of 4.0 GiB for about 525K tokens; lowest free memory also moved between those runs (one run each, so not attributable to the step size; see [Memory margin](#memory-margin)) |

**When the cap applies.** In this build the long-prompt cap is skipped when only one request is present (`scheduler.py`: the threshold is used only `if num_eligible_reqs > 1`), so a prompt read on its own still uses the full 4,096 tokens per step.

**Why the cache missed.** Sliding-window layers only need the last 512 tokens, and vLLM by default keeps that window only where the latest request ended (`--prefix-cache-retention-interval` 0). A follow-up turn continues from there and hits. A new question branches off where the document ends, but by then the answer has moved the window on and the blocks behind it were freed. With a checkpoint every 4,096 tokens, a branch re-reads at most about 4K tokens.

**Why the scheduler needed a patch.** vLLM's scheduler can already pause a partly-read prompt so that only writing requests run in that step (`defer_prefills` in `vllm/v1/core/sched/scheduler.py`); it does so only when at least one request is writing, so a prompt read on its own is never slowed. In this build, the engine turns that on only in data-parallel deployments: the single-replica `EngineCore._should_throttle_prefills` "never throttles otherwise". `docker/Dockerfile.sched1` makes it honour `--prefill-schedule-interval` too.

**Choosing the interval.** One prompt-reading step costs about 0.15 s plus 0.41 ms per token; a write-only step with two requests writing costs about 0.078 s (a two-point fit to our runs; it agrees with the measured interval-5 and interval-9 results within 5%). For a 100K-token prompt read while two requests write:

| `PREFILL_INTERVAL` | Each writing request during the read | That read |
|---|---|---|
| 1 (off) | about 0.6 tokens/s (calculated at the 3,584-token capped step) | about 45 s |
| 5 | **2.63 tokens/s (measured)** | **55.8 s (measured)** |
| **9 (v2)** | **3.97 tokens/s (measured)** | **65.0 s (measured)** |
| 13 | about 5.0 tokens/s (calculated) | about 72 s |
| 17 | about 5.9 tokens/s (calculated) | about 80 s |

- The extra read time is the writing itself: each write-only step gives every writing request a token and costs about 0.078 s. With this build's step costs, no setting makes both run at full speed at once; that would need a second copy of the model (six Sparks).
- **vLLM turns the pausing off while requests are queued** (its own "capacity" rule): if prompts are waiting for a free slot, reads run every step until the queue clears. One document plus a few chats is covered; a backlog is not.
- Speeds above are with two requests writing. More writing requests make each write-only step longer (about 0.125 s with four, from the v1 sweep).

### Longer requests (`MAX_MODEL_LEN`)

Every request takes KV-cache blocks from one shared pool as it grows, up to `MAX_MODEL_LEN`. Beam's own config allows 1,048,576 positions (`max_position_embeddings`; its 9 full-attention layers use no positional encoding, the other 43 see only the last 512 tokens). Our default is 131,072.

The "GPU KV cache size: N tokens" line vLLM prints is not a fixed size: vLLM computes it as *how many full-length requests fit* × `MAX_MODEL_LEN` (in `vllm/v1/core/kv_cache_utils.py`). The memory behind it is the same 4.5 GiB (`KV_BYTES`) at every length. What changes is the share taken by the 43 sliding-window layers: each request reserves a fixed amount for them, whatever its length, so longer requests spread that fixed cost over more tokens.

| `MAX_MODEL_LEN` | vLLM's "KV cache size" | Full-length requests at once (vLLM's figure) |
|---|---|---|
| 131,072 (default) | 524,552 tokens | 4.00 |
| 262,144 (Nov 1) | 599,354 tokens | 2.29 |

From those two lines (our arithmetic, not a vLLM output): each request reserves about 44,000 tokens' worth of blocks for the sliding-window layers, and the pool holds about 700,000 tokens of full-attention context in total. So, by vLLM's own accounting, at the same `KV_BYTES`: one or two requests can each reach 262,144; three at once can each reach about 189,000; four at once about 131,000. When requests outgrow the pool, vLLM pauses (preempts) one until another finishes; nothing fails, but that request waits.

We ran our cluster at 262,144 from Nov 1: `start_all_v1_0_1.sh` started it in 18 minutes with the line above, and the benchmark matched the 131,072 runs (T1 47.6 s, T2 18.5 tok/s, T3 12.7 / 3.93 tok/s with a 65.9 s read, T4 4.7 s / 45.6 s; results file). Answer quality and speed beyond 131,072 tokens are not measured by us; a 250K-token prompt takes over two minutes to read at the rates above.

## Why it is built this way

| Choice | Problem it solves | Cost |
|---|---|---|
| **Tensor parallelism (TP=3), not pipeline parallelism (PP=3)** | Single-stream decode is limited by memory bandwidth (about 273 GB/s per GB10). With PP, the three stages run one after another for every token, so one stream only ever uses one node's bandwidth. With TP, all three read their third of each layer at once. | At least two all-reduces per layer (104 per token) over the ring. We have not yet measured PP=3 on this model. |
| **Pad the checkpoint offline, not at load time** | Stock vLLM asserts that heads and widths divide by the TP degree. Padding the files means a stock image works, and the result can be hashed, verified byte for byte and inspected. | 5.2% more disk and memory, about 8% more bytes read per token |
| **54 query heads, although 48 divides by 3** | KV heads must become 9. Each query head reads KV head ⌊h/6⌋, and keeping the 6:1 ratio is what lets every rank hold whole query-KV groups. vLLM's alternative, duplicating KV heads, only applies when TP is a multiple of the KV count. | 12.5% more attention weights and KV cache |
| **Expert width 2,688, not the smallest valid 2,592** | 2,688 gives 896 per rank, a multiple of 64. vLLM's FlashInfer NVFP4 MoE path pads per-rank width to 64 at load time anyway. We did not confirm b12x's own rule, so this is a precaution, not a proven requirement. | 5.0% extra expert storage instead of 1.25%; extra compute only if the b12x path does not pad per-rank widths itself; 2,592 is untested |
| **Dense width 20,736, not the smallest valid 20,496** | 6,912 per rank is a multiple of 128, which common matrix-multiply tiles divide evenly. We did not measure whether that matters. | 1.25% on three layers |
| **Change vLLM's vocabulary padding instead of the vocabulary** | The tokenizer and embedding tensors stay untouched, and no fake tokens exist | A one-line image patch |
| **FP8 KV cache** | Halves KV memory on nodes with little room left after 103 GB of weights | Small numeric change in attention |
| **`--enforce-eager`** | Faster start-up and less memory | Slower decode |
| **Pinned KV cache (v2)** | vLLM's estimate swung by about 1.2 GiB between identical launches and gave a 180K-token pool; pinning gives a fixed 524,552-token pool | Bypasses vLLM's memory check, so the size must be chosen with margin |
| **Patch one engine function instead of forking vLLM (v2)** | The pausing logic already exists in vLLM, built for data-parallel deployments; only the switch was missing for one replica | One more image to build; tied to this base image, and the build stops if the function changes |
| **4,096-token steps with a 3,584-token cap (v2)** | Faster reading, and about 500 tokens of every step stay free for other requests while a long prompt is read | About 12% less pool per GiB than 2,048-token steps |
| **Native multi-node vLLM, no Ray** | Fewer moving parts; Ray's memory reservations matter on unified memory | — |
| **Pinned digests, throwaway download container, read-only source, no network for the padder** | The model, base image, download environment, host test tools, nccl-tests and benchmark datasets are pinned, and nothing in the build can change the source or reach the internet. The benchmark containers are recorded by digest in the results rather than fixed in the script. | Slightly longer commands |
| **HTTP API on loopback only** | The model API is not exposed by default | Needs a proxy for remote access. The distributed control plane on the LAN is unauthenticated; firewall it to the three nodes. |
| **Reflection's own NVFP4 checkpoint, not a community re-quantization** | Vendor-calibrated weights with published benchmarks to compare against | Larger than 3-bit alternatives (see [Related work](#related-work-and-other-builds)) |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Padder stops with a "file exists" error | The output folder is left over from an earlier run | Delete `$DST` and run Step 3 again |
| Out of disk during download or padding | Original plus padded copy plus images need about 650 GB | Build on the node with the most free disk |
| Download runs at about 12 MB/s | That node's management link negotiated 100 Mb/s | Download on a node with a fast uplink, then copy over the ring |
| Download progress looks frozen | `snapshot_download` writes each file only when it finishes | Watch bytes received, not files on disk |
| `ModuleNotFoundError: numpy` on the host | DGX OS ships without NumPy | Run the padder inside the vLLM image, as shown |
| Pair tests fail over one cable | The cable joins two same-numbered ports | Re-cable port 0 to port 1 (NVIDIA's wiring) |
| `STOP: image beam-tp3:v1 not found` | The launch script expects the image under that name | Build it with Step 4, or tag your existing image: `docker tag <image id> beam-tp3:v1` |
| `ValueError: To serve at least one request with the model's max seq len (131072), (0.99 GiB KV cache is needed, which is larger than the available KV cache memory …` | vLLM's KV-cache estimate came out too small on at least one node | See [Memory margin](#memory-margin): raise `GPU_UTIL` by 0.005, or set `MAX_MODEL_LEN` to the "estimated maximum model length" the error prints |
| After stopping or a failed start, `free -g` shows about 110 GiB used with no container and no GPU process | The server was killed (`docker rm -f`, a crash) rather than stopped | Reboot that node. Next time stop with `scripts/stop_all_v1_0_1.sh` (Step 8), which returned the memory every time we used it |
| `STOP: image beam-tp3:v1-sched1 not found` (by hand) or `rank0 PREFLIGHT_NO_IMAGE` (from `start_all`) | Step 4b not run on node0 | Build it (Step 4b) |
| Step 4b build fails with `core.py: FAILED` and `computed checksum did NOT match` (or `PATCH=FAIL …`) | node0's `beam-tp3:v1` was not built from the pinned base image in `docker/Dockerfile` | Rebuild `beam-tp3:v1` with Step 4 (or copy it from node1, Step 5); do not edit the hashes or force the patch |
| `PREFILL_INTERVAL` set but requests still crawl during reads | node0 is not running `beam-tp3:v1-sched1`, or prompts are queued (vLLM then turns the pausing off) | Check `docker inspect -f '{{.Config.Image}}' beam-tp3` on node0; once the first request has been scheduled, the engine log prints `LOCAL PATCH sched1 active: prefill_schedule_interval=N` |
| The KV pool is smaller than the table for the same `KV_BYTES` | A larger `MAX_BATCHED` reserves more sliding-window blocks per request | See [Memory margin](#memory-margin) |
| `PAGECACHE_HELPER_MISSING` from `start_all` | The launcher's page-cache helper did not start or was stopped | Stop with Step 8 and start again. Without it, leftover page cache competes with the model while it loads |
| A node switches off under load and its logs simply stop (no kernel, GPU or thermal message) | Reported by other DGX Spark owners too; cause not established | What we changed (unproven): [Step 0b](#step-0b-firmware-display-memory-and-a-gpu-clock-cap-what-we-changed-on-nov-1). Other owners report the clock cap helps; reports on firmware conflict. If it recurs, run NVIDIA's field diagnostic and contact NVIDIA support; owners with repeated power-offs were granted or requested RMAs. Log temperatures to a file flushed every few seconds so the last readings survive |
| A GPU runs near 611 MHz under load after a power loss or reboot | Reported stuck state | Shut the node down, remove AC power for about 15 s, power on |
| `ValueError: Free memory on device cuda:0 (…/123.19 GiB) on startup is less than desired GPU memory utilization (0.885, 109.02 GiB)` | Display Reserved Memory was lowered, so the memory total grew and so did vLLM's start-up claim | Set `GPU_UTIL=0.874`, or in general 107.7 ÷ the total the message prints ([Step 0b](#step-0b-firmware-display-memory-and-a-gpu-clock-cap-what-we-changed-on-nov-1)) |
| `fwupdmgr`: `libfwupd version … does not match daemon …` | `fwupd` and `libfwupd` packages out of step | Install Ubuntu's current `fwupd` (check with `apt-get -s install fwupd` first that it changes nothing else) |
| Long silence after the weights load | One-time in-process b12x expert repack | Wait. Keep `VLLM_ENGINE_READY_TIMEOUT_S=3600`. Don't lower memory or context limits in response. |
| `NVRM` out-of-memory lines during start-up | Printed when `nvidia-smi` runs during loading | Watch `MemAvailable` and container state instead |
| Empty answer to a long prompt | Reasoning used the whole `max_tokens` budget | Allow at least 600 tokens |
| A bind-mounted file appears as a root-owned folder | Docker creates missing mount sources as folders | Create mounted files and folders before `docker run` |

## Limitations

1. **No full-model equivalence measurement yet.** The proof is analytic and the numeric test samples three blocks: one attention layer (layer 0), one dense layer and one expert; other attention-layer types were not sampled separately. Planned: serve the unpadded checkpoint at PP=3 (pipeline stages split by layer, so no divisibility problem) and compare per-token probabilities with this build at temperature 0. The same run gives a measured TP-versus-PP speed comparison.
2. **Quality is measured by four public benchmarks, one pass each.** GPQA Diamond and IFBench are consistent with Reflection's published NVFP4 figures; GSM8K and HumanEval match a community 3-bit build. They do not measure quality on any particular real-world task, and run-to-run variation is not measured.
3. **Eager mode only.** All speed figures are with `--enforce-eager` (CUDA graphs off). v1's sweep is a single run with small samples; v2's benchmark is one run per setting. Not tried: CUDA graphs, speculative decoding. Reflection publishes a speculative-decoding drafter for Beam, [`reflection/Beam-1.0-501B-A23B-DFlash2`](https://huggingface.co/reflection/Beam-1.0-501B-A23B-DFlash2) (about 1.9B parameters, BF16); it is untested at TP=3. Its config has 40 attention heads, 8 key-value heads and a 17,408-wide MLP, none of which divide by 3, so it would likely need the same kind of padding as Beam. While another prompt is read, writing requests get about 4 tokens/s with the v2 settings (about 1 in v1); they cannot get full speed during a read on one copy of the model.
4. **Tight memory.** See [Memory margin](#memory-margin). The v2 pool (524,552 tokens) is pinned and bypasses vLLM's memory check; the lowest free memory we measured was 2.07 GiB on node0. Soak result at the v2 settings: see [results/v2_20261031/README.md](results/v2_20261031/README.md).
5. **Kernel behaviour on all-zero blocks is assumed, not proven.** The proof relies on each kernel mapping an all-zero FP4 block to zero. Outputs were finite and correct, but we did not audit every kernel.
6. **Unexplained start-up variance.** One node loaded identical files about 8× slower than another (1,838 s against 224 s).
7. **A slow management link.** One worker's LAN link negotiated 100 Mb/s. vLLM sends rendezvous and per-step scheduler messages to each worker over that link; its effect on decode latency is unmeasured.
8. **Invalid message roles are accepted.** Reflection's verifier expects HTTP 400 for a message with an unknown role; vLLM's chat API accepts custom roles by design, so this build does too.
9. **One repetition loop** appeared once in a chat session using tools and has not been root-caused. A candidate fix is the server variable `VLLM_BEAM_ALWAYS_CONSTRAIN_OUTPUT=0`.
10. **n = 1:** one cluster.
11. **The scheduler patch is tied to the pinned base image** and relies on vLLM's data-parallel pausing logic, which vLLM documents for data-parallel use. It was reviewed against this build's code and measured as above, not tested more widely. With prompts queued, vLLM turns the pausing off.
12. **One unexplained hard power-off.** node0 switched off with no log entry about 18 minutes into the Oct 31 soak. Other owners report that a clock cap helps; reports on firmware conflict, and display memory is a memory change, not a power-off fix. After Step 0b a 2-hour soak at `MAX_MODEL_LEN=262144` ran 16 of 16 cycles with no power-off ([results/v2_20261031/README.md](results/v2_20261031/README.md)); one run does not show the changes prevent it.

## Related work and other builds

**TP=3 on DGX Spark, other models.** All of these patch or fork vLLM, or pad at load time; this build pads the files and serves on the pinned upstream image plus a one-constant patch (v2: plus one engine function on node0).
- [MiniMax M3 on 3 Sparks TP=3](https://forums.developer.nvidia.com/t/minimax-m3-on-3-sparks-tp-3-is-now-working/373388) (Jun 15, 2026): "virtual sharding" in Luke Alonso's vLLM fork, plus b12x.
- [GLM-5.3-Flash NVFP4 on 3× DGX Spark TP=3](https://forums.developer.nvidia.com/t/glm-5-3-flash-nvfp4-on-3x-dgx-spark-tp-3-512k-context-35-tok-s/381534) (Aug 27, 2026): heads and experts padded via vLLM overlay files.
- [DeepSeek-V4-Flash TP3 on 3× DGX Spark](https://huggingface.co/PrblyElect/DeepSeek-V4-Flash-TP3-3x-DGX-Spark) (Oct 2026): padding in a patched or forked vLLM.

**The padding technique.** [marianbusoi/Hy3-NVFP4-FP8-TP3](https://huggingface.co/marianbusoi/Hy3-NVFP4-FP8-TP3) zero-padded attention heads and the dense layer of another large MoE for TP=3, and patched the vocabulary multiple the same way. Beam additionally needs its gated attention and its NVFP4 experts (with their FP8 block scales) padded.

**Other Beam builds** (as of Oct 21, 2026):
- **[Baekpica/Beam-1.0-501B-A23B-VQ-GGUF](https://huggingface.co/Baekpica/Beam-1.0-501B-A23B-VQ-GGUF)** (Oct 20, 2026): a Beam build for a **single** DGX Spark. Routed experts are vector-quantized to about 1.5 bits per weight (about 100 GB in total), served by its own engine, [ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs). It reports about 25 tokens/s decode and labels itself experimental, with reasoning loops and broken output possible. It trades fidelity for fitting on one node; this repo keeps Reflection's NVFP4 weights unchanged apart from zero padding, and spreads them over three.
- **[diffbot/Beam-1.0-501B-A23B-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/Beam-1.0-501B-A23B-EXL3-3bpw-2x-RTX-PRO-6000)**
  - 3-bit experts and 6-bit attention in EXL3, 194 GB, TP=2 on two RTX PRO 6000 cards, served by a patched vLLM with ExLlamaV3 kernels.
  - It reports divergence from the full-precision model (KL divergence on its test set) of 0.0132 for itself and 0.0116 for NVFP4 as served on Blackwell. That is a useful outside reference for how close a served NVFP4 Beam is to the original.
  - Its card states that its vllm-exl3 patch is AGPL-3.0-only.
- **GGUF quantizations** for llama.cpp: [unsloth](https://huggingface.co/unsloth/Beam-1.0-501B-A23B-GGUF) and [AtomicChat](https://huggingface.co/AtomicChat/Beam-1.0-501B-A23B-GGUF), which publishes KL-divergence figures for each size.
- The official [vLLM recipe](https://recipes.vllm.ai/reflection/Beam-1.0-501B-A23B) covers B200/B300 and GB200/GB300 at TP4, and H100/H200 at TP8; the [day-0 blog post](https://vllm.ai/blog/2026-10-16-reflection-beam) covers GB200/GB300 at TP4/TP8. Neither covers DGX Spark or TP=3. Reflection's own [deployment guides](https://developers.reflection.ai/deployment-overview) and [cookbook](https://github.com/reflection-oss/cookbook) cover data-centre serving and fine-tuning, not DGX Spark.
- A [tuned configuration for 4× RTX PRO 6000](https://huggingface.co/reflection/Beam-1.0-501B-A23B-NVFP4/discussions/5) on a single PCIe workstation is posted in the NVFP4 model's discussions.

## Credits, citation and licence

**Built on:**
- [Reflection](https://reflection.ai/blog/introducing-beam): the model.
- [vLLM](https://github.com/vllm-project/vllm): serving, including the prefill-deferral logic (built for data-parallel balancing) that v2's scheduler patch switches on for a single replica.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) and **Luke Alonso's [b12x](https://pypi.org/project/b12x/)** kernels, which ship inside FlashInfer: the FP4 MoE path.
- NVIDIA's [three-Spark ring playbook](https://build.nvidia.com/spark/connect-three-sparks/three-sparks-ring): the wiring.
- **eugr and dbsci** on the NVIDIA developer forum: the switchless-ring NCCL settings this build uses ([Mar 31](https://forums.developer.nvidia.com/t/test-the-sample-about-connect-three-dgx-spark-in-a-ring-topology/365160) and [Apr 1, 2026](https://forums.developer.nvidia.com/t/three-node-spark-clusters-without-a-switch-are-now-supported-in-spark-vllm-docker-and-sparkrun/365296)).
- **marianbusoi:** the Hy3 TP=3 padding approach.
- [nccl-tests](https://github.com/NVIDIA/nccl-tests) and [perftest](https://github.com/linux-rdma/perftest).
- Benchmarks: [GPQA](https://arxiv.org/abs/2311.12022) (Rein et al.), [IFBench](https://github.com/allenai/IFBench) (Pyatkin et al., AllenAI), [GSM8K](https://arxiv.org/abs/2110.14168) (Cobbe et al.), [HumanEval](https://arxiv.org/abs/2107.03374) (Chen et al.), and OpenAI's [simple-evals](https://github.com/openai/simple-evals) for the GPQA prompt and answer format.

**Citing Beam** (Reflection's requested citation):

```bibtex
@misc{reflection2026introducingbeam,
  author = {{Reflection}},
  title  = {Introducing {Beam}: {Reflection}'s 501B open-weight model},
  year   = {2026},
  month  = oct,
  url    = {https://reflection.ai/blog/introducing-beam}
}
```

To cite this repo, use GitHub's "Cite this repository" button, which reads [CITATION.cff](CITATION.cff).

**How this was made:** architected, orchestrated, reviewed, approved, tested and validated by Josh Pearlman on his own three-node DGX Spark cluster. Code and documentation drafted with Claude (Anthropic).

**Licence:**
- Code (`scripts/`, `docker/`, `CITATION.cff`, `REUSE.toml`): [Apache-2.0](LICENSES/Apache-2.0.txt). `docker/beam_sched1_patcher_v1_0_1.py` quotes four lines of vLLM's `vllm/v1/engine/core.py` (Apache-2.0, the vLLM contributors).
- Documentation and results (`README.md`, `CHANGELOG.md`, `docs/`, `results/`): [CC BY 4.0](LICENSES/CC-BY-4.0.txt). Command examples in the documentation may also be used under Apache-2.0.
- The Hugging Face model card and notice (`hf/`): Apache-2.0, matching the model repository they ship in.
- See [REUSE.toml](REUSE.toml).
- Beam's weights are Apache-2.0 from Reflection. This repo distributes no weights; the padded checkpoint is on Hugging Face (see Step 2).

Provided as is, with no warranty. This setup runs each node close to its memory limit.

**Support:** issues are open; help is best effort, with no support promised.
