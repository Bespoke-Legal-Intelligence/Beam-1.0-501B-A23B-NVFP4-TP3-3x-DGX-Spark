#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# launch_node_v0_2_1.sh - start one rank of Beam-1.0 NVFP4 at TP=3. Run the same script on all three nodes,
# node0 first, or let scripts/start_all_v1_0_1.sh run it on all three from node0.
# v0.2.1 = v0.2.0 with clearer failure handling: the "container exists" hint no longer suggests docker rm -f, a failed
#   PROF_DIR mkdir stops the launch, and STARTED is printed only if the container is still running after 10 s.
# v0.2.0 (v2 recipe) = v0.1.4 plus five optional settings; with none of them set it behaves like v0.1.4:
#   PROF_DIR=<host dir>       -> torch profiler on (dir mounted at /prof; enables POST /start_profile and /stop_profile)
#   RETENTION=<tokens>        -> --prefix-cache-retention-interval   (keep sliding-window cache checkpoints every N tokens)
#   PREFILL_INTERVAL=<steps>  -> --prefill-schedule-interval          (needs the scheduler image, docker/Dockerfile.sched1)
#   LONG_PREFILL=<tokens>     -> --long-prefill-token-threshold       (per-step cap for one long prompt when others are active)
#   LONG_PREFILL_ADAPTIVE=1   -> --long-prefill-token-threshold-adaptive
# v2 settings of record (see README, "Concurrency and long prompts"):
#   KV_BYTES=4831838208 MAX_BATCHED=4096 RETENTION=4096 LONG_PREFILL=3584 PREFILL_INTERVAL=9, IMG=beam-tp3:v1-sched1 on node0
# Usage:   NODE_RANK=0 MASTER_ADDR=<node0 LAN IP> bash launch_node_v0_2_1.sh      (node0 serves the API)
#          NODE_RANK=1 MASTER_ADDR=<node0 LAN IP> bash launch_node_v0_2_1.sh      (headless worker)
#          NODE_RANK=2 MASTER_ADDR=<node0 LAN IP> bash launch_node_v0_2_1.sh      (headless worker)
# Every check below must pass or nothing starts. Stop gracefully with scripts/stop_all_v1_0_1.sh, or on each node:
#   docker stop -t 120 beam-tp3 && docker rm beam-tp3 && sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
set -u
NODE_RANK="${NODE_RANK:?set NODE_RANK to 0, 1 or 2}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to the LAN address of node0}"
LAN_IF="${LAN_IF:-enP7s7}"                                     # the Spark's built-in Ethernet port
IMG="${IMG:-beam-tp3:v1}"                                      # built from docker/Dockerfile
MODEL="${MODEL:-/srv/models/tp3/Beam-1.0-501B-A23B-NVFP4-TP3}" # the padded checkpoint
PORT="${PORT:-8600}"
MIN_FREE_GIB="${MIN_FREE_GIB:-109}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"                        # lower it if vLLM reports too little KV cache memory; see README
GPU_UTIL="${GPU_UTIL:-0.885}"                  # 0.88 left too little KV-cache memory on some launches; 0.874 with Display Reserved Memory at 0.5 GB; see README
MAX_BATCHED="${MAX_BATCHED:-}"           # optional: --max-num-batched-tokens (smaller = less start-up memory, slower prefill)
KV_BYTES="${KV_BYTES:-}"                 # optional: --kv-cache-memory-bytes (fixes the KV cache size; skips vLLM's estimate)
LOG_LEVEL="${LOG_LEVEL:-INFO}"           # DEBUG prints vLLM's full start-up memory breakdown
PROF_DIR="${PROF_DIR:-}"                 # optional: host directory for torch-profiler traces (enables /start_profile)
RETENTION="${RETENTION:-}"; PREFILL_INTERVAL="${PREFILL_INTERVAL:-}"; LONG_PREFILL="${LONG_PREFILL:-}"; LONG_PREFILL_ADAPTIVE="${LONG_PREFILL_ADAPTIVE:-}"   # optional; see header
RING_PORTS="enp1s0f0np0 enP2p1s0f0np0 enp1s0f1np1 enP2p1s0f1np1"
OK=1; fail() { echo "STOP: $*"; OK=0; }

case "$NODE_RANK" in 0|1|2) ;; *) fail "NODE_RANK must be 0, 1 or 2";; esac
IP=$(ip -4 -o addr show "$LAN_IF" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n 1)
echo "NODE $(hostname) rank=$NODE_RANK lan_ip=${IP:-none} master=$MASTER_ADDR max_model_len=$MAX_MODEL_LEN gpu_util=$GPU_UTIL max_batched=${MAX_BATCHED:-default} kv_bytes=${KV_BYTES:-auto} prof_dir=${PROF_DIR:-off} retention=${RETENTION:-default} prefill_interval=${PREFILL_INTERVAL:-default} long_prefill=${LONG_PREFILL:-default} long_prefill_adaptive=${LONG_PREFILL_ADAPTIVE:-off}"
for x in "$RETENTION" "$PREFILL_INTERVAL" "$LONG_PREFILL"; do case "$x" in '') ;; *[!0-9]*) fail "RETENTION, PREFILL_INTERVAL and LONG_PREFILL must be whole numbers, got '$x'";; esac; done
case "$LONG_PREFILL_ADAPTIVE" in ''|1) ;; *) fail "LONG_PREFILL_ADAPTIVE must be 1 or unset";; esac
[ -n "$IP" ] || fail "no IPv4 address on $LAN_IF (set LAN_IF)"
[ "$NODE_RANK" = 0 ] && [ "$IP" != "$MASTER_ADDR" ] && fail "rank 0 must run on MASTER_ADDR ($MASTER_ADDR); this node is $IP"
docker image inspect "$IMG" >/dev/null 2>&1 || fail "image $IMG not found (build docker/Dockerfile; beam-tp3:v1-sched1 comes from docker/Dockerfile.sched1)"
[ "$( (wc -l < "$MODEL/SHA256SUMS") 2>/dev/null)" = 64 ] || fail "padded checkpoint missing at $MODEL (want 64 lines in SHA256SUMS)"
docker ps -a --format '{{.Names}}' | grep -qx beam-tp3 && fail "a beam-tp3 container already exists (stop it gracefully: scripts/stop_all_v1_0_1.sh, or docker stop -t 120 beam-tp3 && docker rm beam-tp3; docker rm -f kills a running server and can leave its memory held until a reboot)"
for p in $RING_PORTS; do [ "$(cat /sys/class/net/$p/operstate 2>/dev/null)" = up ] || fail "ring port $p is not up"; done
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
GP=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
[ "$GP" = 0 ] || fail "$GP process(es) already use the GPU; stop every other GPU workload first"
if [ "$OK" = 1 ]; then
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; sleep 3   # GB10 memory is shared; free the page cache
  AV=$(awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo)
  echo "MEM_AVAILABLE_GIB=$AV (need >= $MIN_FREE_GIB)"
  [ "$AV" -ge "$MIN_FREE_GIB" ] || fail "not enough free memory"
fi
[ -n "$PROF_DIR" ] && { mkdir -p "$PROF_DIR" 2>/dev/null || fail "cannot create PROF_DIR $PROF_DIR"; }
if [ "$OK" != 1 ]; then echo "NOTHING STARTED"; else
  EXTRA="--headless"
  OPT=""; [ -n "$MAX_BATCHED" ] && OPT="$OPT --max-num-batched-tokens $MAX_BATCHED"; [ -n "$KV_BYTES" ] && OPT="$OPT --kv-cache-memory-bytes $KV_BYTES"
  [ "$NODE_RANK" = 0 ] && EXTRA="--host 127.0.0.1 --port $PORT --served-model-name beam --reasoning-parser beam --tool-call-parser beam --enable-auto-tool-choice"
  SC=""
  [ -n "$RETENTION" ] && SC="$SC --prefix-cache-retention-interval $RETENTION"
  [ -n "$PREFILL_INTERVAL" ] && SC="$SC --prefill-schedule-interval $PREFILL_INTERVAL"
  [ -n "$LONG_PREFILL" ] && SC="$SC --long-prefill-token-threshold $LONG_PREFILL"
  [ "$LONG_PREFILL_ADAPTIVE" = 1 ] && SC="$SC --long-prefill-token-threshold-adaptive"
  OPT="$OPT$SC"; echo "SCHED_CACHE_OPTS ${SC:- none}"
  PROF=""; if [ -n "$PROF_DIR" ]; then PROF="-v $PROF_DIR:/prof"; OPT="$OPT --profiler-config.profiler torch --profiler-config.torch_profiler_dir /prof --profiler-config.torch_profiler_with_stack false"; echo "PROFILER_ENABLED traces -> $PROF_DIR"; fi
  docker run -d --name beam-tp3 --network host --ipc host --gpus all --device /dev/infiniband \
    --ulimit memlock=-1 --ulimit stack=67108864 -v "$MODEL:$MODEL:ro" \
    -e VLLM_HOST_IP="$IP" -e GLOO_SOCKET_IFNAME="$LAN_IF" -e NCCL_SOCKET_IFNAME="$LAN_IF" \
    -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 \
    -e NCCL_IB_MERGE_NICS=0 -e NCCL_NET_PLUGIN=none -e NCCL_DEBUG=WARN \
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e VLLM_LOGGING_LEVEL="$LOG_LEVEL" $PROF \
    "$IMG" "$MODEL" --tensor-parallel-size 3 --nnodes 3 --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port 29510 --distributed-executor-backend mp \
    --moe-backend b12x --load-format safetensors --kv-cache-dtype fp8 \
    --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 4 --enforce-eager $OPT $EXTRA >/dev/null \
  && sleep 10 && { [ "$(docker inspect -f '{{.State.Running}}' beam-tp3 2>/dev/null)" = true ] || { echo "FAILED: beam-tp3 exited within 10 s; see: docker logs beam-tp3"; false; }; } \
  && echo "STARTED beam-tp3 rank=$NODE_RANK state=$(docker inspect -f '{{.State.Status}}' beam-tp3)" \
  && echo "Start-up took about 17 minutes on our nodes (an estimated 65-70 on our first-ever launch), with a long silent phase after the weights load. Do not stop it. Follow it with: docker logs -f beam-tp3" \
  && { # GB10 memory is shared: page cache filled by reading the weights counts against vLLM's KV-cache estimate.
       # After the weights finish loading, drop the page cache every 10 s until vLLM has sized its KV cache.
       sudo -b sh -c 'loaded=0; for i in $(seq 1 720); do [ "$(docker inspect -f "{{.State.Running}}" beam-tp3 2>/dev/null)" = true ] || exit 0;
         docker logs beam-tp3 2>&1 | grep -qE "Available KV cache memory|GPU KV cache size" && exit 0;
         [ $loaded = 1 ] || { docker logs beam-tp3 2>&1 | grep -qE "checkpoint shards: +100%|Model loading took|MoEPrepareAndFinalize" && loaded=1; };
         [ $loaded = 1 ] && { sync; echo 1 > /proc/sys/vm/drop_caches; }; sleep 10; done' >/dev/null 2>&1;
       echo "page-cache helper running until the KV cache is sized"; } \
  || echo "FAILED: beam-tp3 did not start (see the error above)"
fi
