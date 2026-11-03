#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# start_all_v1_0_1.sh - start Beam TP=3 on all three nodes with one command. Run on node0, ideally inside tmux.
# Reads scripts/cluster.conf (copy of cluster_v2.conf.example).
# v1.0.1: runs launch_node_v0_2_1.sh; clears ~/beam_start_last.txt first, so a failed sudo cannot show an old run's
#   output; if GPU_UTIL is not set, derives it from the largest MemTotal on the three nodes (107.7 GiB / MemTotal,
#   rounded down to 3 decimals: 0.885 at the default 2 GB display memory, 0.874 at 0.5 GB) and prints it.
# 1. Preflight on all three nodes before anything starts: reachable over SSH, launcher present, image present,
#    no beam-tp3 container left over.
# 2. For node0, node1, node2 in that order: asks that node's sudo password once, runs scripts/launch_node_v0_2_1.sh
#    with the config's settings under nohup with its output in ~/beam_start_last.txt (so a dropped SSH session cannot
#    stop it), prints that output, and stops before the next node unless it says "STARTED beam-tp3" and the container
#    is running.
# 3. Confirms each node's page-cache helper is alive. The launcher starts it with "sudo -b"; with Ubuntu's default
#    "Defaults use_pty" it runs in its own session, so it survives the SSH session closing.
# 4. Waits for /health=200, one line a minute, then prints the image, the KV-cache pool and the engine settings.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; CONF="${CONF:-$HERE/cluster.conf}"
[ -f "$CONF" ] || { echo "STOP: $CONF not found (copy scripts/cluster_v2.conf.example to scripts/cluster.conf and edit it)"; exit 2; }
. "$CONF"
NODE1_SSH="${NODE1_SSH:-}"; NODE2_SSH="${NODE2_SSH:-}"; MASTER_ADDR="${MASTER_ADDR:-}"
REPO_DIR="${REPO_DIR:-$HERE/..}"; REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-}"
IMG_RANK0="${IMG_RANK0:-beam-tp3:v1-sched1}"; IMG_WORKERS="${IMG_WORKERS:-beam-tp3:v1}"
KV_BYTES="${KV_BYTES:-}"; MAX_BATCHED="${MAX_BATCHED:-}"; RETENTION="${RETENTION:-}"; LONG_PREFILL="${LONG_PREFILL:-}"
PREFILL_INTERVAL="${PREFILL_INTERVAL:-}"; PROF_DIR="${PROF_DIR:-}"
PORT="${PORT:-8600}"; LAN_IF="${LAN_IF:-enP7s7}"; GPU_UTIL="${GPU_UTIL:-}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MODEL="${MODEL:-/srv/models/tp3/Beam-1.0-501B-A23B-NVFP4-TP3}"
for v in "$NODE1_SSH" "$NODE2_SSH" "$MASTER_ADDR" "$REMOTE_REPO_DIR"; do
  case "$v" in ''|*"<"*) echo "STOP: fill in NODE1_SSH, NODE2_SSH, MASTER_ADDR and REMOTE_REPO_DIR in $CONF"; exit 2;; esac
done
for v in "$KV_BYTES" "$MAX_BATCHED" "$LONG_PREFILL" "$PREFILL_INTERVAL" "$RETENTION" "$PORT" "$MAX_MODEL_LEN"; do
  case "$v" in *[!0-9]*) echo "STOP: KV_BYTES, MAX_BATCHED, LONG_PREFILL, PREFILL_INTERVAL, RETENTION, PORT and MAX_MODEL_LEN must be whole numbers or empty, got '$v'"; exit 2;; esac
done
for v in "$IMG_RANK0" "$IMG_WORKERS" "$LAN_IF" "$MODEL" "$PROF_DIR" "$REMOTE_REPO_DIR" "$REPO_DIR"; do
  case "$v" in *[!A-Za-z0-9._:/~-]*) echo "STOP: unexpected characters in '$v'"; exit 2;; esac
done
ssh_q() { ssh -o BatchMode=yes -o ConnectTimeout=5 "$@"; }
if [ -z "$GPU_UTIL" ]; then
  MT=$(awk '/MemTotal/{print $2}' /proc/meminfo)
  for n in "$NODE1_SSH" "$NODE2_SSH"; do m=$(ssh_q "$n" "awk '/MemTotal/{print \$2}' /proc/meminfo" 2>/dev/null); case "$m" in ''|*[!0-9]*) ;; *) [ "$m" -gt "$MT" ] && MT=$m;; esac; done
  GPU_UTIL=$(awk -v m="$MT" 'BEGIN{printf "%.3f", int(107.7*1048576/m*1000)/1000}')
  echo "GPU_UTIL not set in $CONF: using $GPU_UTIL (107.7 GiB of the largest MemTotal, $MT kB)"
fi
case "$GPU_UTIL" in 0.[0-9]*) ;; *) echo "STOP: GPU_UTIL must look like 0.885, got '$GPU_UTIL'"; exit 2;; esac
SET="MASTER_ADDR=$MASTER_ADDR KV_BYTES=$KV_BYTES MAX_BATCHED=$MAX_BATCHED RETENTION=$RETENTION LONG_PREFILL=$LONG_PREFILL PREFILL_INTERVAL=$PREFILL_INTERVAL PROF_DIR=$PROF_DIR PORT=$PORT LAN_IF=$LAN_IF GPU_UTIL=$GPU_UTIL MAX_MODEL_LEN=$MAX_MODEL_LEN MODEL=$MODEL"
echo "START_ALL v1.0.1 $(date -Is)"; echo "SETTINGS $SET IMG_RANK0=$IMG_RANK0 IMG_WORKERS=$IMG_WORKERS"

echo "== PREFLIGHT"
BAD=0
pre() {  # $1 label, $2 ssh target or "local", $3 image, $4 repo dir
  local q="if [ ! -f $4/scripts/launch_node_v0_2_1.sh ]; then echo NO_LAUNCHER; elif ! docker image inspect $3 >/dev/null 2>&1; then echo NO_IMAGE; elif docker ps -a --format '{{.Names}}' | grep -qx beam-tp3; then echo CONTAINER_EXISTS; else echo OK; fi"
  local r; if [ "$2" = local ]; then r=$(bash -c "$q"); else r=$(ssh_q "$2" "$q" 2>/dev/null) || r=UNREACHABLE; fi
  echo "$1 PREFLIGHT_${r:-UNREACHABLE}"; [ "$r" = OK ] || BAD=1
}
pre rank0 local "$IMG_RANK0" "$REPO_DIR"; pre rank1 "$NODE1_SSH" "$IMG_WORKERS" "$REMOTE_REPO_DIR"; pre rank2 "$NODE2_SSH" "$IMG_WORKERS" "$REMOTE_REPO_DIR"
if [ "$BAD" != 0 ]; then
  echo "STOP: preflight failed; nothing was started. NO_LAUNCHER: git pull/clone the repo there. NO_IMAGE: README Steps 4, 4b, 5."
  echo "      CONTAINER_EXISTS: run scripts/stop_all_v1_0_1.sh first. UNREACHABLE: check SSH (README Step 0)."; exit 1
fi

R=0
for n in local "$NODE1_SSH" "$NODE2_SSH"; do
  if [ "$n" = local ]; then IMG=$IMG_RANK0; D=$REPO_DIR; else IMG=$IMG_WORKERS; D=$REMOTE_REPO_DIR; fi
  CMD=": > ~/beam_start_last.txt; sudo -v && cd $D && env IMG=$IMG NODE_RANK=$R $SET nohup bash scripts/launch_node_v0_2_1.sh > ~/beam_start_last.txt 2>&1 < /dev/null; cat ~/beam_start_last.txt"
  echo "== START rank $R ($n) $(date +%T) (type that node's sudo password if asked)"
  if [ "$n" = local ]; then bash -c "$CMD"; else ssh -t -o ConnectTimeout=5 "$n" "$CMD"; fi
  sleep 2
  Q="grep -q '^STARTED beam-tp3' ~/beam_start_last.txt && docker inspect -f '{{.State.Running}}' beam-tp3 2>/dev/null"
  if [ "$n" = local ]; then S=$(bash -c "$Q"); else S=$(ssh_q "$n" "$Q" 2>/dev/null); fi
  if [ "$S" != "true" ]; then
    echo "STOP: rank $R ($n) did not start (no 'STARTED beam-tp3' line, or the container is not running); the remaining nodes were NOT started."
    echo "      Read that node's ~/beam_start_last.txt, then run scripts/stop_all_v1_0_1.sh before trying again."; exit 1
  fi
  echo "rank $R CONTAINER_RUNNING"; R=$((R+1))
done

echo "== PAGE-CACHE HELPERS (must be running on all three while the model loads)"
HC='P=$(pgrep -f "loaded=[0]; for i in" | head -1); if [ -n "$P" ]; then echo "$(hostname) PAGECACHE_HELPER_RUNNING pid=$P"; else echo "$(hostname) PAGECACHE_HELPER_MISSING"; fi'
bash -c "$HC"; for n in "$NODE1_SSH" "$NODE2_SSH"; do ssh_q "$n" "$HC" || echo "$n UNREACHABLE"; done

echo "== WAITING FOR READY (one line a minute; 17-18 minutes on our nodes)"
c=000
for i in $(seq 1 90); do
  c=$(curl -s -o /dev/null -m 3 -w '%{http_code}' "127.0.0.1:$PORT/health")
  echo "$(date +%T) minute $i health=$c | $(docker logs --tail 1 beam-tp3 2>&1 | grep -viE 'api.?key|non-default' | cut -c1-110)"
  [ "$c" = 200 ] && break
  if [ "$(docker inspect -f '{{.State.Running}}' beam-tp3 2>/dev/null)" != "true" ]; then
    echo "CONTAINER_NOT_RUNNING"; docker logs --tail 15 beam-tp3 2>&1 | grep -viE 'api.?key|non-default' | cut -c1-200; break
  fi
  sleep 60
done
echo "IMAGE $(docker inspect -f '{{.Config.Image}}' beam-tp3 2>/dev/null)"
docker logs beam-tp3 2>&1 | grep -E 'GPU KV cache size' | cut -c1-200 | tail -1
echo "ENGINE_SETTINGS $(docker logs beam-tp3 2>&1 | grep -oE "(prefix_cache_retention_interval|max_num_batched_tokens|long_prefill_token_threshold|prefill_schedule_interval|kv_cache_memory_bytes)'?:? ?[0-9]+" | sort -u | tr '\n' ' ')"
if [ "$c" = 200 ]; then echo "START_ALL_READY $(date +%T)"; else echo "START_ALL_NOT_READY $(date +%T)"; exit 1; fi
