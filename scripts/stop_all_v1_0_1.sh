#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# stop_all_v1_0_1.sh - graceful stop of Beam TP=3 on all three nodes with one command. Run on node0.
# Reads scripts/cluster.conf (copy of cluster_v2.conf.example).
# Phase 1 (all nodes at once, no password): docker stop -t 120, save the container's log to ~/beam_tp3_log_<time>.txt,
#   docker rm. Exit code 137 means the server did not stop within 120 s and was killed: reported as KILLED_AFTER_TIMEOUT.
# Phase 2 (one node at a time; each may ask for its sudo password): drop the page cache, then report MEM_BACK
#   (>= 109 GiB available) or MEM_HELD. Exits non-zero if any node was unreachable, killed, or held its memory.
# A graceful stop returned the memory on every stop we recorded (results/v2_20261031/README.md).
# v1.0.1: a container that had already exited before the stop is reported as ALREADY_EXITED with its exit code (its
#   log is still saved), and the run ends in STOP_ALL_DONE_WITH_NOTES; a failed page-cache drop is reported as
#   DROP_FAILED and makes the run STOP_ALL_INCOMPLETE; log names include seconds.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"; CONF="${CONF:-$HERE/cluster.conf}"
[ -f "$CONF" ] || { echo "STOP: $CONF not found (copy scripts/cluster_v2.conf.example to scripts/cluster.conf and edit it)"; exit 2; }
. "$CONF"
NODE1_SSH="${NODE1_SSH:-}"; NODE2_SSH="${NODE2_SSH:-}"
for v in "$NODE1_SSH" "$NODE2_SSH"; do case "$v" in ''|*"<"*|*" "*) echo "STOP: fill in NODE1_SSH and NODE2_SSH in $CONF"; exit 2;; esac; done
NODES=(local "$NODE1_SSH" "$NODE2_SSH")
STOP='H=$(hostname); if docker ps -a --format "{{.Names}}" | grep -qx beam-tp3; then T=$(date +%Y%m%d_%H%M%S); W=$(docker inspect -f "{{.State.Running}}" beam-tp3); if [ "$W" = true ]; then docker stop -t 120 beam-tp3 >/dev/null; fi; E=$(docker inspect -f "{{.State.ExitCode}}" beam-tp3); docker logs beam-tp3 > ~/beam_tp3_log_${T}.txt 2>&1; docker rm beam-tp3 >/dev/null; if [ "$W" != true ]; then echo "$H ALREADY_EXITED exit_code=$E log=~/beam_tp3_log_${T}.txt (it had stopped before this run; read the log) $(date +%T)"; elif [ "$E" = 137 ]; then echo "$H KILLED_AFTER_TIMEOUT exit_code=$E log=~/beam_tp3_log_${T}.txt $(date +%T)"; else echo "$H STOPPED_AND_REMOVED exit_code=$E log=~/beam_tp3_log_${T}.txt $(date +%T)"; fi; elif docker info >/dev/null 2>&1; then echo "$H NO_CONTAINER"; else echo "$H DOCKER_ERROR (cannot talk to docker)"; fi'
DROP='sync; if sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"; then echo "$(hostname) PAGE_CACHE_DROPPED"; else echo "$(hostname) DROP_FAILED"; fi; sleep 3'
MEMCHK='A=$(awk "/MemAvailable/{printf \"%d\",\$2/1048576}" /proc/meminfo); G=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .); if [ "$A" -ge 109 ]; then echo "$(hostname) MEM_BACK MEM_AVAILABLE_GIB=$A GPU_PROCS=$G"; else echo "$(hostname) MEM_HELD MEM_AVAILABLE_GIB=$A GPU_PROCS=$G (reboot this node before relaunching)"; fi'
OUT=$(mktemp); FAIL=0
echo "STOP_ALL v1.0.1 $(date -Is)"
echo "== PHASE 1: stopping the container on all nodes at once"
for n in "${NODES[@]}"; do
  if [ "$n" = local ]; then printf '%s\n' "$STOP" | bash | tee -a "$OUT" &
  else { printf '%s\n' "$STOP" | ssh -o BatchMode=yes -o ConnectTimeout=5 "$n" bash -s || echo "$n UNREACHABLE (not stopped)"; } | tee -a "$OUT" & fi
done
while [ "$(jobs -r | wc -l)" -gt 0 ]; do sleep 10; R=$(jobs -r | wc -l); [ "$R" -gt 0 ] && echo "  ... $(date +%T) still stopping on $R node(s)"; done
wait
grep -qE 'UNREACHABLE|KILLED_AFTER_TIMEOUT|DOCKER_ERROR' "$OUT" && FAIL=1
echo "== PHASE 2: free memory on each node (type that node's password if sudo asks)"
for n in "${NODES[@]}"; do
  if [ "$n" = local ]; then bash -c "$DROP" | tee -a "$OUT"; M=$(bash -c "$MEMCHK")
  else { ssh -t -o ConnectTimeout=5 "$n" "$DROP" || echo "$n DROP_FAILED (sudo/ssh step failed)"; } | tee -a "$OUT"; M=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$n" "$MEMCHK" 2>/dev/null) || M="$n UNREACHABLE (memory not checked)"; fi
  echo "$M" | tee -a "$OUT"
done
grep -qE 'MEM_HELD|UNREACHABLE|DROP_FAILED' "$OUT" && FAIL=1
NOTE=$(grep -c ALREADY_EXITED "$OUT" || true)
rm -f "$OUT"
if [ "$FAIL" = 0 ] && [ "$NOTE" = 0 ]; then echo "STOP_ALL_DONE $(date +%T)"; elif [ "$FAIL" = 0 ]; then echo "STOP_ALL_DONE_WITH_NOTES $(date +%T) ($NOTE container(s) had already exited before the stop: read their logs)"; else echo "STOP_ALL_INCOMPLETE $(date +%T) (see the lines above)"; exit 1; fi
