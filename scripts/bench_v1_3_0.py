# SPDX-License-Identifier: Apache-2.0
# bench_v1_3_0.py - before/after benchmark for one Beam TP=3 restart (run on node0 with the server idle).
# v1.3.0 = v1.2.0 that stops with BENCH=FAIL (exit 1) instead of reporting a time when a T3 or T4 request fails, and
#          gives up on T3 after 10 minutes if the two chats never start writing; tests and timing unchanged.
# v1.2.0 = the v1.1.0 we ran for the v2 results, with the three nodes' SSH targets read from BENCH_NODES instead of
# being written in; the tests and their timing are unchanged.
# Four tests, each on a FRESH document (nothing reused from cache), plus free memory on every node every 5 s:
#  T1 lone read      : one ~100K-token prompt, max_tokens=1 -> time to read it
#  T2 single chat    : 300 tokens on a short prompt -> writing speed
#  T3 chats + read   : two chats writing, then a ~100K-token prompt arrives -> chat speed during the read, read time
#  T4 short behind   : a ~100K-token prompt, then a one-line question 2 s later -> how long the short question waits
# Usage: BENCH_NODES="node0=local node1=<user>@<node1> node2=<user>@<node2>" python3 -I bench_v1_3_0.py <label>
#        (BENCH_NODES unset: node0 only.) API defaults to http://127.0.0.1:8600; override with BENCH_API.
import json, os, random, subprocess, sys, threading, time, urllib.request as u
API = os.environ.get("BENCH_API", "http://127.0.0.1:8600"); LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"; T0 = time.time()
HOSTS = [(p.split("=", 1)[0], None if p.split("=", 1)[1] == "local" else p.split("=", 1)[1]) for p in (os.environ.get("BENCH_NODES") or "node0=local").split() if "=" in p] or [("node0", None)]
MEM = {h: [] for h, _ in HOSTS}; STOP = [False]
def post(path, body):
    try:
        return json.load(u.urlopen(u.Request(API + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1200))
    except Exception as e:
        print("HTTP_ERROR %s %s %s" % (getattr(e, "code", "-"), path, (e.read().decode(errors="replace") if hasattr(e, "read") else str(e))[:400]), flush=True); return None
def stream(prompt, max_tokens, stamps):
    body = {"model": "beam", "max_tokens": max_tokens, "ignore_eos": True, "temperature": 0, "stream": True, "messages": [{"role": "user", "content": prompt}]}
    try:
        r = u.urlopen(u.Request(API + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1200)
        for line in r:
            line = line.decode(errors="replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            try: d = json.loads(line[5:])
            except Exception: continue
            ch = d.get("choices") or []
            if ch and any(isinstance(v, str) and v for k, v in (ch[0].get("delta") or {}).items() if k != "role"): stamps.append(time.time())
    except Exception as e:
        print("STREAM_ERROR %s" % str(e)[:200], flush=True)
def mem_once():
    cmd = "awk '/MemAvailable/{printf \"%.2f\", $2/1048576}' /proc/meminfo"
    for h, ssh in HOSTS:
        try:
            out = subprocess.run((["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", ssh, cmd] if ssh else ["sh", "-c", cmd]), capture_output=True, text=True, timeout=8).stdout.strip()
            MEM[h].append(float(out))
        except Exception: pass
def sampler():
    n = 0
    while not STOP[0]:
        mem_once(); n += 1
        if n % 4 == 0: print("  ... %s %.0f s  MemAvailable GiB: %s" % (time.strftime("%H:%M:%S"), time.time() - T0, "  ".join("%s=%.2f" % (h, MEM[h][-1]) for h, _ in HOSTS if MEM[h])), flush=True)
        time.sleep(5)
r = random.Random(5); w = "pressure valve filter gauge manifold compressor coolant bearing gasket torque".split()
secs = ["Section %d. The %s must be checked before the %s is replaced." % (i, r.choice(w), r.choice(w)) for i in range(1, 20001)]
n = 5579
for k in range(4):
    j = post("/tokenize", {"model": "beam", "prompt": "Bench x. " + " ".join(secs[:n])})
    if not j: break
    c = j["count"]; print("SIZING sections=%d tokens=%d" % (n, c), flush=True)
    if 95000 <= c <= 102000: break
    n = min(int(n * 99000 / c), len(secs))
BODY = " ".join(secs[:n])
def doc(tag): return "Bench %s %s %x. " % (LABEL, tag, int(time.time() * 1e6)) + BODY + "\n\nQuestion: Which section first mentions a gasket?"
def one(prompt, mt=1):
    t = time.time(); j = post("/v1/chat/completions", {"model": "beam", "max_tokens": mt, "ignore_eos": True, "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
    return (time.time() - t, j)
def die(why): STOP[0] = True; print("BENCH=FAIL %s" % why, flush=True); sys.exit(1)
threading.Thread(target=sampler, daemon=True).start(); time.sleep(1)
print("== T1 lone read (%s)" % time.strftime("%H:%M:%S"), flush=True)
d1, j = one(doc("T1"))
if not j: STOP[0] = True; print("BENCH=FAIL T1"); sys.exit(1)
pt = j["usage"]["prompt_tokens"]; print("T1_LONE_READ %s prompt_tokens=%d secs=%.1f tok_per_s=%.0f" % (LABEL, pt, d1, pt / d1), flush=True); time.sleep(3)
print("== T2 single chat (%s)" % time.strftime("%H:%M:%S"), flush=True)
s2 = []; t = time.time(); stream("Bench %s T2 %x. Describe, in detail, a yearly maintenance schedule for an industrial air compressor." % (LABEL, int(t * 1e6)), 300, s2)
if len(s2) > 20: print("T2_SINGLE_CHAT %s tokens=%d first_token_secs=%.1f tok_per_s=%.1f" % (LABEL, len(s2), s2[0] - t, (len(s2) - 1) / (s2[-1] - s2[0])), flush=True)
else: print("T2_SINGLE_CHAT %s FAIL tokens=%d" % (LABEL, len(s2)), flush=True)
time.sleep(3)
print("== T3 two chats writing, then a long read arrives (%s)" % time.strftime("%H:%M:%S"), flush=True)
c1, c2 = [], []
th = [threading.Thread(target=stream, args=("Bench %s T3 chat%d %x. Write a long, detailed essay on the history of steam engines." % (LABEL, i, int(time.time() * 1e6)), 600, s)) for i, s in ((1, c1), (2, c2))]
for x in th: x.start()
t3w = time.time()
while (len(c1) < 30 or len(c2) < 30) and time.time() - t3w < 600 and any(x.is_alive() for x in th): time.sleep(0.2)
if len(c1) < 30 or len(c2) < 30: die("T3 the two chats did not start writing (tokens %d / %d)" % (len(c1), len(c2)))
tb = time.time(); before = sum((len(s) - 1) / max(s[-1] - s[0], 1e-6) for s in (c1, c2)) / 2
d3, j3 = one(doc("T3")); te = time.time()
if not j3: die("T3 long read failed")
during = sum(1 for s in c1 + c2 if tb <= s < te) / max(te - tb, 1e-6) / 2
print("T3_CHATS_DURING_READ %s per_chat_tok_per_s_before=%.1f during=%.2f read_secs=%.1f" % (LABEL, before, during, d3), flush=True)
for x in th: x.join()
time.sleep(3)
print("== T4 short question behind a long read (%s)" % time.strftime("%H:%M:%S"), flush=True)
res = {}
def longdoc(): res["doc"], res["j"] = one(doc("T4"))
tl = threading.Thread(target=longdoc); tl.start(); time.sleep(2)
d4, j4 = one("Bench %s T4 %x. What is 2 + 2? Answer with the number only." % (LABEL, int(time.time() * 1e6)))
tl.join()
if not j4 or not res.get("j"): die("T4 a request failed (short ok=%s, long ok=%s)" % (bool(j4), bool(res.get("j"))))
print("T4_SHORT_BEHIND_READ %s short_secs=%.1f long_read_secs=%.1f" % (LABEL, d4, res.get("doc", -1)), flush=True)
STOP[0] = True; time.sleep(1)
print("MIN_MEMAVAILABLE_GIB %s" % "  ".join("%s=%.2f" % (h, min(MEM[h])) if MEM[h] else "%s=n/a" % h for h, _ in HOSTS))
print("BENCH_DONE %s" % LABEL)
