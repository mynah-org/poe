#!/usr/bin/env python3
# coding_eval.py — quick behavioral eval for pruned-vs-full MoE checkpoints.
#
# Runs a fixed set of 10 small coding tasks (easy -> hard) against one or
# more GGUF models through llama-server, extracts the generated program,
# actually compiles/runs it on hidden test cases, and reports PASS/FAIL per
# task plus prompt/generation throughput. Zero dependencies beyond the
# Python stdlib, a C compiler, and llama-server.
#
#   python3 tools/coding_eval.py model-a.gguf [model-b.gguf ...]
#
# env: LLAMA_SERVER (default: llama-server on PATH), EVAL_PORT (8089),
#      EVAL_NGL (99), EVAL_MAX_TOKENS (2400), EVAL_CTX (8192)
#
# The point is regression *detection*, not leaderboard accuracy: identical
# prompts, temperature 0, exact-output checkers. A pruned model that loses
# tasks the full model wins is a REAP regression signal.
#
# SPDX-License-Identifier: MIT

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

SERVER = os.environ.get("LLAMA_SERVER", "llama-server")
PORT = int(os.environ.get("EVAL_PORT", "8089"))
NGL = os.environ.get("EVAL_NGL", "99")
MAX_TOKENS = int(os.environ.get("EVAL_MAX_TOKENS", "2400"))
CTX = os.environ.get("EVAL_CTX", "8192")
RUN_TIMEOUT = 10          # seconds per test-case execution
GEN_TIMEOUT = 600         # seconds per completion request

# ── tasks ────────────────────────────────────────────────────────────────
# Each task: name, difficulty, language, prompt, [(stdin, expected_stdout)].
# Prompts pin the I/O contract exactly so a correct program has exactly one
# right output. Checkers compare byte-exact after trailing-space stripping.

CONTRACT = ("Reply with a single fenced code block containing a complete "
            "{lang} program that reads from standard input and writes to "
            "standard output. No explanations outside the code block.")

TASKS = [
    {
        "name": "sum-ints", "diff": "easy", "lang": "python",
        "prompt": "Read one integer per line until EOF and print their sum.",
        "cases": [("1\n2\n3\n", "6"), ("-5\n5\n40\n", "40"), ("7\n", "7")],
    },
    {
        "name": "fizzbuzz", "diff": "easy", "lang": "python",
        "prompt": ("Read an integer N and print the FizzBuzz sequence from 1 "
                   "to N, one entry per line: multiples of 3 print Fizz, of 5 "
                   "print Buzz, of both print FizzBuzz, otherwise the number."),
        "cases": [("5\n", "1\n2\nFizz\n4\nBuzz"),
                  ("15\n", "1\n2\nFizz\n4\nBuzz\nFizz\n7\n8\nFizz\nBuzz\n"
                           "11\nFizz\n13\n14\nFizzBuzz")],
    },
    {
        "name": "reverse-lines", "diff": "easy", "lang": "c",
        "prompt": ("Read lines until EOF and print each line reversed "
                   "(the line's characters in reverse order). Lines are at "
                   "most 1000 characters."),
        "cases": [("abc\nhello\n", "cba\nolleh"), ("a\n\nxy\n", "a\n\nyx")],
    },
    {
        "name": "brackets", "diff": "medium", "lang": "python",
        "prompt": ("Read lines until EOF. Each line is a string of the "
                   "characters ()[]{}. For each line print yes if the "
                   "brackets are balanced and properly nested, else no. "
                   "An empty line is balanced (yes)."),
        "cases": [("([]{})\n([)]\n((\n", "yes\nno\nno"),
                  ("{}\n]\n()[]{}([])\n", "yes\nno\nyes")],
    },
    {
        "name": "mergesort", "diff": "medium", "lang": "python",
        "prompt": ("Read an integer N, then N integers (one per line). Print "
                   "them in non-decreasing order, one per line. Implement "
                   "merge sort yourself; do not call sort() or sorted()."),
        "cases": [("5\n3\n1\n2\n1\n9\n", "1\n1\n2\n3\n9"),
                  ("1\n42\n", "42"),
                  ("6\n-1\n-5\n0\n-5\n7\n2\n", "-5\n-5\n-1\n0\n2\n7")],
    },
    {
        "name": "list-reverse", "diff": "medium", "lang": "c",
        "prompt": ("Read integers until EOF, store them in a singly linked "
                   "list in input order, reverse the list in place by "
                   "re-linking nodes (no arrays), then print the values one "
                   "per line."),
        "cases": [("1\n2\n3\n", "3\n2\n1"), ("10\n", "10"),
                  ("4\n-2\n0\n7\n5\n", "5\n7\n0\n-2\n4")],
    },
    {
        "name": "lru-cache", "diff": "medium", "lang": "python",
        "prompt": ("First line: the cache capacity C. Then commands until "
                   "EOF, one per line: 'put K V' inserts/updates key K with "
                   "value V, 'get K' prints the value of K or -1 if absent. "
                   "Both put and get make the key most-recently used; when a "
                   "put exceeds capacity, evict the least-recently used key."),
        "cases": [("2\nput 1 1\nput 2 2\nget 1\nput 3 3\nget 2\nget 3\n",
                   "1\n-1\n3"),
                  ("1\nput 5 9\nget 5\nput 6 7\nget 5\nget 6\n", "9\n-1\n7")],
    },
    {
        "name": "toposort", "diff": "hard", "lang": "python",
        "prompt": ("First line: N M (vertices 0..N-1, M edges). Then M lines "
                   "'u v' meaning u must come before v. Print a topological "
                   "order, one vertex per line, choosing the smallest "
                   "available vertex first (lexicographically smallest "
                   "order). If the graph has a cycle print only: cycle"),
        "cases": [("4 3\n0 1\n1 2\n0 3\n", "0\n1\n2\n3"),
                  ("3 3\n0 1\n1 2\n2 0\n", "cycle"),
                  ("5 4\n4 0\n3 0\n2 1\n1 4\n", "2\n1\n3\n4\n0")],
    },
    {
        "name": "dijkstra", "diff": "hard", "lang": "python",
        "prompt": ("First line: N M (vertices 0..N-1, M undirected weighted "
                   "edges). Then M lines 'u v w' with positive integer weight "
                   "w. Print the length of the shortest path from vertex 0 "
                   "to vertex N-1, or the word unreachable."),
        "cases": [("4 4\n0 1 1\n1 3 2\n0 2 4\n2 3 1\n", "3"),
                  ("3 1\n0 1 5\n", "unreachable"),
                  ("5 6\n0 1 2\n0 2 9\n1 2 4\n1 3 7\n2 4 3\n3 4 1\n", "9")],
    },
    {
        "name": "lis-nlogn", "diff": "hard", "lang": "python",
        "prompt": ("First line: N (up to 100000). Then N integers, one per "
                   "line. Print the length of the longest strictly "
                   "increasing subsequence. N can be large, so an O(n^2) "
                   "algorithm is too slow — use O(n log n)."),
        "cases": [("6\n3\n1\n2\n5\n4\n6\n", "4"),
                  ("1\n7\n", "1"),
                  ("8\n5\n5\n5\n5\n5\n5\n5\n5\n", "1"),
                  ("\n".join(["20000"] + [str((i * 7919) % 20011)
                                          for i in range(20000)]) + "\n",
                   None)],  # expected computed below (reference LIS)
    },
]


def ref_lis(nums):
    import bisect
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def fill_computed_cases():
    for t in TASKS:
        if t["name"] == "lis-nlogn":
            for i, (stdin, expect) in enumerate(t["cases"]):
                if expect is None:
                    nums = [int(x) for x in stdin.split()[1:]]
                    t["cases"][i] = (stdin, str(ref_lis(nums)))


# ── llama-server client ──────────────────────────────────────────────────

def wait_health(port, proc, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if json.load(r).get("status") == "ok":
                    return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(2)
    return False


def complete(port, prompt):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": MAX_TOKENS, "seed": 42,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=GEN_TIMEOUT) as r:
        resp = json.load(r)
    wall = time.time() - t0
    text = resp["choices"][0]["message"]["content"] or ""
    timings = resp.get("timings", {})
    gen_tps = timings.get("predicted_per_second")
    pp_tps = timings.get("prompt_per_second")
    if gen_tps is None:  # fall back to wall-clock estimate, labeled as such
        toks = resp.get("usage", {}).get("completion_tokens", 0)
        gen_tps = toks / wall if wall > 0 else 0.0
    return text, pp_tps, gen_tps


# ── program extraction and execution ─────────────────────────────────────

FENCE = re.compile(r"```[a-zA-Z0-9+]*\n(.*?)```", re.S)


def extract_code(text):
    blocks = FENCE.findall(text)
    return blocks[-1].strip() + "\n" if blocks else None


def run_program(lang, code, stdin_data, workdir):
    if lang == "python":
        argv = [sys.executable, os.path.join(workdir, "prog.py")]
        with open(argv[1], "w") as f:
            f.write(code)
    else:  # c
        src = os.path.join(workdir, "prog.c")
        exe = os.path.join(workdir, "prog")
        with open(src, "w") as f:
            f.write(code)
        cc = subprocess.run(["cc", "-O2", "-o", exe, src, "-lm"],
                            capture_output=True, text=True, timeout=60)
        if cc.returncode != 0:
            return None, "compile error: " + cc.stderr.strip()[:200]
        argv = [exe]
    try:
        r = subprocess.run(argv, input=stdin_data, capture_output=True,
                           text=True, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if r.returncode != 0:
        return None, "runtime error: " + r.stderr.strip()[:200]
    return r.stdout, None


def norm(s):
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def check_task(task, reply, workdir):
    code = extract_code(reply)
    if code is None:
        return False, "no code block"
    for stdin_data, expect in task["cases"]:
        out, err = run_program(task["lang"], code, stdin_data, workdir)
        if err:
            return False, err
        if norm(out) != norm(expect):
            return False, "wrong output (got %r)" % norm(out)[:60]
    return True, "ok"


# ── driver ───────────────────────────────────────────────────────────────

def eval_model(model_path, port):
    print(f"\n=== {model_path} ===", flush=True)
    proc = subprocess.Popen(
        [SERVER, "-m", model_path, "--port", str(port), "-ngl", NGL,
         "-c", CTX, "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = {}
    tps_all, pp_all = [], []
    try:
        if not wait_health(port, proc):
            print("  server failed to start", flush=True)
            return None, None, None
        for task in TASKS:
            prompt = (task["prompt"] + "\n\n" +
                      CONTRACT.format(lang="C" if task["lang"] == "c"
                                      else "Python 3") + " /no_think")
            try:
                reply, pp, tps = complete(port, prompt)
            except Exception as e:  # noqa: BLE001 - report and continue
                results[task["name"]] = (False, f"request failed: {e}")
                continue
            if tps:
                tps_all.append(tps)
            if pp:
                pp_all.append(pp)
            with tempfile.TemporaryDirectory() as wd:
                ok, why = check_task(task, reply, wd)
            results[task["name"]] = (ok, why)
            print(f"  [{'PASS' if ok else 'FAIL'}] {task['name']:<14}"
                  f" ({task['diff']}, {task['lang']})"
                  f"{'' if ok else '  — ' + why}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    avg = (sum(tps_all) / len(tps_all)) if tps_all else 0.0
    avg_pp = (sum(pp_all) / len(pp_all)) if pp_all else 0.0
    return results, avg, avg_pp


def main():
    models = sys.argv[1:]
    if not models:
        print(__doc__ or "usage: coding_eval.py <model.gguf> [more...]")
        return 2
    if shutil.which(SERVER) is None and not os.path.exists(SERVER):
        print(f"llama-server not found ({SERVER}); set $LLAMA_SERVER")
        return 2
    fill_computed_cases()

    all_results = {}
    for i, m in enumerate(models):
        res, tps, pp = eval_model(m, PORT + i)
        if res is None:
            return 1
        all_results[m] = (res, tps, pp)

    print("\n──── summary ────")
    names = [t["name"] for t in TASKS]
    width = max(len(n) for n in names) + 2
    header = " " * width + "".join(
        f"{os.path.basename(m)[:24]:>26}" for m in models)
    print(header)
    for t in TASKS:
        row = f"{t['name']:<{width}}"
        for m in models:
            ok, _ = all_results[m][0].get(t["name"], (False, "missing"))
            row += f"{'PASS' if ok else 'FAIL':>26}"
        print(row + f"   ({t['diff']})")
    print()
    for m in models:
        res, tps, pp = all_results[m]
        wins = sum(1 for ok, _ in res.values() if ok)
        print(f"{os.path.basename(m)}: {wins}/{len(TASKS)} pass, "
              f"prompt {pp:.0f} t/s, generation {tps:.1f} t/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
