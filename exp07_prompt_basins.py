#!/usr/bin/env python3
"""Exp 07 — prompt-space basin maps for an ordinary LLM (API, DeepSeek V4 Flash).

For a few fixed 4x4 sudoku puzzles, generate many SEMANTICALLY IDENTICAL
prompt variants (instruction phrasings x grid presentation formats), solve
each at temperature 0, and record (final answer identity, token count).
If answers organize into coherent interlocking regions of prompt space with
fractal-ish boundaries, that is the ordinary-LLM analog of our latent-slice
basin maps.

Output: runs/exp07/prompt_basins.json + a map figure per puzzle.
"""
import json, os, time
from pathlib import Path
from collections import Counter

import numpy as np
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp07"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("FB_LLM", "deepseek/deepseek-v4.1-flash")

PUZZLES = {  # 4x4 sudoku, 0 = empty; rows of 4
    "p4_easy": ["1000", "0230", "0302", "0010"],
    "p4_med": ["0300", "4000", "0003", "0020"],
    "p4_hard": ["0003", "0200", "0000", "3000"],
}

PHRASINGS = [
    "Solve this 4x4 sudoku (digits 1-4, each once per row, column, and 2x2 box). 0 means empty. Return ONLY the four solved rows as digit strings, one per line.",
    "Here is a mini sudoku puzzle. Fill in the blanks (0) so every row, column and 2x2 box contains 1,2,3,4 exactly once. Output just the completed grid, four lines of four digits.",
    "Complete the following 4x4 Latin-square-style puzzle (each digit 1-4 once per row, column and 2x2 block). Reply with the four rows only, no commentary.",
    "This is a 4x4 sudoku. Digits 1-4, rows/columns/2x2 boxes. Replace each 0. Answer format: four lines, four digits each, nothing else.",
]
FORMATS = [
    lambda rows: "\n".join(rows),
    lambda rows: "\n".join(" ".join(r) for r in rows),
    lambda rows: "\n".join(",".join(r) for r in rows),
    lambda rows: "[" + ", ".join(r for r in rows) + "]",
    lambda rows: "\n".join(f"row {i+1}: {r}" for i, r in enumerate(rows)),
]


def variants():
    for phr in PHRASINGS:
        for fmt in FORMATS:
            for pi, (pid, rows) in enumerate(PUZZLES.items()):
                yield pid, f"{phr}\n\n{fmt(rows)}"


def solve(prompt, retries=2):
    key = os.environ["OPENROUTER_API_KEY"]
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": {"enabled": False, "exclude": True},
        "max_tokens": 1500, "temperature": 0.0}).encode()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions", data=body,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=90))
            msg = r["choices"][0]["message"]["content"]
            usage = r.get("usage", {})
            return msg, usage.get("completion_tokens", 0)
        except Exception as e:
            if attempt == retries:
                return f"__ERROR__ {e}", 0
            time.sleep(2 * (attempt + 1))


def parse_grid(text):
    digits = []
    for line in text.strip().splitlines():
        ds = "".join(ch for ch in line if ch in "1234")
        if len(ds) == 4:
            digits.append(ds)
        if len(digits) == 4:
            break
    return tuple(digits) if len(digits) == 4 else ("__UNPARSED__",)


def check(rows):
    for r in rows:
        if sorted(r) != list("1234"):
            return False
    for c in range(4):
        if sorted(rows[r][c] for r in range(4)) != list("1234"):
            return False
    for br in (0, 2):
        for bc in (0, 2):
            if sorted(rows[br+i][bc+j] for i in range(2) for j in range(2)) != list("1234"):
                return False
    return True


def solve4x4(rows):
    """Brute-force solver: returns True if the puzzle has a solution."""
    g = [list(r) for r in rows]
    def ok(r, c, v):
        for i in range(4):
            if g[r][i] == v or g[i][c] == v:
                return False
        br, bc = 2 * (r // 2), 2 * (c // 2)
        for i in range(2):
            for j in range(2):
                if g[br + i][bc + j] == v:
                    return False
        return True
    def rec(pos=0):
        while pos < 16 and g[pos // 4][pos % 4] != "0":
            pos += 1
        if pos == 16:
            return True
        r, c = pos // 4, pos % 4
        for v in "1234":
            if ok(r, c, v):
                g[r][c] = v
                if rec(pos + 1):
                    return True
                g[r][c] = "0"
        return False
    return rec()


def main():
    PUZZLES = {k: v for k, v in globals()["PUZZLES"].items() if solve4x4(v)}
    print("solvable puzzles:", list(PUZZLES))
    results = []
    vs = list(variants())
    print(f"{len(vs)} prompt variants x {len(PUZZLES)} puzzles", flush=True)
    for i, (pid, prompt) in enumerate(vs):
        text, toks = solve(prompt)
        grid = parse_grid(text)
        ok = grid != ("__UNPARSED__",) and check(grid)
        results.append(dict(variant=i, puzzle=pid, prompt=prompt,
                            grid=list(grid), valid=ok, tokens=toks,
                            raw=text[:120]))
        if i % 10 == 0:
            print(f"[{i+1}/{len(vs)}] {pid} -> {grid} valid={ok} tok={toks}", flush=True)
    (OUT / "prompt_basins.json").write_text(json.dumps(results, indent=1))

    # analysis: per puzzle, distinct answer identities + token-count spread
    for pid in PUZZLES:
        rs = [r for r in results if r["puzzle"] == pid]
        ids = Counter(tuple(r["grid"]) for r in rs)
        toks = [r["tokens"] for r in rs if r["valid"]]
        print(f"\n{pid}: {len(rs)} variants, {len(ids)} distinct answers "
              f"(top: {ids.most_common(3)}), valid {sum(r['valid'] for r in rs)}, "
              f"tokens min/med/max = {min(toks)}/{int(np.median(toks))}/{max(toks)}"
              if toks else f"{pid}: no valid answers")


if __name__ == "__main__":
    main()
