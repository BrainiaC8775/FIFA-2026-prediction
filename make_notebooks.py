"""
make_notebooks.py
=================
Converts every .py file in this directory into a Colab-ready .ipynb notebook.

Cell splitting strategy
-----------------------
  • Lines matching  # ─────────────────  (20+ dashes) → new cell boundary
  • The first comment line after a boundary → markdown section header
  • Module docstrings → markdown intro cell
  • `if __name__ == "__main__":` guard → stripped; body dedented

Run with any Python 3.x:
    python make_notebooks.py
"""

import json, re, os, textwrap

# ─────────────────────────────────────────────────────────────
# COLAB SETUP CELL  (prepended to every notebook)
# ─────────────────────────────────────────────────────────────

SETUP_CELL = '''\
# ═══════════════════════════════════════════════════════════════
#  COLAB SETUP  — run this cell first
# ═══════════════════════════════════════════════════════════════

# 1. Install dependencies
import subprocess, sys
for pkg in ["catboost", "pandas", "numpy", "scipy", "scikit-learn"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print("✓ packages ready")

# 2. Mount Google Drive and set working directory
try:
    from google.colab import drive
    drive.mount("/content/drive")
    import os
    # ⚠️  Update this path to match your Drive folder
    DATA_DIR = "/content/drive/MyDrive/football_prediction_model/FIFA-2026-prediction"
    os.chdir(DATA_DIR)
    print(f"✓ working directory: {os.getcwd()}")
except ImportError:
    # Running locally — no Drive mount needed
    import os
    print(f"✓ local run, cwd: {os.getcwd()}")
'''

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

DIVIDER_RE = re.compile(r"^#\s*[─━═]{20,}")
MAIN_RE    = re.compile(r"^if\s+__name__\s*==\s*[\"']__main__[\"']\s*:")


def strip_main_guard(text: str) -> str:
    """Remove `if __name__ == '__main__':` wrapper and dedent its body."""
    lines = text.split("\n")
    out, in_guard = [], False
    for line in lines:
        if MAIN_RE.match(line):
            in_guard = True
            continue
        if in_guard:
            if line == "" or line.startswith("    ") or line.startswith("\t"):
                out.append(line[4:] if line.startswith("    ") else
                           line[1:] if line.startswith("\t") else line)
            else:
                in_guard = False
                out.append(line)
        else:
            out.append(line)
    return "\n".join(out)


def extract_docstring(text: str):
    """Return the module-level triple-quoted docstring, or None."""
    m = re.match(r'\s*"""(.*?)"""', text, re.DOTALL)
    return m.group(1).strip() if m else None


def split_into_sections(text: str):
    """
    Split on long-dash dividers.
    Returns list of (header: str | None, code: str).
    """
    segments, current, pending_header = [], [], None
    lines = text.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]
        if DIVIDER_RE.match(line):
            # Flush current block
            block = "\n".join(current).strip()
            if block:
                segments.append((pending_header, block))
            current = []
            pending_header = None
            # Peek at next non-blank line for a section title
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and DIVIDER_RE.match(lines[j]):
                # Another divider immediately — skip both
                i = j + 1
                continue
            if j < len(lines) and lines[j].startswith("#"):
                # Use this as the section header
                raw = lines[j].lstrip("# ").rstrip()
                if raw and len(raw) < 120:
                    pending_header = raw
        else:
            current.append(line)
        i += 1

    block = "\n".join(current).strip()
    if block:
        segments.append((pending_header, block))

    return [(h, c) for h, c in segments if c.strip()]


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def make_notebook(py_path: str) -> str:
    name  = os.path.basename(py_path)
    stem  = name.replace(".py", "")
    title = stem.replace("_", " ").title()

    with open(py_path, encoding="utf-8") as f:
        raw = f.read()

    docstring = extract_docstring(raw)
    text      = strip_main_guard(raw)
    sections  = split_into_sections(text)

    cells = []

    # ── Title / description ──────────────────────────────────────
    intro = f"# {title}\n\n"
    if docstring:
        intro += docstring.replace("=", "─").strip()
    else:
        intro += f"Converted from `{name}` — Google Colab ready."
    cells.append(md_cell(intro))

    # ── Colab setup ───────────────────────────────────────────────
    cells.append(code_cell(SETUP_CELL))

    # ── Content sections ──────────────────────────────────────────
    for header, code in sections:
        # Skip if this block is just the module docstring (already shown)
        if docstring and code.lstrip().startswith('"""') and docstring[:40] in code:
            continue
        if header:
            cells.append(md_cell(f"## {header}"))
        cells.append(code_cell(code))

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0",
            },
            "colab": {"provenance": [], "toc_visible": True},
        },
        "cells": cells,
    }

    nb_path = py_path.replace(".py", ".ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    return nb_path


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    py_files = sorted(
        f for f in os.listdir(base)
        if f.endswith(".py") and f != "make_notebooks.py"
    )

    print(f"Converting {len(py_files)} Python files -> Jupyter notebooks\n")
    ok = err = 0
    for fname in py_files:
        src = os.path.join(base, fname)
        try:
            nb = make_notebook(src)
            nb_name = os.path.basename(nb)
            print(f"  OK  {fname:<35}  ->  {nb_name}")
            ok += 1
        except Exception as exc:
            print(f"  ERR {fname:<35}  ->  ERROR: {exc}")
            err += 1

    print(f"\nDone — {ok} converted, {err} errors.")
    if ok:
        print("\nColab usage:")
        print("  1. Upload the .ipynb files (and any .py files they import) to Google Drive")
        print("  2. Open a notebook in Colab")
        print("  3. Update DATA_DIR in the Setup cell to your Drive path")
        print("  4. Run all cells top-to-bottom")
