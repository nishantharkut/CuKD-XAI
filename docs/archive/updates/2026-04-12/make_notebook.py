"""Convert cukd_xai_colab.py into a Jupyter notebook (.ipynb) for Colab.

!!!! DANGER — READ BEFORE RUNNING !!!!

Running this script OVERWRITES cukd_xai_colab.ipynb and WIPES the Status Banner
cell at the top. The Status Banner is your primary orientation tool inside Colab
and lives ONLY in the .ipynb (not in the .py source).

If you run this by accident, restore the Status Banner by asking Claude, or by
re-reading RESUME_HERE.md (which has the same information).

This script is useful ONLY when:
  (1) You've made major structural changes in the .py file (adding/removing cells)
      AND you're willing to re-apply the Status Banner manually afterwards.
  (2) You're doing a clean rebuild from scratch.

For normal inline code edits (like the April 11 bug fixes), edit the .ipynb
directly via Colab or a Jupyter front-end — do NOT regenerate.

See RESUME_HERE.md for full project context.
"""
import json
import re
import sys

# Safety prompt — require explicit confirmation
print("WARNING: This will OVERWRITE cukd_xai_colab.ipynb and wipe the Status Banner cell.")
print("Type YES (uppercase) to proceed, anything else to abort:")
try:
    response = input().strip()
except EOFError:
    response = ""
if response != "YES":
    print("Aborted — no changes made.")
    sys.exit(0)

with open('cukd_xai_colab.py', 'r') as f:
    source = f.read()

# Split by cell markers
cells_raw = re.split(r'# ={70,}\n# CELL \d+: (.*?)\n# ={70,}\n', source)
# cells_raw[0] = header before first cell, then alternating (title, code, title, code...)

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        },
        "colab": {
            "provenance": []
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

# Add title markdown
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# CuKD-XAI: Curriculum-Guided Knowledge Distillation with Explainability\n",
        "## Lightweight WSN Intrusion Detection\n",
        "\n",
        "**Author:** Nishant Harkut (2023IMG-040), ABV-IIITM Gwalior\n",
        "\n",
        "**Upload WSN-DS.csv to Colab before running.**"
    ]
})

# Process cells
i = 1
while i < len(cells_raw):
    title = cells_raw[i].strip()
    code = cells_raw[i + 1] if (i + 1) < len(cells_raw) else ''

    # Add markdown heading
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {title}"]
    })

    # Add code cell
    # Strip trailing whitespace and make list of lines
    code_lines = code.rstrip().split('\n')
    code_lines = [line + '\n' for line in code_lines[:-1]] + [code_lines[-1]] if code_lines else []

    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": code_lines
    })

    i += 2

# Write notebook
with open('cukd_xai_colab.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Created cukd_xai_colab.ipynb with {len(notebook['cells'])} cells")
