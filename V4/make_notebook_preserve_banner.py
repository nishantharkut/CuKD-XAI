"""Safely regenerate cukd_xai_colab.ipynb from cukd_xai_colab.py.

This preserves the hand-written top notebook orientation/banner cells from the
existing notebook. The older converter under Update - 12 april 2026 intentionally
warns that it wipes those cells; use this script for the final project route.
"""
import json
import re
import shutil
from pathlib import Path

SOURCE = Path('cukd_xai_colab.py')
NOTEBOOK = Path('cukd_xai_colab.ipynb')
BACKUP = Path('cukd_xai_colab.ipynb.bak')


def source_to_cells(source_text: str):
    parts = re.split(r'# ={70,}\n# CELL \d+: (.*?)\n# ={70,}\n', source_text)
    cells = []
    i = 1
    while i < len(parts):
        title = parts[i].strip()
        code = parts[i + 1] if i + 1 < len(parts) else ''
        cells.append({
            'cell_type': 'markdown',
            'metadata': {},
            'source': [f'## {title}'],
        })
        code_lines = code.rstrip().split('\n')
        if code_lines:
            code_source = [line + '\n' for line in code_lines[:-1]] + [code_lines[-1]]
        else:
            code_source = []
        cells.append({
            'cell_type': 'code',
            'metadata': {},
            'execution_count': None,
            'outputs': [],
            'source': code_source,
        })
        i += 2
    return cells


def cell_text(cell):
    return ''.join(cell.get('source', []))


def preserved_prefix(existing_nb):
    cells = existing_nb.get('cells', [])
    prefix = []
    for cell in cells:
        text = cell_text(cell).strip()
        if text == '## Install dependencies':
            break
        prefix.append(cell)
    if not prefix:
        prefix = [{
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# CuKD-XAI: Curriculum-Guided Knowledge Distillation with Explainability\n',
                '## Lightweight WSN Intrusion Detection\n',
                '\n',
                '**Author:** Nishant Harkut (2023IMG-040), ABV-IIITM Gwalior\n',
                '\n',
                '**Upload WSN-DS.csv to Colab before running.**',
            ],
        }]
    return prefix


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not NOTEBOOK.exists():
        raise FileNotFoundError(NOTEBOOK)

    existing = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    generated_cells = source_to_cells(SOURCE.read_text(encoding='utf-8'))
    nb = {
        'cells': preserved_prefix(existing) + generated_cells,
        'metadata': existing.get('metadata', {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.10'},
            'colab': {'provenance': []},
            'accelerator': 'GPU',
        }),
        'nbformat': existing.get('nbformat', 4),
        'nbformat_minor': existing.get('nbformat_minor', 0),
    }

    shutil.copy2(NOTEBOOK, BACKUP)
    NOTEBOOK.write_text(json.dumps(nb, indent=1), encoding='utf-8')
    print(f'Wrote {NOTEBOOK} with {len(nb["cells"])} cells')
    print(f'Backup saved to {BACKUP}')


if __name__ == '__main__':
    main()
