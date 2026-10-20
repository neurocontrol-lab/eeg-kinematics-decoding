"""Validate current project navigation, syntax, and configured local data."""
import argparse
import ast
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from eeg_kinematics.paths import RAW_DATA_DIR, REDUCED_DATA_DIR

def check():
    errors=[];links=0;scripts=0;cells=0
    files=[ROOT/'README.md', ROOT/'CHANGELOG.md']
    for folder in ['documentation','experiments','presentations','src','scripts','tests']:
        files += list((ROOT/folder).rglob('*'))
    files += [f for f in (ROOT/'outputs').rglob('*.md') if f.name!='declared_protocol.md']
    for f in files:
        if not f.is_file() or any(p in {'__pycache__','exports'} for p in f.relative_to(ROOT).parts):continue
        if f.suffix=='.py':
            try:ast.parse(f.read_text(encoding='utf-8'),filename=str(f));scripts+=1
            except SyntaxError as e:errors.append(str(e))
        if f.suffix=='.ipynb':
            n=json.loads(f.read_text(encoding='utf-8'));texts=[]
            for c in n['cells']:
                source=''.join(c.get('source',[]))
                if c['cell_type']=='markdown':texts.append(source)
                elif source.strip() and not any(line.lstrip().startswith(('!','%')) for line in source.splitlines()):
                    try:ast.parse(source);cells+=1
                    except SyntaxError as e:errors.append(f'{f}: {e}')
            text='\n'.join(texts)
        elif f.suffix=='.md':text=f.read_text(encoding='utf-8')
        else:continue
        for dest in re.findall(r'\]\(([^\s)]+)\)',text):
            if re.match(r'^[a-zA-Z]+:|^#|^/',dest):continue
            target=unquote(dest.split('#')[0]);path=(f.parent/target).resolve()
            if not path.exists():errors.append(f'{f.relative_to(ROOT)} -> {dest}')
            links+=1
    if not RAW_DATA_DIR.is_dir():errors.append(f'Missing raw data: {RAW_DATA_DIR}')
    if not REDUCED_DATA_DIR.is_dir():errors.append(f'Missing reduced data: {REDUCED_DATA_DIR}')
    for i in range(1,13):
        for f in [RAW_DATA_DIR/f'P{i}/P{i}_AllLifts.mat',REDUCED_DATA_DIR/f'eeg_P{i}.mat',REDUCED_DATA_DIR/f'kin_P{i}.mat',REDUCED_DATA_DIR/f'session_start_P{i}.mat']:
            if not f.is_file():errors.append(f'Missing required input: {f}')
    print(json.dumps({'local_links_checked':links,'python_files_checked':scripts,'notebook_cells_checked':cells,'errors':errors},indent=2))
    return not errors

if __name__=='__main__':sys.exit(0 if check() else 1)
