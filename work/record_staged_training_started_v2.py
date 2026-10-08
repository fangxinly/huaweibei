"""Explicit UTF-8 on Windows; original v1 failed before state updates."""
from pathlib import Path
p=Path(__file__).with_name('record_staged_training_started_v1.py')
s=p.read_text(encoding='utf-8').replace('.read_text()',".read_text(encoding='utf-8')")
exec(compile(s,str(p),'exec'))
