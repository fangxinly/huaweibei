import hashlib
import json
import tarfile
from pathlib import Path

source = Path("outputs/selective_flow_prototype")
target = Path("work/vscode_visible_files.tar.gz")
files = [p for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
manifest = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
Path("work/vscode_files_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
with tarfile.open(target, "w:gz", encoding="utf-8") as archive:
    for p in files:
        archive.add(p, arcname=p.relative_to(source).as_posix())
    archive.add("work/vscode_files_manifest.json", arcname="vscode_files_manifest.json")
print(f"Bundled {len(files)} project files; {target.stat().st_size} bytes")
