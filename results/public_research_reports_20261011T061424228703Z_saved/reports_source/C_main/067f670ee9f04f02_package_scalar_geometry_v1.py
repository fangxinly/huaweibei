from pathlib import Path
import argparse,datetime,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args()
read=lambda p:json.loads(Path(p).read_text())
assert read(a.root/"execute_exit.json")["exit_code"]==0
assert read(a.root/"execute/independent_audit.json")["status"]=="ORIGINAL_SCALAR_GEOMETRY_EXECUTE_INDEPENDENT_AUDIT_PASSED"
out=a.root/"preservation";out.mkdir(exist_ok=False)
members={};pending=out/"snapshot.pending"
with zipfile.ZipFile(pending,"x",zipfile.ZIP_DEFLATED) as z:
    for path in sorted(a.root.rglob("*")):
        if not path.is_file() or out in path.parents or "__pycache__" in path.parts or ".pending" in path.name:continue
        if path.suffix not in [".json",".npz",".npy",".py",".log",".txt"]:continue
        name=path.relative_to(a.root).as_posix();data=path.read_bytes()
        z.writestr(name,data);members[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    z.writestr("member_manifest.json",json.dumps(members,indent=2))
with zipfile.ZipFile(pending) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
final=out/"snapshot.zip";pending.replace(final)
r=dict(status="SCALAR_GEOMETRY_ORIGINAL_PACKAGE_COMPLETE",actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    sha256=hashlib.sha256(final.read_bytes()).hexdigest(),bytes=final.stat().st_size,members=len(members),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),old_fullweights_transferred=False)
(out/"receipt.json").write_text(json.dumps(r,indent=2));print(json.dumps(r))

