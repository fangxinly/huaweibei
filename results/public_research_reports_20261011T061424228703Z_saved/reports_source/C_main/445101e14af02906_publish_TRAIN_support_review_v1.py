import datetime, hashlib, json, os, pathlib, shutil, subprocess, sys
P = pathlib.Path
sys.path.insert(0, str(P(__file__).parent))
from prepare_github_source_upload import sanitize
from raw_TRAIN_capture_v1 import raw, sha, seal

r = P(sys.argv[1])
dc = r.parent.parent / 'candidate_posttrain_lowC_20261009T005229Z'
c = P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
assert shutil.disk_usage('C:/').free > 200 * 1024**2
assert shutil.disk_usage('D:/').free > 40 * 1024**2
pub = json.loads((r / 'Release_receipt.json').read_bytes())
proof = json.loads((r / 'D_preservation_receipt.json').read_bytes())
assert pub['remote_digest_verified'] and pub['source_SHA'] == proof['archive_SHA']
assert sha(r / 'complete_actual_support_independent_review_original.zip') == proof['archive_SHA']
state = json.loads((dc / 'D_current_research_state.json').read_bytes())
assert P(state['latest_TRAIN_support_complete_review']['root']) == r
assert state['latest_TRAIN_support_analysis_sent']['full_reply_received']
git = 'C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
bare = dc / 'publication.git'
def cmd(args, data=None):
    return subprocess.check_output([git, '--git-dir=' + str(bare)] + args, input=data, cwd=r)

out = r / 'public_source_and_summary'
out.mkdir()
names = ['TRAIN支持范围诊断_独立数学与科学边界_20261010.md',
         'TRAIN支持诊断独立审阅_20261010.json', 'independent_decision.json',
         'audit_saved_TRAIN_support_batch_20261010.py',
         'save_TRAIN_support_complete_review_v1.py', 'D_preservation_receipt.json',
         'Release_receipt.json']
for n in names:
    shutil.copyfile(r / n, out / n)
shutil.copyfile(__file__, out / P(__file__).name)
old = cmd(['rev-parse', 'HEAD']).decode().strip()
assert old == cmd(['ls-remote', 'origin', 'refs/heads/main']).decode().split()[0]
cmd(['read-tree', 'HEAD'])
manifest = json.loads(cmd(['show', 'HEAD:source_manifest.json']))
records = {v['path']: v for v in manifest['records']}
files = sorted(out.iterdir())
for f in files:
    rel = 'results/TRAIN_support_independent_review_20261010/' + f.name
    original = f.read_bytes()
    data, changes = sanitize(original)
    oid = cmd(['hash-object', '-w', '--stdin'], data).decode().strip()
    cmd(['update-index', '--add', '--cacheinfo', '100644', oid, rel])
    assert cmd(['show', ':' + rel]) == data
    records[rel] = dict(path=rel, origin=str(f), kind='saved_support_independent_review',
                        bytes=len(data), original_sha256=hashlib.sha256(original).hexdigest(),
                        published_sha256=hashlib.sha256(data).hexdigest(), sanitizations=changes)
manifest['records'] = [records[k] for k in sorted(records)]
data = raw(manifest)
oid = cmd(['hash-object', '-w', '--stdin'], data).decode().strip()
cmd(['update-index', '--add', '--cacheinfo', '100644', oid, 'source_manifest.json'])
assert cmd(['show', ':source_manifest.json']) == data
tree = cmd(['write-tree']).decode().strip()
commit = cmd(['-c', 'user.name=Codex', '-c', 'user.email=codex@users.noreply.github.com',
              'commit-tree', tree, '-p', old, '-m',
              'Preserve independent TRAIN support review and provenance limits']).decode().strip()
cmd(['update-ref', 'refs/heads/main', commit, old])
cmd(['push', 'origin', 'refs/heads/main:refs/heads/main'])
assert cmd(['ls-remote', 'origin', 'refs/heads/main']).decode().split()[0] == commit
receipt = dict(status='D_BARE_TRAIN_SUPPORT_REVIEW_EXACT_BLOB_GITHUB_VERIFIED',
               actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               commit=commit, parent_commit=old, C_checkout_not_modified=True,
               D_bare=str(bare), files=len(files)+1, all_published_blobs_verified=True,
               primary_results_unchanged=True, full_neural_video_fivefold_complete=False)
for p in [dc / 'GitHub_publication_receipt.json', r / 'GitHub_publication_receipt.json']:
    p.write_bytes(raw(receipt))
state['github_source'] = receipt
state['latest_TRAIN_support_complete_review']['Release_publication'] = pub
state['latest_TRAIN_support_complete_review']['GitHub_publication'] = receipt
state['updated_at_utc'] = receipt['actual_UTC']
for dest in [dc / 'D_current_research_state.json', c / '自主优化实际接续.json']:
    tmp = dest.with_name(dest.name + '.reviewpublished.tmp')
    tmp.write_bytes(raw(state))
    os.replace(tmp, dest)
assert (dc / 'D_current_research_state.json').read_bytes() == (c / '自主优化实际接续.json').read_bytes()
sync = dict(actual_UTC=receipt['actual_UTC'], D_C_same_SHA=sha(dc / 'D_current_research_state.json'),
            GitHub_commit=commit, Release_remote_digest_verified=True,
            advisor_complete_no_poll_or_resend=True)
(r / 'publication_and_state_sync_receipt.json').write_bytes(raw(sync))
with (c / '研究接续状态.md').open('a', encoding='utf8') as f:
    f.write('\n' + receipt['actual_UTC'] + ' 完整TRAIN支持独立审阅ZIP744d7ebb的Release远端digest及D bare exact blob GitHub ' + commit + ' 实际过，D/C同字节；完整建议已决定，不再轮询重发。原诊断/预测/五项/once均不重做，整体研究仍未完成。\n')
closure = r / 'publication_closure'
closure.mkdir()
for n in ['GitHub_publication_receipt.json', 'publication_and_state_sync_receipt.json', 'Release_receipt.json']:
    shutil.copyfile(r / n, closure / n)
shutil.copyfile(__file__, closure / P(__file__).name)
closed = seal(closure, 'complete_actual_review_publication_closure_original.zip')
(r / 'publication_closure_D_receipt.json').write_bytes(raw(closed))
print(json.dumps(dict(receipt=receipt, closure=closed)))
