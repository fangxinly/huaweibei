"""Preserve the failed report operator; read full fixed prediction bytes for arithmetic."""
import argparse, hashlib, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--actualclock', required=True)
a = p.parse_args()
ws = Path(__file__).resolve().parent.parent
pointer = json.loads((ws/'work/official_upgrade_pointer.json').read_text(encoding='utf8'))
d = Path(pointer['D'])
stamp = a.actualclock.replace('-', '').replace(':', '').replace(' UTC', 'Z').replace(' ', 'T')
failure = d / ('report_operator_failure_' + stamp)
failure.mkdir()
original = ws/'work/build_official_upgrade_report.py'
raw = original.read_bytes()
(failure/original.name).write_bytes(raw)
evidence = dict(actualclock_UTC=a.actualclock,stage='local independent report arithmetic',
    original_operator_SHA=hashlib.sha256(raw).hexdigest(),natural_exit=1,
    failure='AssertionError at error<1e-12: CSV float32 decimal rendering changed float64 parsed values by at most 3.869905618181235e-9.',
    remedy='Use SHA-qualified original frozen NPZ; verify CSV round-trips to identical float32 bits, then aggregate original arrays.',
    no_model_forward_or_training_or_original_scoring_rerun=True,
    original_scoring_and_frozen_reporting_spec_unchanged=True)
(failure/'actual_failure.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf8')
source = raw.decode('utf8')
before = "    training = read(Path(pointer['D'])/'A_train/extracted_small/out/training_result.json')"
after = """    frozen_path = Path(pointer['D'])/'A_infer/extracted/out/fixed_official_VAL_TEST_prediction.npz'
    assert sha(frozen_path) == actual['prediction_SHA']
    frozen = np.load(frozen_path, allow_pickle=False)
    training = read(Path(pointer['D'])/'A_train/extracted_small/out/training_result.json')"""
assert source.count(before)==1
source = source.replace(before,after)
before = "        new=np.asarray([float(r['new']) for r in rows]);off=np.asarray([float(r['messages_off']) for r in rows])"
after = """        csvnew=np.asarray([float(r['new']) for r in rows]);csvoff=np.asarray([float(r['messages_off']) for r in rows])
        assert ids.tolist()==frozen[role+'_ids'].tolist()
        assert np.array_equal(csvnew.astype(np.float32),frozen[role+'_prediction'])
        assert np.array_equal(csvoff.astype(np.float32),frozen[role+'_p0'])
        new=frozen[role+'_prediction'].astype(float);off=frozen[role+'_p0'].astype(float)"""
assert source.count(before)==1
source = source.replace(before,after)
source = source.replace("baseline_original_bytes=verified,", "baseline_original_bytes=verified,\n        frozen_prediction_source=dict(path=str(frozen_path),SHA=sha(frozen_path),CSV_float32_roundtrip_exact=True),\n        reporting_operator_repair=read(Path('"+str(failure/'actual_failure.json').replace('\\','/')+"')),")
source = source.replace("完整最终/最佳model、Adam4000、scheduler、RNG、100轮预测和原源码已跨D/C分片保存", "完整最终/最佳model、Adam4000、scheduler、RNG、100轮预测和原源码已以两份分片完整保存到D；C预留分片位置未使用")
source = source.replace("完整五折和五项超过的整体目标仍未完成。", "完整五折和五项超过的整体目标仍未完成。CSV默认float32十进制输出会引入极小舍入差，独立算术和bootstrap采用SHA核验的原始NPZ，CSV还原float32后逐值完全相同。")
destination = ws/'work'/('build_official_upgrade_report_fullprecision_'+stamp+'.py')
with destination.open('x',encoding='utf8') as f:f.write(source)
print(json.dumps(dict(new_operator=str(destination),failure_evidence=str(failure),new_operator_SHA=hashlib.sha256(destination.read_bytes()).hexdigest()),ensure_ascii=False))
