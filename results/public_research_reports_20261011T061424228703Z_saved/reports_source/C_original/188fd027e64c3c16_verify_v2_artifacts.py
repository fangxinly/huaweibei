import hashlib
import json
from pathlib import Path

import numpy as np

p = Path("outputs/selective_flow_prototype/v2")
evaluation = json.loads((p / "test_results.json").read_text(encoding="utf-8"))
selection = json.loads((p / "selection.json").read_text(encoding="utf-8"))
assert evaluation["selection_sha256"] == hashlib.sha256((p / "selection.json").read_bytes()).hexdigest()
archive = np.load(p / "test_predictions.npz", allow_pickle=False)
y = archive["truth"].reshape(-1)
assert len(evaluation["results"]) == 12 and len(y) == 686
assert archive["complete_nonempty_mask"].sum() == 685
for result in evaluation["results"]:
    pred = archive[f"{result['name']}_seed{result['seed']}"]
    assert abs(np.abs(pred - y).mean() - result["test"]["mae"]) < 1e-6
    assert abs(((pred >= 0) == (y >= 0)).mean() - result["test"]["acc2"]) < 1e-9
    assert abs(np.corrcoef(pred, y)[0, 1] - result["test"]["corr"]) < 1e-9
    if result["name"] in ("inv_plain", "inv_flow"):
        assert result["weight"] == selection["selected_weight"]
assert len(list(p.glob("*.pt"))) == 24
for filename in ("结果汇总.md", "summary.json", "diagnostics.png", "diagnostics.pdf", "geometry.json"):
    assert (p / filename).stat().st_size > 0
print("Verified: 12 evaluation records, 24 checkpoints, matching locked selection and saved predictions")
