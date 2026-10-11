"""Strengthen label separation before any GPU deployment; retain v1 preparation."""
from pathlib import Path
work = Path(__file__).resolve().parent
source = (work/'diagnose_train_oracle_v1.py').read_text(encoding='utf-8')
source = source.replace('terminal, y):', 'terminal, y=None):', 1)
start = source.index('        # Select oracle candidates in double')
end = source.index('        shift = value - base', start)
block = source[start:end]
source = source[:start]+'        if y is not None:\n'+''.join('    '+line+'\n' for line in block.splitlines())+source[end:]
source = source.replace("    with torch.no_grad():\n        best_prediction = terminal(feedback.context(old, best * scale))\n    return dict(", "    best_values = {}\n    if y is not None:\n        with torch.no_grad():\n            best_prediction = terminal(feedback.context(old, best * scale))\n        best_values = dict(best_true_prediction=best_prediction, best_true_step=best_index)\n    return dict(")
source = source.replace('                best_true_prediction=best_prediction, best_true_step=best_index)', '                **best_values)')
source = source.replace("result = solve(learner.feedback, old, messages, p0, rho.detach(), terminal, y)", "result = solve(learner.feedback, old, messages, p0, rho.detach(), terminal, y if name == 'oracle' else None)")
source = source.replace("            if name != 'oracle':\n                result.pop('best_true_prediction'); result.pop('best_true_step')\n", '')
target = work/'diagnose_train_oracle_v2.py'; assert not target.exists(); target.write_text(source,encoding='utf-8')
freeze = (work/'freeze_train_oracle_plan_v1.py').read_text(encoding='utf-8')
freeze = freeze.replace('diagnose_train_oracle_v1.py','diagnose_train_oracle_v2.py').replace('train_oracle_plan_v1.json','train_oracle_plan_v2.json')
freeze = freeze.replace("source_ast_parsed=True)", "source_ast_parsed=True, labels_separated_in_controls=True)")
target = work/'freeze_train_oracle_plan_v2.py'; assert not target.exists(); target.write_text(freeze,encoding='utf-8')
print('V2_PREPARED_NOT_EXECUTED')
