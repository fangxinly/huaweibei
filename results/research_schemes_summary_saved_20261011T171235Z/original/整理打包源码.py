"""Package existing research evidence; no scientific imports or execution."""
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE/'work'))
from prepare_github_source_upload import sanitize

OLD = Path('C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen')
ARCH = BASE/'outputs/public_research_reports_20261011T061424228703Z'
DC = Path('D:/CodexBackups/selective_flow_20261003_1105/candidate_posttrain_lowC_20261009T005229Z')
DB = DC.parent/'candidate_final100_actual_20261009T012656Z'
EXPECTED_ARCHIVE_SHA = 'af63021e369f3c07841d3ae217796842073ebef952302053af6045112845c576'

def sha(b): return hashlib.sha256(b).hexdigest()
def write(p, o):
    p.write_text(json.dumps(o, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')

REFERENCES = [
 ('E01', OLD/'outputs/selective_flow_prototype/实验记录.md'),
 ('E02', OLD/'outputs/selective_flow_prototype/v2/结果汇总.md'),
 ('E03', OLD/'outputs/selective_flow_prototype/v3_uncertainty_run/结果汇总.md'),
 ('E04', OLD/'outputs/selective_flow_prototype/v4_overlap_run/结果汇总.md'),
 ('E05', OLD/'outputs/parallel_gpu_20261003_1800/experiment_statistics_20261004T1253Z/统计总表.md'),
 ('E06', OLD/'outputs/own_flow_v7_native/README.md'),
 ('E07', OLD/'outputs/own_flow_v8_identifiability/README.md'),
 ('E08', OLD/'outputs/own_flow_v9_task_feedback/推导实现与迭代实验_20261004.md'),
 ('E09', OLD/'outputs/own_flow_v9_task_feedback/共享交换与对比学习_20261005.md'),
 ('E10', OLD/'outputs/own_flow_v9_task_feedback/预测等价交换候选_20261005.md'),
 ('E11', OLD/'outputs/own_flow_v9_task_feedback/固定更新预算与正则化交互_20261005.md'),
 ('E12', OLD/'outputs/own_flow_v9_task_feedback/视频互斥校准与迁移失败_20261005.md'),
 ('E13', BASE/'outputs/条件流实验分析.md'),
 ('E14', BASE/'outputs/门控实验分析.md'),
 ('E15', BASE/'outputs/逐样本效用实验分析.md'),
 ('E16', BASE/'outputs/主任务反事实效用实验分析.md'),
 ('E17', BASE/'outputs/TRAIN_Oracle可达性实验分析.md'),
 ('E18', BASE/'outputs/视频隔离教师100轮与OOF效用分析.md'),
 ('E19', BASE/'outputs/OOF教师驱动消息修改实验分析.md'),
 ('E20', BASE/'outputs/方向幅度与接受规则交叉机制分析.md'),
 ('E21', BASE/'outputs/效用校准与保守消息修改下一步优化.md'),
 ('E22', BASE/'outputs/首步标量方向几何GPU核验与优化决策.md'),
 ('E23', BASE/'outputs/固定CAL视频等权残差信号实际结果.md'),
 ('E24', BASE/'outputs/匹配消息候选对照实际结果与优化决策.md'),
 ('E25', BASE/'outputs/原生位移半径CAL零标签机制实际结果.md'),
 ('E26', BASE/'outputs/正式双方TEST685一次五项实际结果.md'),
 ('E27', BASE/'outputs/所有固定模型VAL_TEST五项实际对齐结果.md'),
 ('E28', BASE/'outputs/官方划分原方案消息改进与CaReFlow对齐实际接续.json'),
 ('E29', BASE/'outputs/MSE100官方VAL_TEST实际结果.md'),
 ('E30', DB/'AB_official_five_complete_saved_actual_20261009T180914Z/actual_verified_results.json'),
 ('E31', BASE/'outputs/新方案40轮实际完成与完整保存接续.md'),
 ('E32', BASE/'outputs/Claude建议核验与残差消息诊断实际结果_20261008.md'),
 ('E33', BASE/'outputs/标量捷径核验与best20_best36逐样本导出实际结果_20261008.md'),
 ('E34', BASE/'outputs/历史TEST优势方案与CaReFlow五折执行设计.md'),
 ('E36', BASE/'outputs/同T0锁定常数外折开发测量实际结果.md'),
 ('E37', BASE/'outputs/完整流单参考100完成与完整保存实际结果.md'),
 ('E38', BASE/'outputs/完整流单候选232零标签实际结果.md'),
 ('E39', BASE/'outputs/完整流匹配U_W头201一次开发五指标实际结果.md'),
 ('E40', BASE/'outputs/下一步原流方向残差控制与实际原机验证.md'),
 ('E41', DB/'raw_TRAIN_retry_prepared_actual_20261009T194523Z/TRAIN固定视频五折实际结果.md'),
 ('E42', DB/'TRAIN_support_retry_actual_20261010T004247Z/TRAIN原拟合状态支持诊断结果.md'),
 ('E43', BASE/'outputs/连续向量三臂实验分析.md'),
 ('E44', BASE/'outputs/Jacobian方向约束三臂实验分析.md'),
 ('E45', BASE/'outputs/有限任务风险C2修订实验分析.md'),
 ('E46', BASE/'outputs/三级消息控制建议评估与短实验修订设计.md'),
 ('E47', BASE/'outputs/弱情绪直接修正对照与新训练方案实际结果.md'),
 ('E48', BASE/'outputs/原方案改进VAL_TEST五项实际接续.json'),
]

def metric_excerpt(obj):
    found = []
    def visit(o, path):
        if isinstance(o, dict):
            numeric = {k:v for k,v in o.items() if isinstance(v, (int,float)) and not isinstance(v,bool)}
            if any('mae' in k.lower() or k.lower() in {'acc7','acc2','corr','pearson','f1','non0_acc_2','non0_f1_score'} for k in numeric):
                found.append(dict(original_JSON_path=path, numeric_metrics=numeric))
            for k,v in o.items(): visit(v, path+'/'+str(k))
        elif isinstance(o, list):
            for i,v in enumerate(o): visit(v, path+'/'+str(i))
    visit(obj,'')
    status = {k:v for k,v in obj.items() if isinstance(v,(bool,int,float)) or (isinstance(v,str) and k in {'status','phase','actual_UTC','actualclock_UTC'})}
    return dict(excerpt_only=True, status_fields=status, metric_records=found,
                note='只摘原JSON中指标及状态；未重算科学结果，完整结构仍由原件SHA绑定。')

def main():
    free = {d:shutil.disk_usage(d+':/').free for d in ['C','D']}
    assert free['C'] > 200*1024**2 and free['D'] > 40*1024**2
    now = dt.datetime.now(dt.timezone.utc)
    stamp = now.strftime('%Y%m%dT%H%M%SZ')
    out = BASE/'outputs'/('研究方案总整理_'+stamp)
    out.mkdir(); (out/'references').mkdir()
    mapping = json.loads((ARCH/'payload/PUBLIC_FILE_MAP.json').read_bytes())
    records = []
    for code, src in REFERENCES:
        assert src.is_file(), str(src)
        raw = src.read_bytes()
        origin_sha = sha(raw)
        if src.suffix == '.json':
            excerpt = metric_excerpt(json.loads(raw))
            assert excerpt['metric_records'], code+' has no original metric records'
            cleaned = (json.dumps(excerpt,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
            changes = {'metric_and_status_excerpt':True}
            target = out/'references'/(code+'.json')
            map_path = None
        else:
            matches = [r for r in mapping if r.get('original_SHA256') == origin_sha and r['original_relative_path'].replace('\\','/').endswith(src.name)]
            if matches:
                match = matches[0]
                public = ARCH/'payload'/match['public_path']
                cleaned = public.read_bytes()
                assert sha(cleaned)==match['public_SHA256']
                changes = match['sanitizations']; map_path=match['public_path']
            else:
                cleaned, changes = sanitize(raw); map_path=None
            target = out/'references'/(code+'.md')
        target.write_bytes(cleaned)
        records.append(dict(id=code,title=src.name,original_path=str(src),original_SHA256=origin_sha,
                            original_bytes=len(raw),package_path=target.relative_to(out).as_posix(),
                            package_SHA256=sha(cleaned),package_bytes=len(cleaned),sanitizations=changes,
                            existing_public_path=map_path))
    # Current status excerpt binds the actual pointer bytes without copying private transport details.
    status_paths = [BASE/'work/group5_careflow0_complete_pointer.json',BASE/'work/group5_careflow1_native_capture_pointer.json',
                    BASE/'work/new_node23668_formal_pointer.json',BASE/'work/group5_stopped_CPU_capsule_pointer.json',
                    DC/'D_current_research_state.json']
    status_sources = [dict(path=str(p),SHA256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in status_paths]
    formal = json.loads(status_paths[2].read_bytes())
    assert formal['training_dispatched'] is False and formal['once_consumed'] is False
    current = dict(assembled_actual_UTC=now.isoformat(),status='RESEARCH_SCHEMES_ORGANIZED_NO_NEW_TRAINING',
                   latest_user_instruction='算了 你还是整理清楚到目前位置 所有的方案吧',
                   final_method_fold_groups_planned=25,training_stages_planned=35,
                   frozen_unscored_outer_prediction_groups=1,scored_outer_groups=0,
                   careflow_fold0_formal100_complete=True,careflow_fold1_native3_complete=True,
                   careflow_fold1_formal100_complete=False,new_node_formal_dispatched=formal['training_dispatched'],
                   new_node_train_once_consumed=formal['once_consumed'],local_training_started=False,
                   composite_CPU_and_CPU_CUDA_resume_actual_qualification_passed=False,
                   source_only_synthetic_checks_not_native_qualification=True,
                   first_formal_wrapper_before_scientific_import_path_failure='原失败保留；修正plan未派发。',
                   no_scientific_rerun_or_new_label_decoding_in_this_summary=True,original_sources=status_sources)
    target=out/'references/E35.json';write(target,current)
    records.append(dict(id='E35',title='当前真实进度与原指针SHA摘录',original_sources=status_sources,
                        package_path='references/E35.json',package_SHA256=sha(target.read_bytes()),package_bytes=target.stat().st_size,
                        generated_status_excerpt=True))
    records.sort(key=lambda r:r['id'])
    draft=(BASE/'work/research_scheme_summary_20261012.md').read_text(encoding='utf8')
    # Turn shorthand ranges into individual clickable references.
    def expand(m): return ' '.join('[E%02d]'%i for i in range(int(m.group(1)),int(m.group(2))+1))
    draft=re.sub(r'\[E(\d{2})[–—-]E(\d{2})\]',expand,draft)
    links='\n'.join('- **'+r['id']+'** ['+r['title']+']('+r['package_path']+')' for r in records)
    defs='\n'.join('['+r['id']+']: '+r['package_path'] for r in records)
    draft=draft.replace('<!-- EVIDENCE_LINKS -->',links+'\n\n'+defs)
    used=set(re.findall(r'\[(E\d{2})\]',draft));defined={r['id'] for r in records}
    assert used<=defined,used-defined
    nav=draft.split('## 3. 全部方案导航')[1].split('## 4.')[0]
    ids=re.findall(r'\| (S\d{2}) \|',nav)
    assert ids==['S%02d'%i for i in range(1,56)]
    cleaned, changes=sanitize(draft.encode());assert not changes,changes
    (out/'研究方案总览.md').write_bytes(cleaned)
    zip_src=ARCH/'public_research_reports_source.zip'; archive=zip_src.read_bytes()
    assert sha(archive)==EXPECTED_ARCHIVE_SHA and len(archive)==4250645
    shutil.copyfile(zip_src,out/'历史研究报告与源码.zip')
    shutil.copyfile(ARCH/'payload/PUBLIC_FILE_MAP.json',out/'历史文件映射.json')
    assert len(mapping)==1745 and len({r['public_path'] for r in mapping})==1401
    # List the true preserved source names as an index; never execute them.
    source_rows=[r for r in mapping if r['original_relative_path'].endswith('.py')]
    lines=['# 源码与方案入口','',
      '历史包按内容去重；先读总览，再按下面的文件名关键词在「历史文件映射.json」中定位 public_path，解压历史包后打开该文件。',
      '同名文件可能有多个版本，必须同时核对 original_relative_path、original_SHA256 和 public_SHA256。下面只列原始映射，代码未在本次执行。',
      '', '| 路线 | 建议检索词 |', '|---|---|',
      '| S01–S05 | selective_flow_prototype、overlap、uncertainty |',
      '| S06–S18 | own_flow_v5、own_flow_v6、v7、v8、v9、careflow、swap、feedback |',
      '| S19–S25 | conditional、gated、utility、counterfactual、jacobian、finite |',
      '| S26–S34 | oracle、oof、teacher、geometry、candidate、native |',
      '| S35–S39 | constant、fullflow、reference、residual、official |',
      '| S40–S48 | anchored、weak、message、huber、mse、factorized、regression |',
      '| S49–S50 | raw_TRAIN、support、ridge |',
      '| S51–S52 | group5、epoch、resume、CPU、capsule |',
      '| S53–S55 | 以总览列明的设计稿为准，未实现部分没有可运行源码 |',
      '', '## 原始Python映射清单', '', '| 原始根 | 原路径 | 历史包内路径 |', '|---|---|---|']
    lines += ['| '+r['root']+' | '+r['original_relative_path'].replace('|','\\|')+' | '+r['public_path']+' |' for r in source_rows]
    (out/'方案源码入口.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    readme='''# 在Linux阅读本研究包

本包用于整理、复查与迁移；不自动恢复或开始训练。

1. 解压完整ZIP，先打开「研究方案总览.md」。相对链接指向 references/ 下的原报告副本。
2. 打开「方案证据索引.json」核对来源和SHA。历史报告保留当时措辞，最新进度以总览及E35为准。
3. 如需源码，单独解压「历史研究报告与源码.zip」。在「历史文件映射.json」中按原路径/文件名查询 public_path。
4. 路径、环境、资产、数据角色与运行资格需另行绑定。Windows绝对路径是来源记录，不能在Linux直接照抄执行。

历史包：1745条原路径映射、1401份去重报告/指标/源码；原包SHA256：
af63021e369f3c07841d3ae217796842073ebef952302053af6045112845c576

本包不含聊天全文、私有工具日志、密码、私钥、任务标签数组或大模型权重。
不把整理核对称为新科学实验；所有已完成once及未评分预测保持原状态。
'''
    (out/'README_在Linux阅读.md').write_text(readme,encoding='utf8')
    index=dict(actual_UTC=now.isoformat(),scope='截至北京时间2026-10-12的已有方案、结果、失败和未执行设计',
               scheme_count=55,evidence_count=len(records),references=records,
               historical_archive=dict(path='历史研究报告与源码.zip',SHA256=EXPECTED_ARCHIVE_SHA,bytes=len(archive),
                 original_file_mappings=1745,deduplicated_documents_and_sources=1401,
                 existing_public_URL='https://github.com/fangxinly/huaweibei/releases/download/autonomous-flow-health-20261008/research-Linux-reports-source-af63021e369f.zip'),
               no_new_training_inference_scoring_or_model_array_decoding=True)
    write(out/'方案证据索引.json',index)
    checks=dict(actual_UTC=now.isoformat(),scheme_navigation_S01_to_S55_exact=True,
                all_used_evidence_links_resolved=True,reference_copies_SHA_verified=True,
                historical_zip_exact_SHA_verified=True,historical_mapping_count=1745,
                historical_unique_public_paths=1401,source_mapping_rows=len(source_rows),
                source_only_no_scientific_rerun=True,new_formal_dispatch_false_verified=True,
                new_formal_once_false_verified=True,summary_credential_sanitizer_no_changes=True)
    write(out/'整理核对结果.json',checks)
    pointer=dict(actual_UTC=now.isoformat(),folder=str(out),main=str(out/'研究方案总览.md'),
                 evidence_index=str(out/'方案证据索引.json'),checks=checks,
                 status='LOCAL_READABLE_SUMMARY_COMPLETE_PRESERVATION_PENDING')
    write(BASE/'work/research_scheme_summary_pointer.json',pointer)
    print(json.dumps(pointer,ensure_ascii=False))

if __name__=='__main__': main()
