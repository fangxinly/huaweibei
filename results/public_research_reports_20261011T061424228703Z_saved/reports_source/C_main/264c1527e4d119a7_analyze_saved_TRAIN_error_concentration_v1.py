"""Describe existing per-video error reports without scoring or fitting again."""
import datetime,hashlib,json,math,pathlib,zipfile,os,sys,shutil
P=pathlib.Path;r=P(sys.argv[1]);dc=r.parent.parent/'candidate_posttrain_lowC_20261009T005229Z';c=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
def raw(v):return (json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=r/'closed_verified_original/analysis/TRAIN_finite_prediction_report.json';archive=r/'remote_original/complete_actual_raw_TRAIN_closed.zip'
assert sha(archive)=='6c3799526decdedbe2ff328573e34d916593d2bcaf167597ddf87edef21c802f'
with zipfile.ZipFile(archive) as z:
 manifest=json.loads(z.read('capture_member_manifest.json'));expected=next(v for v in manifest if v['name']=='analysis/TRAIN_finite_prediction_report.json');assert sha(source)==expected['sha256']
report=json.loads(source.read_bytes());videos=report['per_video'];assert len(videos)==52 and sum(v['rows'] for v in videos)==report['rows']==1281
paths={}
for name,mse in report['pooled_row_weighted_MSE'].items():
 total=math.fsum(v['rows']*v['MSE'][name] for v in videos);assert math.isclose(total/1281,mse,rel_tol=1e-12,abs_tol=1e-12)
 ranked=sorted([dict(video_id=v['video_id'],rows=v['rows'],fold=v['fold'],MSE=v['MSE'][name],squared_error_sum=v['rows']*v['MSE'][name],fraction_of_path_total_SSE=v['rows']*v['MSE'][name]/total) for v in videos],key=lambda v:-v['squared_error_sum'])
 paths[name]=dict(original_pooled_row_MSE=mse,total_squared_error=total,largest_three_video_contributions=ranked[:3],all_five_fold_squared_errors=[dict(fold=f,rows=sum(v['rows'] for v in videos if v['fold']==f),SSE=math.fsum(v['rows']*v['MSE'][name] for v in videos if v['fold']==f)) for f in range(5)])
comparisons=[]
for pair in report['paired_comparisons']:
 a=pair['baseline'];b=pair['augmented'];delta=[v['MSE'][a]-v['MSE'][b] for v in videos]
 comparisons.append(dict(baseline=a,augmented=b,original_pooled_reduction=pair['pooled_row_MSE_reduction'],video_count_positive=sum(x>0 for x in delta),video_count_negative=sum(x<0 for x in delta),video_count_equal=sum(x==0 for x in delta),video_counts_not_significance_or_weighted_score=True))
now=datetime.datetime.now(datetime.timezone.utc);out=r/('saved_error_concentration_actual_'+now.strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
result=dict(status='SAVED_PER_VIDEO_ERROR_CONCENTRATION_DESCRIBED',actual_UTC=now.isoformat(),original_report_SHA=sha(source),original_archive_SHA=sha(archive),paths=paths,comparisons=comparisons,no_prediction_or_label_array_read=True,no_fit_or_scoring_rerun=True,all_52_videos_retained=True,original_primary_results_unchanged=True,outlier_not_excluded=True,not_claimed=['Modality contains no information','Uniform harm across videos','Statistical significance','A/B mechanism','Confirmed cause of extrapolation error'])
(out/'saved_error_concentration.json').write_bytes(raw(result));shutil.copyfile(source,out/'original_per_video_report.json');shutil.copyfile(__file__,out/P(__file__).name)
v=paths['T_V']['largest_three_video_contributions'][0];joint=paths['T_A_V']['largest_three_video_contributions'][0];a=paths['T_A']['largest_three_video_contributions'][0]
assert v['video_id']==joint['video_id'] and v['rows']==joint['rows']==16
lines=['# 已保存TRAIN探针的误差集中程度','',f"仅读原每视频报告（SHA {sha(source)}），没有重新读取预测/标签数组，没有再拟合或计分。全部52视频保留，四路原总体MSE不变。",'',f"视觉路径T+V：视频 {v['video_id']} 的16行贡献该路径总平方误差的 {100*v['fraction_of_path_total_SSE']:.3f}%。",f"联合路径T+A+V：同一视频贡献总平方误差的 {100*joint['fraction_of_path_total_SSE']:.3f}%。",f"音频路径T+A：视频 {a['video_id']} 的{a['rows']}行贡献总平方误差的 {100*a['fraction_of_path_total_SSE']:.3f}%。",'', '因此原负增量有明显的误差集中，不能说音频/视觉在所有视频上普遍无用。外视频外推失稳只是待解释的候选原因，尚未做输入尺度、缺失和原来源的定向因果核验。', '', '不事后删除视频、重写主要MSE、改阈值、按TEST选择参数或重跑原once。各视频正负增量计数只是描述，不等于显著性；原固定估计器仍无总体正向增量。','']
(out/'已保存TRAIN误差集中分析.md').write_text('\n'.join(lines),encoding='utf8')
target=out/'complete_saved_error_concentration_original.zip';members=[]
with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(out.iterdir()):
  if p==target:continue
  b=p.read_bytes();z.writestr(p.name,b);members.append(dict(name=p.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
 z.writestr('member_manifest.json',raw(members))
with zipfile.ZipFile(target) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())=={v['name'] for v in members}|{'member_manifest.json'}
 for m in members:assert hashlib.sha256(z.read(m['name'])).hexdigest()==m['sha256'] and len(z.read(m['name']))==m['bytes']
receipt=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(out),archive=str(target),archive_SHA=sha(target),all_member_SHA_CRC_unique_exact_set_passed=True,analysis_of_saved_report_only=True)
(out/'D_preservation_receipt.json').write_bytes(raw(receipt));state=json.loads((dc/'D_current_research_state.json').read_bytes());state['latest_raw_TRAIN_saved_error_concentration']=dict(D_receipt=receipt,source_report_SHA=sha(source),T_V_top_video=v,T_A_V_top_video=joint,T_A_top_video=a,primary_results_unchanged=True,no_prediction_fit_or_score_rerun=True);state['updated_at_utc']=receipt['actual_UTC']
for dest in [dc/'D_current_research_state.json',c/'自主优化实际接续.json']:
 tmp=dest.with_name(dest.name+'.concentration.tmp');tmp.write_bytes(raw(state));os.replace(tmp,dest)
assert sha(dc/'D_current_research_state.json')==sha(c/'自主优化实际接续.json')
with (c/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n\n'+receipt['actual_UTC']+' 仅已保存每视频报告独立分解：16行视频 '+v['video_id']+' 贡献T+V平方误差约'+str(round(100*v['fraction_of_path_total_SSE'],3))+'%、T+A+V约'+str(round(100*joint['fraction_of_path_total_SSE'],3))+'%；音频最大62行视频占约'+str(round(100*a['fraction_of_path_total_SSE'],3))+'%。全部52视频/原主要MSE保持，不删除异常视频救分、不重拟合计分、不冒外推原因已确认或模态普遍无信息。完整D '+str(out)+' SHA '+receipt['archive_SHA']+' 全SHA/CRC/unique/exactset过，D/C同字节。新建议仍仅inProgress部分commentary，未冒完整分析。\n')
print(json.dumps(dict(receipt=receipt,T_V_top=v,T_A_V_top=joint,T_A_top=a,comparisons=comparisons)))
