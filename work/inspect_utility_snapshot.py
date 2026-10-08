from pathlib import Path
import argparse,json,zipfile
a=argparse.ArgumentParser();a.add_argument('directory',type=Path);c=a.parse_args()
for node,mode in [('a','none'),('b','fixed'),('c','predicted')]:
    with zipfile.ZipFile(c.directory/node/'snapshot.zip') as z:
        prefix='inflow_utility_v4_deployment_20261005T0346Z/'
        h=json.loads(z.read(prefix+'run_'+mode+'/history.json'));res=json.loads(z.read('actual_resources.json'))
        print(json.dumps({'node':node,'epochs':len(h),'complete_marker':'INFLOW_UTILITY_RUN_COMPLETE' in z.read(prefix+'training.log').decode(),'compute':res['compute']['output'],'disk':res['disk_free_bytes'],'selection_present':prefix+'run_'+mode+'/selection.json' in z.namelist()}))
