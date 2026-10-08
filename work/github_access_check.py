"""Check Git credential helper access without exposing credentials."""
import base64,json,os,subprocess,urllib.request,urllib.error
env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
r=subprocess.run(['git','-c','credential.interactive=never','credential','fill'],
    input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,env=env,timeout=25)
fields=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
if r.returncode or not fields.get('password'):
    print(json.dumps(dict(credential_helper_available=False,returncode=r.returncode)))
else:
    key=base64.b64encode((fields.get('username','')+':'+fields['password']).encode()).decode()
    req=urllib.request.Request('https://api.github.com/repos/fangxinly/huaweibei',
        headers={'Authorization':'Basic '+key,'User-Agent':'Codex-source-upload','Accept':'application/vnd.github+json'})
    try:
        with urllib.request.urlopen(req,timeout=20) as response: data=json.load(response)
        print(json.dumps(dict(credential_helper_available=True,repository=data['full_name'],
            default_branch=data['default_branch'],private=data['private'],permissions=data.get('permissions'))))
    except urllib.error.HTTPError as e: print(json.dumps(dict(credential_helper_available=True,http_status=e.code)))
