"""Credential access is opt-in; imports and default invocation do nothing private."""
import argparse,json

def check_repo_access():
    import base64,os,subprocess,urllib.request,urllib.error
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    try:
        r=subprocess.run(['git','-c','credential.interactive=never','credential','fill'],
            input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,env=env,timeout=25)
        fields=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
        if r.returncode or not fields.get('password'):
            return dict(credential_helper_available=False,returncode=r.returncode)
        key=base64.b64encode((fields.get('username','')+':'+fields['password']).encode()).decode()
        req=urllib.request.Request('https://api.github.com/repos/fangxinly/huaweibei',headers={
            'Authorization':'Basic '+key,'User-Agent':'Codex-source-upload','Accept':'application/vnd.github+json'})
        with urllib.request.urlopen(req,timeout=20) as response:data=json.load(response)
        return dict(credential_helper_available=True,repository=data['full_name'],default_branch=data['default_branch'],
                    private=data['private'],permissions=data.get('permissions'))
    except urllib.error.HTTPError as exc:return dict(http_status=exc.code)
    except Exception as exc:return dict(error_type=type(exc).__name__)

def main(argv=None):
    parser=argparse.ArgumentParser(description='Optional read-only access check for fangxinly/huaweibei; credentials never printed.')
    parser.add_argument('--check-repo-access',action='store_true',help='Explicitly permit the Git credential helper read and HTTPS repository access check.')
    args=parser.parse_args(argv)
    result=check_repo_access() if args.check_repo_access else dict(status='SKIPPED_NO_EXPLICIT_CREDENTIAL_ACCESS',credential_helper_read=False,network_access=False)
    print(json.dumps(result))

if __name__=='__main__':main()
