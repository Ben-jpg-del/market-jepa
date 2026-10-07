"""Upload a local artifact with bounded chunks through the official Colab CLI."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, subprocess, tempfile

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli',required=True)
    parser.add_argument('--session',required=True)
    parser.add_argument('--source',required=True)
    parser.add_argument('--remote',required=True)
    parser.add_argument('--manifest',required=True)
    args=parser.parse_args()
    source=Path(args.source).resolve()
    remote=PurePosixPath(args.remote)
    if not remote.is_absolute() or '..' in remote.parts or not str(remote).startswith('/content/'):
        raise ValueError('Remote artifact must be under /content')
    def call(command):
        result=subprocess.run([args.cli,*command],capture_output=True,text=True,timeout=240)
        if result.returncode:
            raise RuntimeError('Official Colab transfer operation failed; raw response suppressed')
        return result.stdout
    with tempfile.TemporaryDirectory(prefix='fjepa-transfer-') as temp:
        staging=Path(temp);parts=[];digest=hashlib.sha256()
        with source.open('rb') as stream:
            while block:=stream.read(16*1024**2):
                name=f'{remote.name}.part{len(parts):04d}'
                path=staging/name;path.write_bytes(block);digest.update(block)
                parts.append({'name':name,'size':len(block),'sha256':hashlib.sha256(block).hexdigest()})
                call(['upload','--session',args.session,str(path),str(remote.parent/name).lstrip('/')])
                path.unlink()
                print(json.dumps({'file':source.name,'uploaded_bytes':sum(p['size'] for p in parts),'total_bytes':source.stat().st_size}),flush=True)
        manifest={'source_name':source.name,'remote':str(remote),'size':source.stat().st_size,'sha256':digest.hexdigest(),'parts':parts}
        code="""from pathlib import Path
import hashlib,json,os
m=MANIFEST_LITERAL
destination=Path(m['remote'])
digest=hashlib.sha256()
temporary=destination.with_suffix(destination.suffix+'.uploading')
with temporary.open('wb') as output:
    for part in m['parts']:
        block=(destination.parent/part['name']).read_bytes()
        assert len(block)==part['size'] and hashlib.sha256(block).hexdigest()==part['sha256']
        digest.update(block);output.write(block)
assert digest.hexdigest()==m['sha256'] and temporary.stat().st_size==m['size']
os.replace(temporary,destination)
for part in m['parts']:(destination.parent/part['name']).unlink()
print('ARTIFACT_VERIFIED',m['sha256'])
""".replace('MANIFEST_LITERAL',repr(manifest))
        script=staging/'assemble.py';script.write_text(code)
        output=call(['exec','--session',args.session,'--file',str(script),'--timeout','120'])
        if 'ARTIFACT_VERIFIED '+manifest['sha256'] not in output:
            raise RuntimeError('Remote digest verification marker missing')
        path=Path(args.manifest);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(manifest,indent=2))
        print(json.dumps({'verified':True,'file':source.name,'sha256':manifest['sha256']}),flush=True)

if __name__=='__main__':main()
