"""Transfer the prepared panel in bounded chunks and verify remote bytes."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[2]
CLI=ROOT/'work/colab-client-env/bin/colab'
SESSION='fjepa-pilot-20260917'
DATA=ROOT/'work/fjepa-cloud-data'
STAGING=ROOT/'work/fjepa-upload-parts'
STAGING.mkdir(exist_ok=True)
source=DATA/'panel_raw.npz'
parts=[]
whole=hashlib.sha256()
with source.open('rb') as stream:
    while block:=stream.read(16*1024**2):
        name=f'panel.part{len(parts):04d}'
        (STAGING/name).write_bytes(block)
        whole.update(block)
        parts.append({'name':name,'size':len(block),'sha256':hashlib.sha256(block).hexdigest()})
manifest={'size':source.stat().st_size,'sha256':whole.hexdigest(),'parts':parts}
(STAGING/'transfer_manifest.json').write_text(json.dumps(manifest,indent=2))
for i,part in enumerate(parts,1):
    subprocess.run([str(CLI),'upload','--session',SESSION,str(STAGING/part['name']),
                    'content/fjepa-data/'+part['name']],check=True)
    print(json.dumps({'uploaded_part':i,'total_parts':len(parts)}),flush=True)
for local in (STAGING/'transfer_manifest.json',DATA/'coverage.json'):
    subprocess.run([str(CLI),'upload','--session',SESSION,str(local),'content/fjepa-data/'+local.name],check=True)
remote='''from pathlib import Path
import hashlib,json,os
root=Path('/content/fjepa-data')
m=json.loads((root/'transfer_manifest.json').read_text())
digest=hashlib.sha256()
with (root/'panel_raw.npz.tmp').open('wb') as output:
    for part in m['parts']:
        block=(root/part['name']).read_bytes()
        assert len(block)==part['size'] and hashlib.sha256(block).hexdigest()==part['sha256']
        digest.update(block);output.write(block)
assert digest.hexdigest()==m['sha256']
os.replace(root/'panel_raw.npz.tmp',root/'panel_raw.npz')
assert (root/'panel_raw.npz').stat().st_size==m['size']
for part in m['parts']:(root/part['name']).unlink()
print('FINANCIAL_PANEL_TRANSFER_VERIFIED',m['size'],m['sha256'])
'''
remote_path=STAGING/'assemble.py';remote_path.write_text(remote)
subprocess.run([str(CLI),'exec','--session',SESSION,'--file',str(remote_path),'--timeout','120'],check=True)
(ROOT/'outputs/colab/transfer_manifest.json').write_text(json.dumps(manifest,indent=2))
for part in parts:(STAGING/part['name']).unlink()
