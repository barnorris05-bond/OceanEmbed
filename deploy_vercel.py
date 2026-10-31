"""Build Vercel deployment payload."""
import json, base64, os

files_to_upload = {}

# Core files (no requirements.txt — using vercel.json installCommand)
for f in ['api_server.py', 'land_mask.py', 'train_model.py', 'vercel.json', 'build.sh', '.python-version', 'api/index.py', 'ocean_data.csv', 'test_sample.csv']:
    if os.path.exists(f):
        with open(f, 'rb') as fh:
            files_to_upload[f] = base64.b64encode(fh.read()).decode()

# Frontend
with open('frontend/index.html', 'rb') as fh:
    files_to_upload['public/index.html'] = base64.b64encode(fh.read()).decode()

# Model artifact
with open('model.pkl', 'rb') as fh:
    files_to_upload['model.pkl'] = base64.b64encode(fh.read()).decode()
    print(f'model.pkl: {os.path.getsize("model.pkl"):,} bytes')

# Land data
for root, dirs, fnames in os.walk('land_data'):
    for fname in fnames:
        fpath = os.path.join(root, fname).replace(os.sep, '/')
        with open(fpath, 'rb') as fh:
            files_to_upload[fpath] = base64.b64encode(fh.read()).decode()

print(f'Total files: {len(files_to_upload)}')

files_payload = [{'file': name, 'data': data, 'encoding': 'base64'} for name, data in files_to_upload.items()]

deployment = {
    'name': 'oceanembed',
    'project': 'prj_7YoqW3TcdksV2gJebPm3EfyCUgcM',
    'target': 'production',
    'files': files_payload,
    'buildCommand': 'echo done',
    'outputDirectory': 'public',
    'framework': None
}

with open('vercel_deploy.json', 'w') as f:
    json.dump(deployment, f)

payload_size = os.path.getsize('vercel_deploy.json')
print(f'Payload: {payload_size:,} bytes ({payload_size/1024/1024:.1f} MB)')
