import subprocess, os
os.environ['PORT'] = '5001'
cwd = os.getcwd()
p = subprocess.Popen(
    [os.path.join(cwd, 'venv', 'Scripts', 'python.exe'), os.path.join(cwd, 'api_server.py')],
    cwd=cwd,
    creationflags=0x08000000,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    env={**os.environ, 'PORT': '5001'}
)
print(f"PID: {p.pid}")
