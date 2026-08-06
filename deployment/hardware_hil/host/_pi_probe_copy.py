import os
import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(
    "10.94.138.123",
    username="project",
    password=os.environ["CUKD_PI_PASSWORD"],
    timeout=20,
    allow_agent=False,
    look_for_keys=False,
)
cmd = (
    "ps aux | grep -E 'arduino|stream_vectors|compile' | grep -v grep; "
    "echo '---PORTS---'; "
    "ls -la /dev/ttyACM* /dev/ttyUSB* 2>&1; "
    "ls -la /dev/serial/by-id 2>&1; "
    "echo '---ESP CORE---'; "
    "export PATH=$HOME/.local/bin:$PATH; arduino-cli core list 2>&1; "
    "echo '---BUNDLES---'; "
    "ls $HOME/Desktop/CuKD-XAI/deployment/hardware_hil/build | grep esp32 || true"
)
_i, o, e = c.exec_command(cmd, timeout=60)
print(o.read().decode(errors="replace"))
print(e.read().decode(errors="replace"))
c.close()
