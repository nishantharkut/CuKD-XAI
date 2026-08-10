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
cmd = r"""
ps -p 7347 -o pid,etime,pcpu,pmem,cmd 2>/dev/null || echo 'pid 7347 gone'
OUTA=$HOME/Desktop/CuKD-XAI/results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_A
OUTB=$HOME/Desktop/CuKD-XAI/results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_B
echo '---A dir---'
ls -la "$OUTA" 2>&1
echo '---A lines---'
wc -l "$OUTA"/* 2>/dev/null || true
echo '---B dir---'
ls -la "$OUTB" 2>&1
echo '---B lines---'
wc -l "$OUTB"/* 2>/dev/null || true
"""
_i, o, e = c.exec_command(cmd, timeout=60)
print(o.read().decode(errors="replace"))
print(e.read().decode(errors="replace"))
c.close()
