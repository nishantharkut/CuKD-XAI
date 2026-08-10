import os
import paramiko

host = os.environ.get("CUKD_PI_HOST", "192.168.137.234")
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(
    host,
    username="project",
    password=os.environ["CUKD_PI_PASSWORD"],
    timeout=25,
    allow_agent=False,
    look_for_keys=False,
)
cmd = r"""
echo CONNECTED
hostname; hostname -I
ls -la /dev/ttyUSB* /dev/ttyACM* /dev/serial/by-id 2>&1
ps aux | grep stream_vectors | grep -v grep || echo NO_STREAM
A=$HOME/Desktop/CuKD-XAI/results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_A
B=$HOME/Desktop/CuKD-XAI/results/hardware_hil/train_only_scaler_copy/pi5_esp32c3_student_B
echo ---A---
ls -la "$A" 2>&1
echo -n A_FULL_LINES=
if test -f "$A/full_56200_mcu.csv"; then wc -l < "$A/full_56200_mcu.csv"; else echo 0; fi
echo ---B---
ls -la "$B" 2>&1
echo -n B_FULL_LINES=
if test -f "$B/full_56200_mcu.csv"; then wc -l < "$B/full_56200_mcu.csv"; else echo 0; fi
if test -f "$A/full_56200_sequence.json"; then echo A_SEQ=YES; else echo A_SEQ=NO; fi
if test -f "$B/full_56200_sequence.json"; then echo B_SEQ=YES; else echo B_SEQ=NO; fi
"""
_i, o, e = c.exec_command(cmd, timeout=60)
print(o.read().decode(errors="replace"))
print(e.read().decode(errors="replace"))
c.close()
