# Compile Logs

Keep Arduino IDE compile summaries here as plain text.

Required for final framework-overhead baseline analysis:

```text
esp32c3_serial_baseline_compile.txt
arduino_r4_serial_baseline_compile.txt
```

Expected content format:

```text
Sketch uses XXXXX bytes (YY%) of program storage space. Maximum is XXXXX bytes.
Global variables use XXXXX bytes (YY%) of dynamic memory, leaving XXXXX bytes for local variables. Maximum is XXXXX bytes.
Board: ...
Source: Arduino IDE Verify output for serial-only baseline.
```

Existing model compile logs can stay in this directory:

```text
esp32c3_student_a_compile.txt
arduino_r4_student_a_compile.txt
esp32c3_student_b_compile.txt
arduino_r4_student_b_compile.txt
```

