# CuKD-XAI Tier 1.5 Master Report (Train-Only Seed 42)
**Status: complete**
## Predictive (software)
- Train-only deployment seed 42 RF-KD macro-F1: A **0.9485**, B **0.9449**
- Feature-group 5-seed RF-KD macro-F1: A **0.9141+/-0.0069**, B **0.9281+/-0.0074**
- Feature-group KD-minus-scratch mean macro-F1 delta: A **+0.0002**, B **-0.0017** (descriptive only)
## Host conversion
- ONNX FP32 agreement vs PyTorch: **1.0** (A/B)
- Dynamic INT8 ONNX: size-oriented; macro-F1 drop (A 0.9485->0.8938, B 0.9449->0.9066)
- OpenVINO `E_student_A_KD_from_RF_train_only`: agree_pt=1.0, agree_ort=1.0, p50=0.1098 ms
- OpenVINO `E_student_B_KD_from_RF_train_only`: agree_pt=1.0, agree_ort=1.0, p50=0.1138 ms
## Hardware
- arduino_r4 student A: n=56200, MCU/fixed=1.0, MCU/FP32=0.9919, F1=0.9244, lat_mean=301.5 us
- arduino_r4 student B: n=56200, MCU/fixed=1.0, MCU/FP32=0.9905, F1=0.9180, lat_mean=791.4 us
- esp32c3 student A: n=56200, MCU/fixed=1.0, MCU/FP32=0.9919, F1=0.9244, lat_mean=116.5 us
- esp32c3 student B: n=56200, MCU/fixed=1.0, MCU/FP32=0.9905, F1=0.9180, lat_mean=320.3 us
## Claim boundary
- Archived 10-seed tables remain the multi-seed primary report.
- Train-only seed-42 closes scaler lineage for deployment/HIL.
- Feature-group 5-seed is sensitivity, not a matched significance test.
