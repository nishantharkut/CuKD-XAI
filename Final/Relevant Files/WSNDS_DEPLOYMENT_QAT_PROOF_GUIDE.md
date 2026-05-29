# WSN-DS Deployment + QAT Proof Guide

This route is copied from the v2.3 base and narrowed to a deterministic deployment and quantization proof. It does **not** replace the completed 10-seed statistical WSN-DS results.

## What this proof run does
- Uses PROOF_SEED = 9999 for deterministic evidence only.
- Trains only the selected proof models:
  - A_RF_500
  - B_Full_MLP
  - D_student_A_scratch
  - E_student_A_KD_from_RF
  - D_student_B_scratch
  - E_student_B_KD_from_RF
- Applies dynamic INT8 to PyTorch MLPs and QAT to the student MLPs only.
- Records deployability evidence: size, latency, and compression ratios.

## Expected runtime
Plan for about **2-6 hours** depending on CPU and RAM.

## Inputs
- Place WSN-DS.csv in the same folder as the proof script, or update WSNDS_PATH.

## Outputs (use in the paper)
All outputs are written to:
- wsnds_deployment_qat_outputs/

Key files for the paper:
- wsnds_deployment_summary.csv (main proof table)
- wsnds_deployment_results.json (full metrics + metadata)
- wsnds_qat_summary.csv (QAT-only results)
- wsnds_latency_summary.csv (latency-only results)
- wsnds_environment.json (environment details)

## Reminder
This deployment/QAT proof route is only for deployability, quantization, size, latency, and compression evidence. It is **not** a replacement for the 10-seed statistical WSN-DS result set.
