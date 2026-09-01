# UPV-informed RFsim Version 1 final result

## Research question

Can a deterministic radio-condition model driven by a measured one-cell trace, when translated into bounded RFsim scalar-gain and effective-noise controls, reproduce predefined marginal, joint, and temporal properties of relative RSRP and SINR in OAI?

## Released model

Version 1 is a measurement-driven, KPI-level radio-condition emulator. It consumes synchronized relative RSRP and device-conditioned SINR targets, maps each pair to scalar gain and effective noise through bounded piecewise-affine interpolation, and applies the resulting commands to the AWGN RFsim channel once per second. Targets outside the validated translator hull are projected to the nearest hull boundary and explicitly flagged. The emulator never extrapolates or adapts commands during execution.

This is not a physical channel reconstruction. It does not infer absolute path loss or noise, and it does not reproduce multipath or Doppler.

## Evidence

### Test 1 complete-trace development evaluation

Three fresh UE recreations replayed all 305 target seconds. Every execution passed the frozen runtime and fidelity gates. Mean relative-RSRP MAE was 0.066 dB (range 0.061–0.075); mean SINR MAE was 1.651 dB (range 1.520–1.760). Mean correlations were 0.9996 for relative RSRP and 0.8081 for SINR. Maximum command lateness across the three executions was 0.173 s.

Test 1 is development fidelity and repeatability evidence, not independent final validation.

### Test 6 held-out exploratory evaluation

The unchanged translator replayed all 297 Test 6 target seconds in one execution. The predeclared support gate had already classified the trajectory as unsupported because 21/297 rows (7.07%) required projection. That verdict remains unchanged.

Despite the modest support violation, complete-trace relative-RSRP MAE was 0.081 dB and SINR MAE was 1.603 dB. Correlations were 0.9995 and 0.8237, respectively; scaled joint energy distance was 0.0067. All targets produced telemetry, IP reachability remained available, maximum command lateness was 0.166 s, and rollback completed.

For the 21 clipped rows, mean absolute target-to-projection error was 0.261 dB in relative RSRP and 0.479 dB in SINR. Complete original-target-to-OAI MAE on those rows was 0.291 dB and 1.653 dB, respectively.

Test 6 is genuine held-out exploratory evidence because it did not train or modify the translator. It is not confirmatory validation under the original protocol because its frozen support gate failed before replay, and one execution does not establish Test 6 execution-level repeatability.

## Phase 3M ablation

A post hoc Version 2 diagnostic tested first-order SINR memory using leave-one-complete-execution-out cross-validation. The static model had mean MAE 1.581 dB and mean p95 error 4.030 dB. The memory-only model reduced MAE to 1.404 dB, while the combined model reduced p95 error to 3.718 dB. Neither candidate passed every frozen gate: the combined p95 improvement was 7.74%, below the 8% requirement, and mean absolute residual lag-1 correlation increased from 0.620 to approximately 0.702.

Dynamic inverse compensation was therefore rejected. Version 1 commands and claims were not changed.

## Reproducibility

- Release status: `final_supported_version1_kpi_level_emulator`
- Analysis revision: `6b94c6a6f86d9a35f2c6425bd57ba452bac685c9`
- OAI revision: `70508ebaf52f2aae420566d380c6537f2efb9f0c`
- Primary profile revision: `86180671da37b1943e80cdec7d817678d4cc94f7`
- Test 6 exploratory wrapper revision: `c41e55857782fd15559a4cf4099dd14d17664ed9`
- Channel family: AWGN
- Command interval: 1 s
- Input bundles, generated artifacts, and checksums are recorded in the release inventory.

## Supported claim

Version 1 achieved high-fidelity KPI-level replay of relative RSRP and SINR across the complete Test 1 development trajectory and a held-out exploratory Test 6 trajectory, while maintaining UE attachment, IP reachability, command timing, and rollback integrity.

## Limitations

- RSRP is relative; absolute NEMO-to-OAI RSRP equivalence is unresolved.
- SINR is an empirical, device-conditioned KPI rather than a calibrated physical noise measurement.
- AWGN scalar gain and effective noise do not reconstruct multipath, Doppler, beam dynamics, or a channel impulse response.
- Test 6 exceeded the frozen support gate and was replayed once; it supports an exploratory generalization claim only.
- The evidence does not establish cross-device, cross-site, or population generalization.
- No real attachment-event distribution or throughput distribution was available for validation.
- Version 1 replays observed trajectories; it does not predict or universally generate radio conditions.
