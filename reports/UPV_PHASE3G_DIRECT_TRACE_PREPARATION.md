# Phase 3G direct-trace preparation result

Phase 3G has frozen a direct measured-trace target and a bounded RFsim
gain/noise response experiment. It has not authorized trace replay.

## Target

The target is the predesignated `corrected_test_1_ASUS` single-active-UE
survey. Its 305 paired one-second observations are preserved without
interpolation. Test 6 remains unopened.

The target spans −7.70 to 10.00 dB relative RSRP and 9.15 to 25.80 dB SINR.
The provisional inverse mapping produces gain commands from −17.67 to 0 dB
and noise commands from −34.89 to −17.28 dB. That mapping is an engineering
hypothesis for experiment design, not a calibrated inverse or a physical
noise estimate.

## Hardware response design

The frozen AWGN plan contains 45 clean UE execution units:

- Four gain safety checks and two noise safety checks.
- A 3 × 3 crossed gain/noise design with three executions per state.
- Four target-envelope boundary pairs with three executions per pair.

The crossed data will estimate the local Jacobian from gain and noise commands
to relative RSRP and SINR. Replay can proceed only if the operational checks,
coefficient ranges, conditioning gate, interaction gate, and boundary errors
all pass under execution-level uncertainty.

## Decision

The scientific protocol is ready for runner and profile freezing. The POWDER
reservation gate remains closed until that implementation is pinned and its
checksums are recorded. At that point the planned reservation is one gNB, one
UE, AWGN, d430 core, d740 cell node, for four hours.

No cross-session generalization, absolute-RSRP calibration, environmental-noise
inference, physical-channel reconstruction, ABC inference, or final Test 6
validation is authorized by this preparation.
