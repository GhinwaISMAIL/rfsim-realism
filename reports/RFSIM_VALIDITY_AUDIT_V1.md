# RFsim Validity Audit V1

## Decision

The pinned OAI per-channel noise control does not implement its documented
power-domain dB semantics. The profile now applies a reproducible correction,
but no corrected POWDER image or execution exists yet. Historical raw data and
commands remain immutable.

The audit changes one important conclusion: Phase 3C15 did not establish that a
fixed-noise scalar-gain replay lacks SINR support. Its `noise_power_dB = -30`
condition used the legacy amplitude conversion and is equivalent to corrected
`-60 dB` relative noise power. Corrected fixed-noise support must be measured.

The Phase 3D development result is unchanged because it fitted and evaluated
only UPV measurements. Its final Test 6 payload remains locked.

## Noise-control finding

The pinned RFsim source multiplies independent Gaussian I and Q samples by:

```text
256 * 10^(noise_power_dB/10)
```

An RMS amplitude derived from a power-domain dB value requires:

```text
256 * 10^(noise_power_dB/20)
```

Squaring the legacy amplitude doubles the commanded change in the power domain.
Consequently, a legacy command `n` is reproduced by corrected command `2n`.
This is an equivalence of generated RMS amplitudes, not permission to rewrite
historical commands.

The profile correction also replaces the misleading amplitude-derived log with
the commanded power value. Unit tests verify the power ratio and legacy mapping,
and the complete profile suite passes.

## Other issue findings

- The open physical-nrUE RSRP mismatch report is a warning about absolute
  measurement equivalence, not an independently justified offset for RFsim.
- The tested TDL-B downlink used a 30 ns delay spread represented in seconds and
  exposed 12 taps over a 15-sample impulse response. The reported first-tap-only
  failure is not supported for that execution.
- The pinned TDL constructor mutates shared normalized-delay storage. The profile
  already prevents this by allocating delays per channel. Future TDL images must
  retain that patch.
- Multiple-UE per-channel noise accumulation is outside the one-UE scope. A
  future multi-UE claim requires global noise or a separately validated
  single-addition implementation.
- The CIRDB failure remains a real-time processing result and is unaffected by
  the noise correction.

## Next gate

The corrected profile revision is available on the POWDER profile branch, and
the state order, repetitions, and executable operational gates are frozen in
`corrected_noise_validation_protocol.json`. The reservation gate is therefore
open. Request one gNB and one UE using AWGN for four hours (three hours
minimum), with the first POWDER-dependent action no earlier than the timestamp
in `reservation_gate.json`. Its only purpose is to build the corrected image
and validate the corrected noise response. It does not select a replay noise
state or authorize a gain-noise inverse mapping.
