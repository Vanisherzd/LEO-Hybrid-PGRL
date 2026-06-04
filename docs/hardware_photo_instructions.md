# Hardware Photo Instructions

No real bench photo is currently committed in this repository.

For reviewer-facing artifact support, capture the following three images:

1. A clear close-up of the Semtech LR1121 board mounted on the NUCLEO-L476RG.
2. A clear close-up of the USRP B210 connection path used for IQ capture.
3. A wider bench shot showing both devices, RF path, USB cables, and antennas or coax.

Label the final selected images with these identifiers:

- `LR1121`
- `NUCLEO-L476RG`
- `USRP B210`
- `RF path`
- `USB`

Recommended capture notes:

- Use good room lighting and avoid motion blur.
- Keep the entire RF path visible when possible.
- If coax or attenuators are used, include them in the wide shot.
- Avoid screenshots of plots; this file is for physical testbed photos only.

Recommended artifact placement after capture:

- `hardware/artifacts/testbed_photo/lr1121_nucleo_closeup.*`
- `hardware/artifacts/testbed_photo/usrp_connection_closeup.*`
- `hardware/artifacts/testbed_photo/testbed_wide_shot.*`

Claim boundary for any future photo caption or README:

- The photo documents the physical testbed only.
- It does not upgrade the measurement claim beyond IQ-level RF signal detection.
- It does not imply LR-FHSS decoding, PER measurement, or a full gateway receiver.
