# Mac Hardware Readiness Inventory (READ-ONLY)

> **READ-ONLY discovery pass. NO transmit, NO capture, NO flash, NO reset, NO UART
> monitoring, NO RF command was run.** Hardware is connected but the conducted-HIL
> experiment was NOT executed. `reference_is_measured_truth=false`.

*Generated: 2026-06-13 20:12 UTC (Mac). Branch `experiment-bk2-tle-residual`.
Discovery used only: `git`, `ls /dev/cu.*`, `ioreg -p IOUSB` (read-only),
`which`, and `uhd_find_devices` (enumeration only — loads volatile FX3 firmware
to RAM, does not stream or transmit).*

---

## 1. Git state and latest commit

- Branch: `experiment-bk2-tle-residual`, in sync with
  `origin/experiment-bk2-tle-residual`.
- HEAD: `2a8b6f4  docs: add Mac HIL preflight plan`.
- Working tree: no tracked file modified. Untracked: 13 `docs/review/*` files +
  `logs/` from the prior software-only pass. Nothing staged.
- Recent commits: `2a8b6f4`, `029ea20`, `695b2da`, `df2959d`, `aacd1f1`,
  `b194dd5`, `92da417`, `c0d516d`.

## 2. Detected USB devices (relevant to USRP B210 / ST-Link / LR1121)

Source: `ioreg -p IOUSB -l` (read-only). After `uhd_find_devices` the B210
re-enumerated with its programmed serial.

| Device | USB Vendor | VID:PID | Serial | Role |
|---|---|---|---|---|
| **USRP B210** (Ettus FX3) | Ettus/Cypress "WestBridge" | `0x2500:0x0020` (pre-image) → enumerates as B210 | pre-image `0000000004BE`; programmed `8000304` | conducted SDR receiver |
| **STM32 ST-Link/V2-1** | STMicroelectronics | `0x0483:0x374B` | `0670FF3234584D3043215150` | LR1121 dev-board debugger + USB VCP |
| GenesysLogic USB3.1 Hub | GenesysLogic | `0x1507:0x0616` | — | hub (not target) |
| GenesysLogic USB2.1 Hub | GenesysLogic | `0x1507:0x0610` | — | hub (not target) |

- **USRP B210:** present and discoverable. `uhd_find_devices` reported
  `serial: 8000304, name: Zhixun-wireless_B210, product: B210, type: b200`.
- **LR1121 board:** present via its onboard ST-Link/V2-1 (VID `0x0483`, PID
  `0x374B`). The board also exposes a USB CDC virtual COM port (see §3). No direct
  Semtech/LR1121 USB VID was enumerated because the host link is the ST-Link
  bridge, not a native LR1121 USB device.

## 3. Detected serial device candidates

```
/dev/cu.usbmodem1303      <- LR1121 board ST-Link Virtual COM Port (UART) — candidate
/dev/tty.usbmodem1303     <- same device, tty side
/dev/cu.Bluetooth-Incoming-Port   (not relevant)
/dev/cu.debug-console             (not relevant)
```

- **`/dev/cu.usbmodem1303`** is the LR1121-board UART (ST-Link VCP). It is the
  expected terminal-side TX-log port. **It was NOT opened or monitored** in this
  read-only pass.

## 4. Whether UHD tools are installed

- ✅ Installed: `/opt/homebrew/bin/uhd_find_devices`, UHD **4.10.0.0** (Homebrew).
- ✅ B2xx firmware/FPGA images present in
  `/opt/homebrew/opt/uhd/share/uhd/images/`: `usrp_b200_fw.hex`,
  `usrp_b200_bl.img`, **`usrp_b210_fpga.bin` (4.0 MB)**, plus b200/b200mini images.
- ⚠️ Python `uhd` module is **not importable** (`import uhd` fails) — only the CLI
  is present. Any capture script that imports the Python UHD API would need the
  Python bindings installed.

## 5. Whether USRP is discoverable through read-only discovery

- ✅ **Yes.** `uhd_find_devices` enumerated one device:
  `serial 8000304 · name Zhixun-wireless_B210 · product B210 · type b200`.
- The FX3 firmware image was loaded to volatile RAM during enumeration (normal for
  B2xx discovery). **No streaming and no transmit occurred.**
- `uhd_usrp_probe` was **deliberately NOT run** in this pass: it programs the FPGA
  image and queries the daughterboard, which mutates device state beyond pure
  discovery. It is listed as a human-supervised next step (§10) instead.

## 6. ST-Link tools and read-only probe result

- ❌ ST-Link CLI tools are **NOT installed**: `st-info`, `st-flash`, `st-util` all
  "not found".
- Therefore no `st-info --probe` was possible (and none was run). The ST-Link/V2-1
  hardware itself **is** present and enumerated (VID `0x0483` PID `0x374B`).
- Flashing/firmware is out of scope for the conducted-HIL run regardless, so this
  gap does not block an IQ-level capture; it only blocks any (out-of-scope) MCU
  reflash.

## 7. Missing dependencies

| Dependency | Status | Needed for | Blocking? |
|---|---|---|---|
| UHD CLI 4.10.0.0 | ✅ installed | B210 discovery/capture | no |
| UHD B210 FPGA image | ✅ present | B210 use | no |
| Python `uhd` bindings | ❌ not importable | capture scripts that `import uhd` | maybe — verify which capture path is used |
| `stlink` (`st-info`/`st-flash`) | ❌ not installed | read-only MCU probe / (out-of-scope) flashing | no (flashing out of scope) |
| Serial monitor / pyserial | not verified here | reading LR1121 UART TX log | verify before run |
| GNU Radio | not required for the planned CLI/script capture | optional | no |

## 8. Safe next manual steps for human-supervised conducted-HIL

1. Verify the cabled/conducted RF path physically (TX → attenuator → SDR; no
   antenna radiating) — see `docs/review/mac_conducted_hil_supervised_runbook.md`.
2. (Optional) Install Python UHD bindings or confirm the capture script uses the
   CLI, before any capture.
3. Human-run `uhd_usrp_probe` once to confirm the daughterboard/clock (this loads
   the FPGA — acceptable under supervision, still no TX).
4. Confirm `/dev/cu.usbmodem1303` is the LR1121 UART by opening it **manually**
   under supervision (not in any autonomous pass).
5. Set an explicit `--nominal-center-hz` only after NCC/local frequency-plan
   confirmation (the schedule generator blocks without it — see
   `docs/review/frequency_schema_f0_guard_verification.md`).
6. Execute the conducted run by hand following the supervised runbook.

## 9. Explicit statement

**No transmit, no capture, no flash, no reset, no UART monitoring, and no RF
command was run in this pass.** Discovery was limited to `git`, `ls /dev`,
`ioreg` (read-only), `which`, and `uhd_find_devices` (enumeration only, volatile
FX3 firmware load, no streaming, no TX). `uhd_usrp_probe`, `st-info --probe`, and
all `hardware/` TX/replay/capture scripts were intentionally NOT executed.

## 10. Exact next human-supervised command checklist (DO NOT EXECUTE here)

Run these **by hand, under supervision**, only after the physical checklist in the
runbook is satisfied:

```bash
# (a) Read-only confirmation (loads FPGA; no TX):
uhd_usrp_probe --args "serial=8000304"

# (b) Confirm LR1121 UART identity (manual, read-only monitor — human only):
#     e.g. open /dev/cu.usbmodem1303 at the board's baud and observe boot banner.

# (c) Only after NCC/frequency-plan confirmation, generate a schedule with an
#     EXPLICIT carrier (generator refuses without it):
python3 hardware/ota_iq/generate_real_replay_schedule.py --nominal-center-hz <HZ> ...

# (d) Conducted capture / replay scripts (TX involved — human-supervised ONLY):
#     hardware/ota_iq/usrp_capture_ota_iq.py
#     hardware/ota_iq/replay_driver.py
#     hardware/usrp_scripts/compare_tx_on_off.py
```

These are **not** executed by this readiness pass.

---

*Cross-references:* `docs/review/mac_conducted_hil_supervised_runbook.md`,
`docs/review/mac_hil_preflight_plan.md`, `docs/review/hil_artifact_schema.md`,
`docs/review/frequency_schema_f0_guard_verification.md`.
