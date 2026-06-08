# SWDM001 automated TX modes (no manual reset)

Goal: get an LR-FHSS burst to land **inside** the USRP B210 capture window
without a human pressing the NUCLEO reset button between captures. This matters
for the automated sweep orchestrator
`scripts/run_lr1121_sdr_sweep_auto.py`, which starts the SDR capture, then
needs RF energy to appear during the (short, safe-default) 10 s window.

> **Scope discipline (read first).** Everything here is IQ-level only. A
> `"Packet sent!"` UART line proves the firmware TX command completed; it does
> **not** prove RF was captured. The signal detector
> (`validation_status == "signal_detected"`, DC spike excluded) is the **sole**
> authority for RF waveform detection. Never claim LR-FHSS RF capture, PER, or
> standard decoding from UART logs alone.

---

## How the stock SWDM001 ping demo actually behaves

Verified by reading the firmware
(`~/Desktop/SWDM001/src/demos/lr11xx_lr_fhss_ping/`):

- `lr11xx_lr_fhss_ping_sync.c` already runs a `while(1)` loop: it launches one
  LR-FHSS TX, waits for the `TX_DONE` IRQ, prints `Packet sent!`, sleeps for
  `inter_pkt_delay`, then sends the next packet. So the demo is **already a
  continuous, free-running pinger** — it does not stop after one packet.
- BUT the inter-packet delay is large:
  `lr11xx_lr_fhss_ping_start.c` sets `#define INTER_PKT_DELAY_IN_MS ( 20000 )`,
  i.e. **one short LR-FHSS burst every 20 seconds**. With the safe 10 s capture
  window, a burst frequently falls outside the window — which is exactly why a
  manual reset (to time a fresh burst into the window) has been used.
- Default RF params (verified):
  - `RF_FREQUENCY ( 868000000 )` — 868 MHz
  - `POWER_IN_DBM ( -9 )` — TX power macro (note: low, -9 dBm)
  in `src/demos/lr11xx_lr_fhss_ping/lr11xx_lr_fhss_ping.c`.

Two ways to make a burst reliably land inside the capture window:

---

## Mode A — no firmware change (reset-triggered burst)

Use the stock firmware as-is. The sweep orchestrator times a fresh post-reset
ping into the capture window:

1. `run_lr1121_sdr_sweep_auto.py` starts the USRP capture.
2. It waits `--reset-delay-s` seconds (so the SDR is already recording).
3. It resets the NUCLEO. The firmware reboots, prints `LR11XX-LR-FHSS Ping
   Init`, then almost immediately launches the first LR-FHSS ping — which now
   lands inside the open capture window.
4. The UART logger watches for `Packet sent!` as a TX-attempt confirmation.
5. The signal detector decides whether RF was actually received.

### Reset options (`--reset-method`)

| Method | How | Prerequisite / caveat |
|--------|-----|-----------------------|
| `stlink` | runs `st-flash reset` | needs `brew install stlink` (st-flash/st-info are **not** installed on this Mac). Without it the sweep warns and performs **no** reset. |
| `serial-dtr` | best-effort DTR/RTS toggle over the UART via pyserial | needs `uv add pyserial` (pyserial is **not** installed). May **not** reset all NUCLEO boards — the ST-LINK VCP DTR line does not necessarily drive NRST. Verify it actually reboots by watching for a fresh `LR11XX-LR-FHSS Ping Init`. |
| `manual` | sweep prompts/pauses; you press the black NUCLEO reset button | always works; not hands-free. |
| `none` | no reset attempted | use this with Mode B firmware. |

### Stress / honesty note

`Packet sent!` on UART only proves the firmware TX command pipeline completed.
It does **not** prove RF was emitted on the right port, nor that the SDR
captured it. The authority is `validation_status == "signal_detected"`. The
validation label stays at `firmware_running` / `iq_capture_done` (hardware
bring-up / pending) until the detector passes.

### Exact run command (Mode A)

```bash
uv run python scripts/run_lr1121_sdr_sweep_auto.py \
    --serial 8000304 --freqs "868e6,915e6,923e6" --antennas "RX2,TX/RX" \
    --rate 1e6 --gain 45 --duration 10 \
    --reset-method stlink --reset-delay-s 2 \
    --uart /dev/tty.usbmodem1303 --uart-baud 115200
```

If `stlink` is not installed or no probe is found, swap in
`--reset-method manual` (you press reset when prompted) or
`--reset-method serial-dtr` (after `uv add pyserial`; confirm it really
reboots). Safe defaults are `--rate 1e6 --gain 45 --duration 10`;
`2e6 / 30s` caused a UHD overflow on this Mac.

Trade-off: no rebuild needed, but you only get **one** TX burst per capture and
you depend on a working reset path that is timed into the window.

---

## Mode B — firmware continuous TX (no reset needed)

Make the firmware emit bursts frequently enough that **any** capture window
catches several, so no reset timing is required. Because the demo already loops
forever, the minimal change is to **shrink the inter-packet delay** (and,
optionally, print diagnostics per burst). Then run the sweep with
`--reset-method none`.

> **Do NOT auto-patch the external repo.** The SWDM001 tree at
> `~/Desktop/SWDM001` is a separate vendor checkout. The instructions below are
> **representative C only** — apply them by hand in Keil, rebuild, and reflash.
> This document does not (and must not) edit that repo.

### B.1 Minimal change — shorten the inter-packet delay

File: `src/demos/lr11xx_lr_fhss_ping/lr11xx_lr_fhss_ping_start.c`

Change the existing macro from 20 s to e.g. 1 s so a burst lands in every
capture:

```c
/* was: #define INTER_PKT_DELAY_IN_MS ( 20000 ) */
#define INTER_PKT_DELAY_IN_MS ( 1000 )
```

The existing `while(1)` loop in `lr11xx_lr_fhss_ping_sync.c` then free-runs:
`launch -> wait TX_DONE -> Packet sent! -> sleep 1 s -> continue -> launch ...`.
No new loop is required — the demo is already continuous; you are only changing
the cadence.

### B.2 Optional — print counter / frequency / power before each TX

For easier correlation between UART logs and captures, add diagnostic prints.
The natural place is inside `lr11xx_lr_fhss_ping_launch()` in
`src/demos/lr11xx_lr_fhss_ping/lr11xx_lr_fhss_ping.c`, just before the existing
`lr_fhss_send_packet(...)` call. Representative snippet (adapt to the project's
`SXLIB_LOG` macros — `RF_FREQUENCY` and `POWER_IN_DBM` are already defined at
the top of that file):

```c
void lr11xx_lr_fhss_ping_launch( lr11xx_lr_fhss_ping_state_t* state )
{
    static uint32_t tx_counter = 0;
    sxlib_Gpio_Led_on( state->config->interface->led_interface.led_tx );

    /* --- added: per-burst diagnostics --- */
    SXLIB_LOG( SXLIB_LOG_APP,
        ( "TX #%lu freq=%lu Hz power=%d dBm len=%u" SXLIB_LOG_EOL,
          (unsigned long) tx_counter++,
          (unsigned long) RF_FREQUENCY,
          (int) POWER_IN_DBM,
          (unsigned) state->payload_length ) );
    /* ------------------------------------- */

    lr11xx_status_t status = lr_fhss_send_packet( state, state->payload, state->payload_length );
    if( status != LR11XX_STATUS_OK )
    {
        sxlib_Gpio_Led_off( state->config->interface->led_interface.led_tx );
        SXLIB_LOG( SXLIB_LOG_APP, ( "Failed status=%d" SXLIB_LOG_EOL, status ) );
    }
}
```

Notes:
- `lr11xx_lr_fhss_ping_launch()` is the single function both the first packet
  (`..._start.c`) and every subsequent packet (`..._continue` ->
  `..._launch`) go through, so wrapping it covers all bursts.
- If you also want a different frequency or higher TX power, edit the
  `RF_FREQUENCY` / `POWER_IN_DBM` macros at the top of
  `lr11xx_lr_fhss_ping.c` (current values: 868 MHz, -9 dBm). Keep the SDR
  capture centre frequency in sync with whatever the firmware actually
  transmits. The stock `-9 dBm` is low; a higher (legal, attenuated) power
  makes the burst easier for the SDR to see.

### B.3 Rebuild, reflash, then run the sweep with no reset

Build in Keil (target `lr1121_xtal`,
`project/keil_polling_STM32L476/STM32L476.uvprojx`) and flash
`STM32L476_lr11xx.bin` to `NOD_L476RG`. Then:

```bash
uv run python scripts/run_lr1121_sdr_sweep_auto.py \
    --serial 8000304 --freqs "868e6,915e6,923e6" --antennas "RX2,TX/RX" \
    --rate 1e6 --gain 45 --duration 10 \
    --reset-method none \
    --uart /dev/tty.usbmodem1303 --uart-baud 115200
```

Same honesty rule applies: continuous `TX #N ...` / `Packet sent!` lines on
UART confirm the firmware is transmitting, but only
`validation_status == "signal_detected"` confirms RF was actually received.

---

## Mode A vs Mode B trade-offs

| | Mode A (reset-triggered) | Mode B (continuous TX) |
|--|--------------------------|------------------------|
| Firmware rebuild | none | Keil rebuild + reflash |
| RF availability | one burst per capture, timed into window | continuous bursts (every ~1 s) |
| Depends on reset path | yes (stlink / serial-dtr / manual) | no (`--reset-method none`) |
| Easiest for SDR to catch | no | **yes** |
| Best when | you cannot rebuild firmware | you can reflash and want robust capture |

Recommendation: if you can rebuild SWDM001, use **Mode B** — continuous RF is
by far the easiest for the USRP to catch and removes the fragile reset-timing
dependency.
