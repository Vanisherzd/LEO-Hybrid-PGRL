# Host-commanded per-burst LR-FHSS replay firmware

Minimal change to the Semtech SWDM001 `lr11xx_lr_fhss_ping` demo so the **host**
sets the carrier frequency and TX power **before every burst** over UART. This
unblocks the three replay modes (`no_compensation`, `sgp4_only`,
`pgrl_corrected`), which the stock firmware cannot do because it free-runs at a
fixed `RF_FREQUENCY = 868000000`.

> The firmware lives in an external vendor checkout
> (`~/Desktop/SWDM001/src/demos/lr11xx_lr_fhss_ping/`). **Do not auto-patch it.**
> Apply the change by hand in Keil, rebuild, reflash.

## What changes

- `lr11xx_lr_fhss_ping.h` — add `rf_freq_hz`, `tx_power_dbm`, `burst_index` to the demo state.
- `lr11xx_lr_fhss_ping.c` — use those state fields in `lr_fhss_send_packet()` instead of the `RF_FREQUENCY` / `POWER_IN_DBM` macros; seed them in `_init()`.
- `lr11xx_lr_fhss_ping_sync.c` — replace the free-running auto-ping loop with a command-gated loop: read one UART command, program freq/power, send one burst, ack.
- `replay_cmd.h` / `replay_cmd.c` — new line parser (copy in, add `.c` to the build).
- Provide `replay_uart_getchar()` bound to the ST-LINK VCP UART (the same UART `SXLIB_LOG` prints on).

## Wire protocol

Host → MCU, one ASCII line per burst, `\n`-terminated:

```
B <burst_index> <rf_freq_hz> <tx_power_dbm> <delay_ms>
e.g.  B 7 868018400 -9 0
```

MCU → host:

```
REPLAY_READY                 # once, at startup
RDY                          # before each command read (host may wait for it)
Packet to send: ...          # existing log
Packet sent!                 # existing log (TX_DONE)
BURST_DONE <idx> <freq> <pwr># after each burst, for capture pairing
```

## Apply / build / flash

1. Copy `replay_cmd.h` and `replay_cmd.c` into the demo dir
   `~/Desktop/SWDM001/src/demos/lr11xx_lr_fhss_ping/`.
2. Apply `lr11xx_lr_fhss_ping_host_replay.patch` by hand (Keil editor) to
   `.h`, `.c`, and `_sync.c`.
3. Implement `replay_uart_getchar()` for the board VCP USART, e.g.:
   ```c
   int replay_uart_getchar(void) {
       uint8_t b;
       if (HAL_UART_Receive(&huartX, &b, 1, HAL_MAX_DELAY) == HAL_OK) return b;
       return -1;
   }
   ```
   Use the USART that drives the ST-LINK VCP (so `/dev/cu.usbmodem*` carries it).
4. Add `replay_cmd.c` to the Keil project sources.
5. Rebuild and flash the NUCLEO-L476RG.
6. On boot the UART prints `REPLAY_READY`; the board now waits for `B ...` lines.

## Safety

- Antenna or 50-ohm load attached to the LR1121 RF output **before** any TX.
- Lowest practical TX power; the host sends the value (`--tx-power-dbm`), and you
  may additionally clamp it in firmware to your regulatory limit.
- Short bursts only; locally permitted ISM/lab frequencies only.

## Verify before trusting any replay

After flashing, dry-check the link without the USRP:

```bash
# expect REPLAY_READY at boot, then BURST_DONE acks as commands are sent
uv run python hardware/ota_iq/replay_driver.py \
    --schedule <run>/burst_schedule.csv --uart /dev/cu.usbmodem1303 \
    --tx-power-dbm -9 --out /tmp/replay_check.csv
```

If the driver reports **0 acks**, the host-replay firmware is not active (stock
SWDM001 free-runs and ignores commands) — do not claim any replay result.

## Verified LR-FHSS grid

The demo sets `LR_FHSS_V1_GRID_25391_HZ` → the LR-FHSS frequency grid spacing is
**25391 Hz**. This is the value to use for `grid_spacing_hz` in the replay
configs / adjacent-bin leakage analysis (not 137 Hz). Other demo params:
`modulation GMSK_488`, `CR 5/6`, `BW 1574219 Hz`, hopping enabled.
