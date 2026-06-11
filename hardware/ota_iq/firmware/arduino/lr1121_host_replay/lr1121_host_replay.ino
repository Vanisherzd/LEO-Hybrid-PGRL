/*
  lr1121_host_replay.ino
  ----------------------
  Host-commanded per-burst LR-FHSS replay for NUCLEO-L476RG + Semtech LR1121,
  using RadioLib. The host sets center frequency and TX power before EVERY burst
  over the ST-LINK VCP serial, so the OTA IQ replay modes (no_compensation /
  sgp4_only / pgrl_corrected) can each command a distinct per-burst carrier.

  FQBN:
    STMicroelectronics:stm32:Nucleo_64:pnum=NUCLEO_L476RG,xserial=generic,usb=none,upload_method=MassStorage

  Wire protocol (115200 8N1):
    host -> MCU :  B <burst_index> <rf_freq_hz> <tx_power_dbm> <delay_ms>
    MCU  -> host:  REPLAY_READY            (once, at boot)
                   RDY                     (before each command read)
                   Packet sent!            (after a successful burst)
                   BURST_DONE <idx> <freq> <pwr>

  LR-FHSS params match the SWDM001 ping demo measured earlier:
    CR 5/6, BW 1574.2 kHz, FCC grid (25391 Hz), hopping enabled.

  SAFETY: antenna or 50-ohm load on the LR1121 RF output before any TX;
  lowest practical TX power; short bursts; permitted ISM/lab frequency only.

  Pin mapping (same as the working board — do NOT change):
    MOSI D11  MISO D12  SCK D13  (default STM32 SPI1)
    NSS  D10  RST  D8   BUSY D9   IRQ D3
*/

#include <RadioLib.h>

// ---- LR1121 module: Module(cs/NSS, irq/IRQ, rst/RST, busy/BUSY) -------------
// Pins from the known-good LR1121_TX_RX_test sketch (NSS=D7, IRQ=D5, RST=A0,
// BUSY=D3). The D10/D3/D8/D9 mapping gave -707 SPI_CMD_FAILED on this board.
LR1121 radio = new Module(D7, D5, A0, D3);

// ---- RF switch table: Semtech LR1121 EVK + STM32L476 (DIO5/DIO6) ------------
static const uint32_t rfswitch_dio_pins[] = {
  RADIOLIB_LR11X0_DIO5, RADIOLIB_LR11X0_DIO6,
  RADIOLIB_NC, RADIOLIB_NC, RADIOLIB_NC
};
static const Module::RfSwitchMode_t rfswitch_table[] = {
  { LR11x0::MODE_STBY,  { LOW,  LOW  } },
  { LR11x0::MODE_RX,    { HIGH, LOW  } },
  { LR11x0::MODE_TX,    { HIGH, HIGH } },  // RFO_LP_LF
  { LR11x0::MODE_TX_HP, { LOW,  HIGH } },  // RFO_HP_LF
  { LR11x0::MODE_TX_HF, { LOW,  LOW  } },
  { LR11x0::MODE_GNSS,  { LOW,  LOW  } },
  { LR11x0::MODE_WIFI,  { LOW,  LOW  } },
  END_OF_MODE_TABLE,
};

// ---- LR-FHSS configuration (match SWDM001 measured params) ------------------
static const uint8_t LRFHSS_BW   = RADIOLIB_LRXXXX_LR_FHSS_BW_1574_2; // 1574.2 kHz
static const uint8_t LRFHSS_CR   = RADIOLIB_LRXXXX_LR_FHSS_CR_5_6;    // 5/6
static const bool    NARROW_GRID = false;  // false => FCC grid 25391 Hz (LR_FHSS_V1_GRID_25391_HZ)
static const float   TCXO_V      = 0.0f;   // match known-good LR1121_TX_RX_test (no TCXO ref)

// fixed short payload (presence/structure only; content irrelevant to IQ proxy)
static uint8_t payload[19] = {
  0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,0x29,0x2a,
  0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19
};

static bool    radio_ready = false;
static int16_t init_code   = 0;   // last beginLRFHSS() result, for diagnostics

// Bring the LR1121 up if it is not already; report the code on failure.
static bool ensure_radio_ready() {
  if (radio_ready) return true;
  init_code = radio.beginLRFHSS(868.0f, LRFHSS_BW, LRFHSS_CR, NARROW_GRID, 10, TCXO_V);
  radio.setRfSwitchTable(rfswitch_dio_pins, rfswitch_table);
  radio_ready = (init_code == RADIOLIB_ERR_NONE);
  if (!radio_ready) {
    Serial.print(F("ERR radio_not_ready code ")); Serial.println(init_code);
  }
  return radio_ready;
}

static bool configure_burst(uint32_t rf_freq_hz, int8_t power_dbm) {
  // Re-init LR-FHSS at the commanded frequency + power for this burst.
  float freq_mhz = (float)((double)rf_freq_hz / 1.0e6);
  int16_t st = radio.beginLRFHSS(freq_mhz, LRFHSS_BW, LRFHSS_CR, NARROW_GRID, power_dbm, TCXO_V);
  if (st != RADIOLIB_ERR_NONE) {
    Serial.print(F("ERR beginLRFHSS code ")); Serial.println(st);
    return false;
  }
  radio.setRfSwitchTable(rfswitch_dio_pins, rfswitch_table);
  return true;
}

void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && (millis() - t0) < 3000) { }

  // initial LR-FHSS bring-up at a default freq/power; overwritten per burst
  radio.setRfSwitchTable(rfswitch_dio_pins, rfswitch_table);
  init_code = radio.beginLRFHSS(868.0f, LRFHSS_BW, LRFHSS_CR, NARROW_GRID, 10, TCXO_V);
  radio_ready = (init_code == RADIOLIB_ERR_NONE);
  Serial.print(F("INIT code ")); Serial.println(init_code);
  if (!radio_ready) {
    Serial.print(F("ERR init beginLRFHSS code ")); Serial.println(init_code);
  }

  Serial.println(F("REPLAY_READY"));
}

void loop() {
  Serial.println(F("RDY"));

  // block for one command line
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) {
    return;  // re-prompt
  }
  char op = line.charAt(0);
  if (op != 'B' && op != 'C') {
    return;  // ignore non-command lines, re-prompt
  }
  if (!ensure_radio_ready()) {
    return;  // ensure_radio_ready already reported the code
  }

  if (op == 'B') {
    // ---- LR-FHSS hopping burst: B <idx> <freq_hz> <pwr_dbm> <delay_ms> ----
    long idx = 0, freq = 0, pwr = 0, delay_ms = 0;
    if (sscanf(line.c_str(), "B %ld %ld %ld %ld", &idx, &freq, &pwr, &delay_ms) < 3) {
      Serial.println(F("ERR parse")); return;
    }
    if (delay_ms > 0) delay((uint32_t)delay_ms);
    if (!configure_burst((uint32_t)freq, (int8_t)pwr)) return;
    int16_t tx = radio.transmit(payload, sizeof(payload));
    if (tx == RADIOLIB_ERR_NONE) {
      Serial.println(F("Packet sent!"));
      Serial.print(F("BURST_DONE ")); Serial.print(idx); Serial.print(' ');
      Serial.print(freq); Serial.print(' '); Serial.println(pwr);
    } else {
      Serial.print(F("ERR transmit code ")); Serial.println(tx);
    }
    return;
  }

  // ---- C: non-hopping fixed-channel GFSK diagnostic -----------------------
  //   C <idx> <freq_hz> <pwr_dbm> <duration_ms> <delay_ms>
  // True CW (transmitDirect()->setTxCw()) returned OK but did NOT radiate on
  // this board (USRP saw only noise floor), so the C command uses the narrow
  // FIXED-CHANNEL GFSK path (no hopping) over the known-good transmit() RF
  // path: back-to-back GFSK packets at rf_freq_hz for ~duration_ms. For
  // IQ-level CFO / feedforward validation only; NOT LR-FHSS / PER / decoding.
  long cidx = 0, cfreq = 0, cpwr = 0, cdur = 0, cdelay = 0;
  if (sscanf(line.c_str(), "C %ld %ld %ld %ld %ld", &cidx, &cfreq, &cpwr, &cdur, &cdelay) < 4) {
    Serial.println(F("ERR parse")); return;
  }
  if (cdelay > 0) delay((uint32_t)cdelay);

  float cfreq_mhz = (float)((double)cfreq / 1.0e6);
  // beginGFSK(freq, br=4.8kbps, freqDev=5kHz, rxBw=156.2kHz, power, preamble=16, tcxo)
  int16_t gs = radio.beginGFSK(cfreq_mhz, 4.8, 5.0, 156.2, (int8_t)cpwr, 16, TCXO_V);
  radio.setRfSwitchTable(rfswitch_dio_pins, rfswitch_table);
  if (gs != RADIOLIB_ERR_NONE) {
    Serial.print(F("ERR cw_cfg beginGFSK code ")); Serial.println(gs);
    return;
  }
  // beginGFSK has configured PA + RF switch + freq; now emit a TRUE CW carrier
  // (setTxCw) which is a single narrow tone — needed to resolve sub-kHz offsets
  // that GFSK deviation would smear. Falls back to GFSK packets if CW fails.
  int16_t cw = radio.transmitDirect();   // setTxCw() -> CW at configured freq/power
  if (cw == RADIOLIB_ERR_NONE) {
    delay((uint32_t)(cdur > 0 ? cdur : 50));
    radio.standby();
  } else {
    uint32_t t_end = millis() + (uint32_t)(cdur > 0 ? cdur : 50);
    int16_t tx = RADIOLIB_ERR_NONE;
    do { tx = radio.transmit(payload, sizeof(payload)); }
    while ((int32_t)(millis() - t_end) < 0 && tx == RADIOLIB_ERR_NONE);
    if (tx != RADIOLIB_ERR_NONE) {
      Serial.print(F("ERR cw_tx code ")); Serial.println(tx);
      return;
    }
  }
  Serial.println(F("Carrier sent!"));
  Serial.print(F("CARRIER_DONE ")); Serial.print(cidx); Serial.print(' ');
  Serial.print(cfreq); Serial.print(' '); Serial.print(cpwr); Serial.print(' ');
  Serial.println(cdur);
}
