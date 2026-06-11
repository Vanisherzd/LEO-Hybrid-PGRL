/**
 * @file      replay_cmd.h
 * @brief     Host->MCU per-burst replay command parsing for the LR-FHSS ping demo.
 *
 * REPRESENTATIVE C — drop into the SWDM001 demo dir
 *   ~/Desktop/SWDM001/src/demos/lr11xx_lr_fhss_ping/
 * and apply the companion patch by hand in Keil, then rebuild/reflash.
 * Do NOT auto-patch the vendor tree.
 *
 * Wire protocol (one ASCII line per burst, '\n' terminated), host -> MCU:
 *   B <burst_index> <rf_freq_hz> <tx_power_dbm> <delay_ms>
 * e.g.  B 7 868018400 -9 0
 *
 * MCU -> host after transmit (see patched sync loop):
 *   BURST_DONE <burst_index> <rf_freq_hz> <tx_power_dbm>
 */
#ifndef REPLAY_CMD_H_
#define REPLAY_CMD_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct replay_burst_cmd_s
{
    uint32_t burst_index;  /**< host-provided index, echoed in ack for capture pairing */
    uint32_t rf_freq_hz;   /**< per-burst center frequency to program before TX */
    int8_t   tx_power_dbm; /**< per-burst TX power */
    uint32_t delay_ms;     /**< optional pre-burst delay (0 = none) */
} replay_burst_cmd_t;

/**
 * @brief Board UART single-byte read, blocking. MUST be provided by the port.
 *
 * Bind this to the same VCP UART that SXLIB_LOG prints on. On the NUCLEO-L476RG
 * SWDM001 port this is typically the ST-LINK VCP USART. Example STM32 HAL body:
 *
 *   int replay_uart_getchar(void) {
 *       uint8_t b;
 *       if (HAL_UART_Receive(&huartX, &b, 1, HAL_MAX_DELAY) == HAL_OK) return b;
 *       return -1;
 *   }
 *
 * Return the byte (0..255) or -1 on error/timeout.
 */
extern int replay_uart_getchar( void );

/**
 * @brief Block until one full command line is received and parse it.
 *
 * Lines not beginning with 'B' (e.g. stray bytes) are skipped. Returns true and
 * fills *cmd on a well-formed "B idx freq pwr delay" line; returns false on a
 * malformed line so the caller can re-prompt.
 */
bool replay_cmd_read( replay_burst_cmd_t* cmd );

#ifdef __cplusplus
}
#endif

#endif /* REPLAY_CMD_H_ */
