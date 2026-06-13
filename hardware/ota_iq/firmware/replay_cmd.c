/**
 * @file      replay_cmd.c
 * @brief     Host->MCU per-burst replay command parsing (representative C).
 *
 * Drop into ~/Desktop/SWDM001/src/demos/lr11xx_lr_fhss_ping/ and build with the
 * demo. Depends only on a board-provided replay_uart_getchar() (see header).
 * No dynamic allocation; fixed line buffer.
 */
#include "replay_cmd.h"

#include <stdlib.h>
#include <string.h>

#define REPLAY_LINE_MAX ( 64 )

/* Read one '\n'-terminated line (CR ignored) into buf. Returns length, or -1. */
static int replay_read_line( char* buf, int max )
{
    int n = 0;
    for( ;; )
    {
        int c = replay_uart_getchar( );
        if( c < 0 )
        {
            return -1;
        }
        if( c == '\r' )
        {
            continue;
        }
        if( c == '\n' )
        {
            buf[n] = '\0';
            return n;
        }
        if( n < ( max - 1 ) )
        {
            buf[n++] = ( char ) c;
        }
        /* overflow bytes are dropped; line still terminates on '\n' */
    }
}

bool replay_cmd_read( replay_burst_cmd_t* cmd )
{
    char line[REPLAY_LINE_MAX];
    int  len = replay_read_line( line, REPLAY_LINE_MAX );
    if( len <= 0 )
    {
        return false;
    }

    /* Expect: B <idx> <freq> <pwr> <delay> */
    char* p = line;
    while( *p == ' ' )
    {
        p++;
    }
    if( *p != 'B' )
    {
        return false;
    }
    p++;

    char*         end   = NULL;
    unsigned long idx   = strtoul( p, &end, 10 );
    if( end == p )
    {
        return false;
    }
    p = end;
    unsigned long freq = strtoul( p, &end, 10 );
    if( end == p )
    {
        return false;
    }
    p = end;
    long pwr = strtol( p, &end, 10 );
    if( end == p )
    {
        return false;
    }
    p = end;
    unsigned long delay = strtoul( p, &end, 10 ); /* optional; defaults to 0 */
    if( end == p )
    {
        delay = 0;
    }

    cmd->burst_index  = ( uint32_t ) idx;
    cmd->rf_freq_hz   = ( uint32_t ) freq;
    cmd->tx_power_dbm = ( int8_t ) pwr;
    cmd->delay_ms     = ( uint32_t ) delay;
    return true;
}
