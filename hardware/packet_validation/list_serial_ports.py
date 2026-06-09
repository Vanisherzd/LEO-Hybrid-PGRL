"""List candidate Nucleo / ST-Link / USB-serial ports (macOS focused).

Works without pyserial (globs /dev). If pyserial is present, also prints
manufacturer/product/serial when available. No hardware required.
"""
from __future__ import annotations

import glob
import sys

PATTERNS = [
    "/dev/cu.usbmodem*", "/dev/tty.usbmodem*",
    "/dev/cu.usbserial*", "/dev/tty.usbserial*",
]


def glob_ports():
    found = []
    for pat in PATTERNS:
        found.extend(sorted(glob.glob(pat)))
    return found


def main(argv=None):
    print("Candidate Nucleo / ST-Link modem ports")
    print("Recommended use: /dev/cu.* for serial programs on macOS")
    print("-" * 56)

    ports = glob_ports()
    if not ports:
        print("(no /dev/cu.usbmodem* or usbserial* ports found)")
    else:
        for p in ports:
            print(f"  {p}")

    # optional richer info via pyserial
    try:
        from serial.tools import list_ports  # type: ignore
        print("\npyserial device details:")
        for info in list_ports.comports():
            print(f"  {info.device}  | mfg={info.manufacturer} "
                  f"product={info.product} serial={info.serial_number}")
    except Exception:  # noqa: BLE001
        print("\n(pyserial not installed; showing globbed paths only. "
              "Install with: uv add pyserial)")

    print("\nTip: plug TX board only -> note port; then RX board only -> note "
          "port; then both -> confirm two distinct /dev/cu.* ports.")
    return ports


if __name__ == "__main__":
    main()
