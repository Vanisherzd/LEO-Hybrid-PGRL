"""Packet-delivery / PER validation framework (dry-run + future real-hardware).

DEFAULT MODE IS DRY-RUN/MOCK. Real-hardware backends are conducted/shielded
only (coax, attenuator, dummy load, or shielded enclosure). This package never
performs over-the-air tests and never raises TX power automatically.

IQ-level signal detection is NOT packet delivery: any IQ-only path reports
"PER unavailable" and never counts a packet as delivered.
"""
