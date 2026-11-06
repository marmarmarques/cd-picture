"""Burnable CD-ROM picture images, optionally carrying an ISO 9660 + Joliet file system.

Split out of disc-probe (https://github.com/Ashley-QwQ/disc-probe, GPL-3.0) at commit
f060089. disc-probe stays the measurement tool: its `geo` run gives the spiral geometry
(summary.json) and a burned grid gives the angle correction (correction.json). This package
only reads those files and builds images.
"""
