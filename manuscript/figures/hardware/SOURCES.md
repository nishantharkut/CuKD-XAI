# Hardware reference images (local cache)

**Purpose:** product photos for HIL / system figures and lab docs.  
**Not** photos of *our* lab bench (those should be taken on-site if required for the paper).

| File | Device | Source (downloaded) | Notes |
|---|---|---|---|
| `esp32_c3_devkitm1.png` | ESP32-C3-DevKitM-1 | Espressif docs (`esp-dev-kits` isometric render) | Primary MCU DUT image |
| `esp32_c3_devkitm1.jpg` | ESP32-C3-DevKitM-1 | Mouser product thumbnail | Alternate |
| `arduino_uno_r4_wifi.svg` | Arduino UNO R4 WiFi | Arduino docs static asset | Official product illustration |
| `arduino_uno_r4_wifi.png` | same | Rasterized from SVG via cairosvg | For LaTeX `\includegraphics` |
| `arduino_uno_r4_wifi.pdf` | same | Vector from SVG | Preferred in PDF papers |
| `raspberry_pi_5.jpg` | Raspberry Pi 5 | Adafruit product photo CDN | HIL **host** (orchestrator), not DUT |

## Usage in paper

- Prefer **schematic Graphviz/TikZ** for the system figure; use these photos only if a photo plate is needed.
- Caption must say **HIL host vs DUT** (Pi is not the classifier).
- Respect manufacturer trademarks; for IEEE, product photos are usually OK as factual hardware reference—confirm venue photo policy if using large plates.

## Re-download script

See `tools/download_hardware_images.py` (if present) or re-run downloads from URLs above if files are lost.

## Copyright

Images remain property of their respective vendors (Espressif, Arduino, Raspberry Pi / retailers). Cached here for local research figure production only.
