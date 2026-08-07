"""Download / refresh official hardware product images for HIL figures."""
from __future__ import annotations
import urllib.request
from pathlib import Path

HW = Path(__file__).resolve().parents[1] / "hardware"
HW.mkdir(parents=True, exist_ok=True)

# Only URLs that worked at last successful download
SOURCES = {
    "esp32_c3_devkitm1.png": "https://docs.espressif.com/projects/esp-dev-kits/en/latest/esp32c3/_images/esp32-c3-devkitm-1-v1-isometric.png",
    "arduino_uno_r4_wifi.svg": "https://docs.arduino.cc/static/16cbbb52c1bd49edecf62ab3fe8d1d2e/image.svg",
    "raspberry_pi_5.jpg": "https://cdn-shop.adafruit.com/970x728/5654-00.jpg",
}

def main():
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    for name, url in SOURCES.items():
        out = HW / name
        print("GET", url)
        urllib.request.urlretrieve(url, out)
        print(" ", name, out.stat().st_size)
    # rasterize arduino svg if cairosvg available
    svg = HW / "arduino_uno_r4_wifi.svg"
    if svg.exists():
        try:
            import cairosvg
            cairosvg.svg2png(url=str(svg), write_to=str(HW / "arduino_uno_r4_wifi.png"), output_width=1200)
            cairosvg.svg2pdf(url=str(svg), write_to=str(HW / "arduino_uno_r4_wifi.pdf"))
            print(" rasterized arduino png/pdf")
        except Exception as e:
            print(" cairosvg skip", e)

if __name__ == "__main__":
    main()
