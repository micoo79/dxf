#!/bin/sh
# === PyShape indítása Linuxon / Macen ===
# Előfeltétel: python3 + tkinter (Ubuntun: sudo apt install python3-tk)
cd "$(dirname "$0")" || exit 1
echo "Szükséges csomagok telepítése (első indításkor pár perc)..."
python3 -m pip install --quiet --user -r requirements.txt || \
python3 -m pip install --quiet --user --break-system-packages -r requirements.txt
echo "PyShape indítása..."
exec python3 -m pyshape gui
