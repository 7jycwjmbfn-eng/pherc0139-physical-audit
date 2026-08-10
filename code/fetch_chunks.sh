#!/bin/bash
# fetch a rectangular chunk range from a zarr level in the open-data bucket
set -u
PREFIX="$1"; LEVEL="$2"; DEST="$3"
Z0=$4; Z1=$5; Y0=$6; Y1=$7; X0=$8; X1=$9; PAR=${10:-16}
BASE="https://vesuvius-challenge-open-data.s3.amazonaws.com"
mkdir -p "$DEST"
curl -s -f -o "$DEST/.zarray" "$BASE/$PREFIX/$LEVEL/.zarray"
K="$DEST/.keys"; : > "$K"
for z in $(seq $Z0 $Z1); do for y in $(seq $Y0 $Y1); do for x in $(seq $X0 $X1); do
  echo "$PREFIX/$LEVEL/$z/$y/$x" >> "$K"
done; done; done
echo "[$(basename $DEST)] $(wc -l < "$K") chunks requested"
export BASE DEST PREFIX LEVEL
f(){ k="$1"; rel="${k#$PREFIX/$LEVEL/}"; d="$DEST/$rel"; [ -s "$d" ] && return 0; mkdir -p "$(dirname "$d")"; curl -s -f --max-time 300 -o "$d" "$BASE/$k" || rm -f "$d"; }
export -f f
xargs -a "$K" -P "$PAR" -I{} bash -c "f \"\$@\"" _ {}
echo "[$(basename $DEST)] got $(find "$DEST" -type f ! -name ".*" | wc -l) files, $(du -sh "$DEST" | cut -f1)"
