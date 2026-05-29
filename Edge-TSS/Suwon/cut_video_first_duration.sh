INPUT_DIR="my_lora_raw"
OUTPUT_DIR="my_lora_scenes"
DURATION=5
mkdir -p "$OUTPUT_DIR"
for file in "$INPUT_DIR"/*.mp4; do
    [ -f "$file" ] || continue
    filename=$(basename "$file")
    output="$OUTPUT_DIR/${filename%.*}_first5s.mp4"
    dur=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null)
    if [[ ! "$dur" =~ ^[0-9]+([.][0-9]+)?$ ]] || (( $(echo "$dur < $DURATION" | bc -l) )); then
        echo "SKIP: $filename (too short: ${dur}s)"
        continue
    fi
    echo "Processing: $filename (${dur}s) → ${output##*/}"
    ffmpeg -i "$file" -t "$DURATION" -c:v copy -c:a copy -y "$output" 2>/dev/null && echo "   Done: ${output##*/}"
done
echo "Finished."
