INPUT_DIR="my_lora_raw"
OUTPUT_DIR="my_lora_scenes"
FRAME_COUNT=93
mkdir -p "$OUTPUT_DIR"
for file in "$INPUT_DIR"/*.mp4; do
    [ -f "$file" ] || continue
    filename=$(basename "$file")
    base="${filename%.*}"
    output="$OUTPUT_DIR/${base}_first93frames.mp4"
    echo "Analyzing: $filename"
    frame_count=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 "$file" 2>/dev/null)
    if [[ "$frame_count" == "N/A" || -z "$frame_count" ]]; then
        frame_count=$(ffmpeg -i "$file" -f null - 2>&1 | grep -oP 'frame=\s*\K\d+' | tail -1)
        if [[ -z "$frame_count" ]]; then
            echo "  → Cannot determine frame count, skipping"
            continue
        fi
    fi
    if (( frame_count < FRAME_COUNT )); then
        echo "  → Too short ($frame_count frames < $FRAME_COUNT), skipping"
        continue
    fi
    echo "Processing: $filename (${frame_count} frames) → ${output##*/}"
    ffmpeg -y -i "$file" -frames:v "$FRAME_COUNT" -c:v libx264 -preset ultrafast -crf 23 -c:a copy "$output" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  Done: ${output##*/}"
    else
        echo "  Failed: $filename"
    fi
    echo
done
echo "Finished processing all videos."
