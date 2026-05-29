import os
import subprocess

input_dir = 'my_lora_raw'
output_dir = 'my_lora_frames'
os.makedirs(output_dir, exist_ok=True)
processed = 0
failed = 0
for filename in os.listdir(input_dir):
    if not filename.lower().endswith('.mp4'):
        continue
    input_path = os.path.join(input_dir, filename)
    base_name = os.path.splitext(filename)[0]
    output_image = base_name + '.jpg'
    output_path = os.path.join(output_dir, output_image)
    cmd = ['ffmpeg', '-i', input_path, '-vf', 'select=eq(n\,0)', '-vframes', '1', '-q:v', '2', output_path, '-y']
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Extracted: {filename} → {output_image}")
        processed += 1
    except subprocess.CalledProcessError as e:
        print(f"Failed: {filename} (error: {e})")
        failed += 1
    except Exception as e:
        print(f"Unexpected error with {filename}: {e}")
        failed += 1
print("\n=== Summary ===")
print(f"Successfully extracted first frame from {processed} videos")
print(f"Failed: {failed} videos")
print(f"Frames saved to: {output_dir}")
