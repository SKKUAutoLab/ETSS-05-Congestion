import os.path
from pathlib import Path
from PIL import Image
import cv2
import argparse

def convert_mp4_to_gif(mp4_path, output_path, max_frames=300):
    cap = cv2.VideoCapture(str(mp4_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_use = min(total_frames, max_frames)
    print(f"Process: {mp4_path.name} video with the total number of frame: {total_frames} frames")
    frames = []
    frame_count = 0
    while frame_count < frames_to_use:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        frame_count += 1
    cap.release()
    if frames:
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0) # 100ms -> 10 fps
        print(f"Saved GIF file at: {output_path.name}")
    else:
        print(f"Warning: No frames extracted from {mp4_path.name}")

def main(args):
    source_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    subdirs = [d for d in source_dir.iterdir() if d.is_dir()]
    for subdir in subdirs:
        mp4_files = list(subdir.glob("*.mp4"))
        if not mp4_files:
            print(f"No MP4 file found in {subdir.name}")
            continue
        for mp4_file in mp4_files:
            gif_filename = mp4_file.stem + ".gif"
            output_path = output_dir / gif_filename
            convert_mp4_to_gif(mp4_file, output_path, max_frames=args.max_frames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type_dataset', type=str, default='SENSE')
    parser.add_argument('--input_dir', type=str, default='saved_sense/vis')
    parser.add_argument('--output_dir', type=str, default='saved_sense_gif')
    parser.add_argument('--max_frames', type=int, default=300)
    args = parser.parse_args()

    print('Process dataset:', args.type_dataset)
    main(args)