import os
import cv2

videos_dir = 'videos'
images_dir = 'images'
os.makedirs(images_dir, exist_ok=True)
for video_file in os.listdir(videos_dir):
    if video_file.lower().endswith('.mp4'):
        video_path = os.path.join(videos_dir, video_file)
        base_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join(images_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = f"{base_name}_{frame_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        cap.release()
        print(f"Extracted {frame_count} frames from '{video_file}' → '{output_dir}'")
print("All videos processed successfully!")
