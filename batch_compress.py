import subprocess
from pathlib import Path
import sys

# Directory containing compress_video.py
BASE_DIR = Path(__file__).resolve().parent

# Input directory passed as first argument
if len(sys.argv) < 2:
    print("Usage: python batch_compress.py <input_directory>")
    sys.exit(1)

input_dir = Path(sys.argv[1]).resolve()

# Output directory
output_dir = BASE_DIR / "compressed_videos"
output_dir.mkdir(exist_ok=True)

# Path to compress_video.py
compress_script = BASE_DIR / "compress_video.py"

# Loop through all mp4 files
for video_file in input_dir.glob("*.mp4"):

    output_file = output_dir / f"{video_file.stem}_compressed.mp4"

    print(f"Compressing: {video_file.name}")

    subprocess.run([
        sys.executable,
        str(compress_script),
        str(video_file),
        str(output_file)
    ])

    print(f"Saved: {output_file}")

print("All videos compressed.")