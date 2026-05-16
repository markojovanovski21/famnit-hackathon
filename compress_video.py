import subprocess
import sys


if len(sys.argv) < 3:
    print("Usage:")
    print("python compress.py input.mp4 output.mp4 [crf] [preset]")
    sys.exit(1)

input_video = sys.argv[1]
output_video = sys.argv[2]

crf = "22"
preset = "fast"

if len(sys.argv) >= 4:
    crf = sys.argv[3]

if len(sys.argv) >= 5:
    preset = sys.argv[4]


def compress_video(input_video, output_video, crf, preset):

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,

        # resize
        "-vf", "scale=962:540",

        # REMOVE AUDIO (this is the key line)
        "-an",

        # video codec
        "-c:v", "libx265",
        "-crf", crf,
        "-preset", preset,

        # compatibility
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",

        output_video
    ]

    subprocess.run(cmd, check=True)


compress_video(input_video, output_video, crf, preset)