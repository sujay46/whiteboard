# Whiteboard

Draws a still image as a whiteboard animation: a hand traces the ink, then the finished picture holds on screen. Pass one image, or several cumulative step images.

The drawing engine comes from [Storyboard AI](https://github.com/yogendra-yatnalkar/storyboard-ai) by [Yogendra Yatnalkar](https://github.com/yogendra-yatnalkar). This folder is that animator on its own, without the rest of the pipeline.

The package is `whiteboard/`. It depends only on OpenCV and NumPy. It does not call Gemini or SAM.

## Install

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## One image

```bash
python -m whiteboard path/to/image.png -o out.mp4
```

The whole picture is drawn in one pass. Lines are traced as strokes. Text is written left to right, one character at a time.

## Steps

Each step image is the drawing so far. The tool compares it with the previous step and animates only the new ink, so earlier strokes are not redrawn.

```bash
python -m whiteboard --steps \
  examples/step_1.png \
  examples/step_2.png \
  examples/step_3.png \
  examples/step_4.png \
  -o out.mp4
```

You do not need mask images. Masks are computed from the difference between steps.

Pass either one image or `--steps`, not both.

## Resolution and timing

If you omit the size, the video uses the first image's width and height. Pass one of `--width` or `--height` and the other side keeps the aspect ratio.

```bash
python -m whiteboard image.png -o out.mp4 \
  --width 1280 --height 720 \
  --fps 30 \
  --skip-rate 12 \
  --split-len 14 \
  --end-seconds 2
```

| Flag | Default | Effect |
| --- | --- | --- |
| `-o`, `--output` | `whiteboard.mp4` | Output path |
| `--width`, `--height` | first image | Output resolution |
| `--fps` | `25` | Playback speed of the whole video, including text |
| `--skip-rate` | `7` | Stroke speed. A frame is written every N stroke cells. Higher is faster, lower is slower. |
| `--split-len` | `10` | Stroke cell size in pixels. Larger cells finish a line sooner. Smaller cells follow the line more tightly. |
| `--end-seconds` | `1` | How long the finished image stays on screen |

`--skip-rate` and `--split-len` affect lines and shapes only. Text is fixed in `whiteboard/engine.py`: each letter is two frames, and a dot or dash is one frame. Change `_write_glyph` to speed text up or slow it down. `--fps` scales both strokes and text because it changes how fast those frames play.

The file is an MP4 written with OpenCV's `mp4v` codec. If a player will not open it, re-encode to H.264 with ffmpeg.

## From Python

```python
from whiteboard.engine import animate

animate(
    ["examples/step_1.png", "examples/step_2.png", "examples/step_3.png", "examples/step_4.png"],
    "out.mp4",
    width=1280,
    height=720,
    frame_rate=25,
    skip_rate=7,
    split_len=10,
    end_seconds=1,
)
```

`animate` returns the absolute path of the MP4.

## Move it to another project

Copy the `whiteboard` directory, including `assets/drawing-hand.png` and `assets/hand-mask.png`. The hand images are loaded from `whiteboard/assets` next to `engine.py`.

```bash
pip install opencv-python numpy
python -m whiteboard image.png -o out.mp4
```

Run that from the directory that contains the `whiteboard` folder. You do not need the rest of this repo.
