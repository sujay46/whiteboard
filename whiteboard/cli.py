"""Command-line interface for the whiteboard animator."""

import argparse
import os
import sys

from .engine import animate


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        prog="whiteboard",
        description=(
            "Draw an image as a whiteboard animation. "
            "Pass one image, or several cumulative step images in order."
        ),
    )
    parser.add_argument(
        "image",
        nargs="?",
        help="Single input image (PNG or JPEG).",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        metavar="IMAGE",
        help=(
            "Cumulative step images, in draw order. "
            "Each image is the picture so far; only the new ink is animated."
        ),
    )
    parser.add_argument(
        "-o", "--output",
        default="whiteboard.mp4",
        help="Output MP4 path. Default: whiteboard.mp4",
    )
    parser.add_argument("--fps", type=_positive_int, default=25, help="Frames per second. Default: 25.")
    parser.add_argument("--width", type=_positive_int, help="Output width. Default: the first image's width.")
    parser.add_argument("--height", type=_positive_int, help="Output height. Default: the first image's height.")
    parser.add_argument(
        "--split-len",
        type=_positive_int,
        default=10,
        help="Grid size in pixels for stroke drawing. Default: 10.",
    )
    parser.add_argument(
        "--skip-rate",
        type=_positive_int,
        default=7,
        help="Write one frame every N stroke cells. Default: 7.",
    )
    parser.add_argument(
        "--end-seconds",
        type=_positive_int,
        default=1,
        help="Seconds to hold the finished image. Default: 1.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.steps and args.image:
        print("Pass either one image or --steps, not both.", file=sys.stderr)
        return 2
    images = args.steps if args.steps else ([args.image] if args.image else [])
    if not images:
        build_parser().print_help()
        return 2
    missing = [path for path in images if not os.path.isfile(path)]
    if missing:
        print("Image not found:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1
    try:
        output = animate(
            images,
            args.output,
            frame_rate=args.fps,
            width=args.width,
            height=args.height,
            split_len=args.split_len,
            skip_rate=args.skip_rate,
            end_seconds=args.end_seconds,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
