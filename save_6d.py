import sys

from hand_tracking.capture import capture_to_csv


def main():
    if len(sys.argv) == 2:
        print("Camera Mode: SVO")
        filename = sys.argv[1]
        show_windows = True
        print_fps = False
    else:
        print("Camera Mode: Live Streaming")
        filename = None
        timestamped = True
        show_windows = False
        print_fps = True

    if 'timestamped' not in locals():
        timestamped = False

    capture_to_csv(
        filename=filename,
        window_title='Image',
        timestamped=timestamped,
        show_windows=show_windows,
        print_fps=print_fps,
    )


if __name__ == "__main__":
    main()
