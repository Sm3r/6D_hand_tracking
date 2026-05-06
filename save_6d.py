import sys

from hand_tracking.capture import capture_to_csv


def main():
    if len(sys.argv) == 2:
        print("Camera Mode: SVO")
        filename = sys.argv[1]
    else:
        print("Camera Mode: Live Streaming")
        filename = None
        timestamped = True

    if 'timestamped' not in locals():
        timestamped = False

    capture_to_csv(filename=filename, window_title='Image', timestamped=timestamped)


if __name__ == "__main__":
    main()
