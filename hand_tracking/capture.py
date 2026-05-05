import os
from datetime import datetime
import cv2
import numpy as np
import pandas as pd

from .tracking import HandTracking
from .zed import Zed
import pyzed.sl as sl


def _initialize_name_dict():
    name_dict = {}
    name_dict['Frame'] = []

    # Palm centroid + orientation fields for left and right
    basic_keys = [
        'left X', 'left Y', 'left Z', 'left Yaw', 'left Pitch', 'left Roll',
        'right X', 'right Y', 'right Z', 'right Yaw', 'right Pitch', 'right Roll',
    ]
    for k in basic_keys:
        name_dict[k] = []

    # Flattened landmark columns: 'left 0 X', 'left 0 Y', 'left 0 Z', ...
    for hand in ['left', 'right']:
        for i in range(21):
            for axis in ['X', 'Y', 'Z']:
                name = f"{hand} {i} {axis}"
                name_dict[name] = []

    return name_dict


def capture_to_csv(filename=None, output_csv=None, window_title='Image', timestamped=False):
    """Capture hand keypoints from a ZED camera or SVO and save to CSV.

    - `filename`: path to SVO file. If None, uses live camera.
    - `output_csv`: optional path to save CSV. If omitted, a sensible default is used.
    - `window_title`: title for the OpenCV window.
    - `timestamped`: when True and no explicit `output_csv` is given, create a timestamped live filename.
    """

    os.makedirs('results', exist_ok=True)

    detector = HandTracking(maxHands=2, detectionCon=0.1, trackCon=0.8, complexity=1)
    cam = Zed(filename)

    cam.print_information()
    if filename:
        try:
            final_frame = cam.zed.get_svo_number_of_frames()
        except Exception:
            final_frame = float('inf')
    else:
        final_frame = float('inf')

    camera_params = cam.camera_params

    name_dict = _initialize_name_dict()

    frame = 0
    lx = ly = lz = lyaw = lpitch = lroll = 0
    rx = ry = rz = ryaw = rpitch = rroll = 0
    first_print = True

    while frame <= final_frame:
        err = cam.zed.grab(cam.runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
            frame += 1
            cam.get_image()
            img = cam.img
            depth_img = cam.depth_img

            pcl = cam.point_cloud

            img = detector.findHands(img)
            data_left, data_right = detector.findpostion(depth_img, pcl, camera_params)

            cv2.imshow(window_title, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            left_orientaton = detector.calculate_orientation(data_left)
            left_centroid = detector.calculate_centroid(data_left)

            right_orientaton = detector.calculate_orientation(data_right)
            right_centroid = detector.calculate_centroid(data_right)

            if isinstance(data_left, np.ndarray) and data_left.shape == (21, 3):
                lx, ly, lz = left_centroid
                lyaw, lpitch, lroll = left_orientaton

            if isinstance(data_right, np.ndarray) and data_right.shape == (21, 3):
                rx, ry, rz = right_centroid
                ryaw, rpitch, rroll = right_orientaton

            # append basic palm 6D info
            name_dict['Frame'].append(frame)
            name_dict['left X'].append(lx)
            name_dict['left Y'].append(ly)
            name_dict['left Z'].append(lz)
            name_dict['left Yaw'].append(lyaw)
            name_dict['left Pitch'].append(lpitch)
            name_dict['left Roll'].append(lroll)
            name_dict['right X'].append(rx)
            name_dict['right Y'].append(ry)
            name_dict['right Z'].append(rz)
            name_dict['right Yaw'].append(ryaw)
            name_dict['right Pitch'].append(rpitch)
            name_dict['right Roll'].append(rroll)

            # Convert landmark lists to numpy arrays once to avoid repeated conversions
            if isinstance(data_left, (list, tuple)):
                dl = np.array(data_left)
            else:
                dl = data_left

            if isinstance(data_right, (list, tuple)):
                dr = np.array(data_right)
            else:
                dr = data_right

            # Append flattened 3D landmarks for left and right hands.
            for i in range(21):
                # left
                for axis_idx, axis in enumerate(['X', 'Y', 'Z']):
                    col = f"left {i} {axis}"
                    if isinstance(dl, np.ndarray) and dl.shape == (21, 3):
                        name_dict[col].append(float(dl[i, axis_idx]))
                    else:
                        name_dict[col].append(np.nan)

                # right
                for axis_idx, axis in enumerate(['X', 'Y', 'Z']):
                    col = f"right {i} {axis}"
                    if isinstance(dr, np.ndarray) and dr.shape == (21, 3):
                        name_dict[col].append(float(dr[i, axis_idx]))
                    else:
                        name_dict[col].append(np.nan)

            # Print detection and frame count on two fixed lines and refresh in-place
            detection = getattr(detector, 'detection_str', '')
            frame_line = f"Frame count: {frame}" + (f" / {final_frame}" if filename else "")
            if not first_print:
                print("\x1b[2A", end='')
            print(detection.ljust(80), flush=True)
            print(frame_line.ljust(80), flush=True)
            first_print = False
        else:
            break

    # Save final results
    df = pd.DataFrame(name_dict)
    if output_csv is None:
        if filename:
            base = os.path.splitext(os.path.basename(filename))[0]
            output_csv = os.path.join('results', base + '.csv')
        elif timestamped:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_csv = os.path.join('results', f'live_{timestamp}.csv')
        else:
            output_csv = os.path.join('results', 'output.csv')

    df.to_csv(output_csv, index=False)
    print(f"\n\nResults saved to: {output_csv}")
    print(f"Total frames recorded: {frame}")
    cv2.destroyAllWindows()

    return output_csv, frame
