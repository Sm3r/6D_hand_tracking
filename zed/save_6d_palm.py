import cv2
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import mediapipe as mp
from HandTrackingModule.HandTracking import HandTracking
from HandTrackingModule.Zed import Zed
import pyzed.sl as sl
import pandas as pd


def main():

    if len(sys.argv) == 2:
        print("Camera Mode: SVO")
        filename = sys.argv[1]
    else:
        print("Camera Mode: Live Streaming")
        filename = None
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
        
    # bring detector
    detector = HandTracking(maxHands=2, detectionCon=0.1, trackCon=0.8, complexity=1)
    # bring zed
    cam = Zed(filename)
    
    # print camera information
    cam.print_information()
    if filename:
        final_frame = cam.zed.get_svo_number_of_frames()
    else:
        final_frame = float('inf')
    camera_params = cam.camera_params
    frame = 0
    lx, ly, lz, lyaw, lpitch, lroll = 0, 0, 0, 0, 0, 0
    rx, ry, rz, ryaw, rpitch, rroll = 0, 0, 0, 0, 0, 0
    first_print = True


    while frame <= final_frame:
        err = cam.zed.grab(cam.runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
            # increment frame
            frame += 1
            # extract images from ZED
            cam.get_image()
            img = cam.img
            depth_img = cam.depth_img

            # Point cloud data
            pcl = cam.point_cloud

            # find hands
            img = detector.findHands(img)
            # find handmark positions
            data_left,data_right = detector.findpostion(depth_img, pcl,camera_params)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            # find the orientation of the palm
            left_orientaton = detector.calculate_orientation(data_left)
            # find the centroid of the palm
            left_centroid = detector.calculate_centroid(data_left)

            # find the orientation of the palm
            right_orientaton = detector.calculate_orientation(data_right)
            # find the centroid of the palm
            right_centroid = detector.calculate_centroid(data_right)

            if len(data_left) != 0:
                lx = left_centroid[0]
                ly = left_centroid[1]
                lz = left_centroid[2]
                lyaw = left_orientaton[0]
                lpitch = left_orientaton[1]
                lroll = left_orientaton[2]

            if len(data_right) != 0:
                rx = right_centroid[0]
                ry = right_centroid[1]
                rz = right_centroid[2]
                ryaw = right_orientaton[0]
                rpitch = right_orientaton[1]
                rroll = right_orientaton[2]

            F.append(frame)
            LX.append(lx)
            LY.append(ly)
            LZ.append(lz)
            LYAW.append(lyaw)
            LPITCH.append(lpitch)
            LROLL.append(lroll)
            RX.append(rx)
            RY.append(ry)
            RZ.append(rz)
            RYAW.append(ryaw)
            RPITCH.append(rpitch)
            RROLL.append(rroll)

            # Print detection and frame count on two fixed lines and refresh in-place
            detection = getattr(detector, 'detection_str', '')
            frame_line = f"Frame count: {frame}" + (f" / {final_frame}" if filename else "")
            if not first_print:
                # Move cursor up two lines to overwrite previous detection and frame lines
                print("\x1b[2A", end='')
            print(detection.ljust(80), flush=True)
            print(frame_line.ljust(80), flush=True)
            first_print = False
        else:
            break
    
    # Save final results
    df = pd.DataFrame(name_dict)
    if filename:
        base = os.path.splitext(os.path.basename(filename))[0]
        csvname = os.path.join('results', base + '.csv')
    else:
        csvname = os.path.join('results', 'output.csv')
    df.to_csv(csvname, index=False)
    print(f"\n\nResults saved to: {csvname}")
    print(f"Total frames recorded: {frame}")
    cv2.destroyAllWindows()




if __name__ == "__main__":
    #  Initialize lists for Pandas DataFrame
    F, RX, RY, RZ, RYAW, RPITCH, RROLL = [], [], [], [], [], [], []
    LX, LY, LZ, LYAW, LPITCH, LROLL = [], [], [], [], [], []
    key = ' '
    name_dict = {
        'Frame': F,
        'left X': LX,
        'left Y': LY,
        'left Z': LZ,
        'left Yaw': LYAW,
        'left Pitch': LPITCH,
        'left Roll': LROLL,
        'right X': RX,
        'right Y': RY,
        'right Z': RZ,
        'right Yaw': RYAW,
        'right Pitch': RPITCH,
        'right Roll': RROLL,     

    }

    # run main
    main()