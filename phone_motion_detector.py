"""
Phone Camera Motion Detector
=============================
Streams video from an iPhone (via an IP-camera app like IP Camera Lite,
EpocCam, or Iriun Webcam) and runs live motion detection on it.

Setup:
1. Install an IP-camera app on your iPhone and start streaming.
   It will display a URL like: http://192.168.1.42:8080/video
2. Make sure your phone and laptop are on the same Wi-Fi network.
3. pip install opencv-python
4. Run:  python phone_motion_detector.py --url http://192.168.1.42:8080/video

Controls:
- Press 'q' to quit.
- Press 's' to manually save a snapshot.
"""

import cv2
import argparse
import datetime
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Motion detector using phone camera stream")
    parser.add_argument("--url", required=True,
                         help="Stream URL from your phone's IP camera app, e.g. http://192.168.1.42:8080/video")
    parser.add_argument("--min-area", type=int, default=800,
                         help="Minimum contour area (in pixels) to count as motion. Lower = more sensitive.")
    parser.add_argument("--cooldown", type=float, default=3.0,
                         help="Seconds to wait between saved snapshots.")
    parser.add_argument("--save-dir", default="motion_snapshots",
                         help="Folder to save motion snapshots into.")
    parser.add_argument("--no-preview", action="store_true",
                         help="Run headless (no window), useful if running on a remote machine.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Connecting to stream: {args.url}")
    cap = cv2.VideoCapture(args.url)

    if not cap.isOpened():
        print("ERROR: Could not open the stream. Check that:")
        print("  - The IP camera app is running and streaming on your phone")
        print("  - Your laptop and phone are on the same Wi-Fi network")
        print("  - The URL is correct (open it in a browser to test)")
        return

    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=False)

    last_saved = 0
    frame_count = 0

    print("Streaming started. Press 'q' to quit, 's' to save a snapshot manually.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection to stream, retrying...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(args.url)
            continue

        frame_count += 1

        # Resize for speed/consistency
        display_frame = cv2.resize(frame, (960, 540))

        # Motion detection
        fg_mask = back_sub.apply(display_frame)
        fg_mask = cv2.medianBlur(fg_mask, 5)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for c in contours:
            if cv2.contourArea(c) < args.min_area:
                continue
            motion_detected = True
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Skip the first ~30 frames while the background model is still learning
        if frame_count < 30:
            motion_detected = False

        status_text = "MOTION DETECTED" if motion_detected else "No motion"
        status_color = (0, 0, 255) if motion_detected else (0, 200, 0)
        cv2.putText(display_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, status_color, 2)
        cv2.putText(display_frame, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (15, display_frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Auto-save snapshot on motion, respecting cooldown
        now = time.time()
        if motion_detected and (now - last_saved) > args.cooldown:
            filename = os.path.join(args.save_dir, f"motion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(filename, display_frame)
            print(f"Motion detected — saved {filename}")
            last_saved = now

        if not args.no_preview:
            cv2.imshow("Phone Camera - Motion Detector", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = os.path.join(args.save_dir, f"manual_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(filename, display_frame)
                print(f"Manual snapshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
