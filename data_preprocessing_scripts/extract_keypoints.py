import os
import json
import cv2
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# Suppress OpenCV warnings in headless environments
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings


def setup_directories(video_data):
    """
    Create necessary directories for frames, annotations, and keypoints.
    
    Args:
        video_data (str): The video identifier used for directory naming
        
    Returns:
        tuple: Paths to frames, annotated frames, and keypoints directories
    """
    frames_dir = os.path.join("data/frames", video_data)
    an_frames_dir = os.path.join("data/frames_annotated", video_data)
    keypoints_dir = os.path.join("data/keypoints", video_data)
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(an_frames_dir, exist_ok=True)
    os.makedirs(keypoints_dir, exist_ok=True)
    
    return frames_dir, an_frames_dir, keypoints_dir


def extract_keypoints(pose_landmarks):
    """
    Extract keypoints from MediaPipe pose landmarks.
    
    Args:
        pose_landmarks: MediaPipe pose landmarks object
        
    Returns:
        list: List of keypoint dictionaries with x, y, z coordinates and visibility
    """
    keypoints = []
    if pose_landmarks:
        for lm in pose_landmarks.landmark:
            keypoints.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })
    return keypoints


def create_annotated_image(image_rgb, pose_landmarks):
    """
    Create an annotated image with pose landmarks drawn.
    
    Args:
        image_rgb (numpy.ndarray): RGB image array
        pose_landmarks: MediaPipe pose landmarks object
        
    Returns:
        numpy.ndarray: Annotated image in BGR format
    """
    annotated_image = image_rgb.copy()
    if pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, 
            pose_landmarks, 
            list(mp_pose.POSE_CONNECTIONS)
        )
    
    # Convert back to BGR for OpenCV saving
    return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)


def save_frame_data(frame, keypoints, frame_id, frames_dir, an_frames_dir, keypoints_dir, annotated_bgr):
    """
    Save frame, keypoints, and annotated image to respective directories.
    
    Args:
        frame (numpy.ndarray): Original frame
        keypoints (list): List of keypoint dictionaries
        frame_id (int): Frame identifier
        frames_dir (str): Directory for original frames
        an_frames_dir (str): Directory for annotated frames
        keypoints_dir (str): Directory for keypoints JSON files
        annotated_bgr (numpy.ndarray): Annotated frame in BGR format
    """
    # Save original frame
    frame_filename = os.path.join(frames_dir, f"frame_{frame_id:04d}.png")
    cv2.imwrite(frame_filename, frame)
    
    # Save annotated frame
    annotated_path = os.path.join(an_frames_dir, f"frame_{frame_id:04d}_annotated.png")
    cv2.imwrite(annotated_path, annotated_bgr)
    
    # Save keypoints as JSON
    keypoints_filename = os.path.join(keypoints_dir, f"frame_{frame_id:04d}.json")
    with open(keypoints_filename, 'w') as f:
        json.dump(keypoints, f, indent=2)


def process_video(video_data="video8", frame_skip=10):
    """
    Process a video file to extract pose keypoints and create annotated frames.
    
    Args:
        video_data (str): Video identifier for file naming
        frame_skip (int): Process every nth frame (default: 10)
    """
    # Setup paths and directories
    video_path = os.path.join("data/videos", f"{video_data}.mp4")
    frames_dir, an_frames_dir, keypoints_dir = setup_directories(video_data)
    
    # Initialize MediaPipe pose with CPU-only mode to avoid graphics warnings
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_id = 0
    processed_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames based on frame_skip parameter
            if frame_id % frame_skip != 0:
                frame_id += 1
                continue
            
            # Convert frame to RGB for MediaPipe processing
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose estimation
            results = pose.process(image_rgb)
            
            # Extract keypoints
            pose_landmarks = getattr(results, 'pose_landmarks', None)
            keypoints = extract_keypoints(pose_landmarks)
            
            # Create annotated image
            annotated_bgr = create_annotated_image(image_rgb, pose_landmarks)
            
            # Save all data
            save_frame_data(frame, keypoints, frame_id, frames_dir, an_frames_dir, keypoints_dir, annotated_bgr)
            
            frame_id += 1
            processed_frames += 1
            
    finally:
        cap.release()
        pose.close()
    
    print(f"Processing complete. Processed {processed_frames} frames from {video_data}")
    print(f"Output directories:")
    print(f"  - Frames: {frames_dir}")
    print(f"  - Annotated: {an_frames_dir}")
    print(f"  - Keypoints: {keypoints_dir}")


if __name__ == "__main__":
    # Configuration
    VIDEO_DATA = "video8"  # Adjust as needed
    FRAME_SKIP = 10        # Process every 10th frame
    
    try:
        process_video(VIDEO_DATA, FRAME_SKIP)
    except Exception as e:
        print(f"Error processing video: {e}")
        exit(1)