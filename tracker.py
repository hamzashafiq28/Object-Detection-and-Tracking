import cv2
import numpy as np
import torch
import time
import os
from pathlib import Path
from collections import deque
from ultralytics import YOLO

from trackers.tracker_factory import create_tracker
from utils.visualization import draw_tracks
from utils.detection_utils import filter_detections

class HumanTracker:
    """Human tracking system using BoxMOT with various tracker options"""
    
    def __init__(self, 
                 model_path="yolo12n.pt", 
                 tracker_type="deepocsort",
                 device=None,
                 conf_threshold=0.3,
                 show_trajectories=True,
                 trajectory_length=30,
                 imgsz=640):
        """
        Initialize the human tracking system
        
        Args:
            model_path: Path to YOLO model
            tracker_type: Type of tracker (deepocsort, boosttrack, strongsort, botsort, bytetrack)
            device: Device to use (cuda or cpu)
            conf_threshold: Confidence threshold for detections
            show_trajectories: Whether to show trajectories
            trajectory_length: Maximum length of trajectories
        """
        # Select device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        
        # Initialize tracker
        self.tracker_type = tracker_type.lower()
        self.tracker = create_tracker(self.tracker_type, self.device)
        
        # Visualization settings
        self.show_trajectories = show_trajectories
        self.trajectory_length = trajectory_length
        self.trajectories = {}  # {track_id: deque of points}
        self.colors = {}  # {track_id: color tuple}
        
        # Stats
        self.frame_count = 0
        self.total_detections = 0
        self.total_tracks = 0
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Annotated frame, tracking results
        """
        self.frame_count += 1
        
        # Run YOLO detection
        results = self.model.predict(
            frame, 
            conf=self.conf_threshold, 
            classes=[0],  # Class 0 is person
            verbose=False,
            imgsz=self.imgsz 
        )
        
        # Get detections
        detections = results[0].boxes
        
        # Extract boxes, confidences, and class IDs
        if len(detections) > 0 and detections.xyxy.numel() > 0:
            boxes = detections.xyxy.cpu().numpy()
            confs = detections.conf.cpu().numpy()
            class_ids = detections.cls.cpu().numpy()
                
            # Filter detections to get only persons and apply other filters
            boxes, confs, class_ids = filter_detections(boxes, confs, class_ids)
                
            # Update statistics
            self.total_detections += len(boxes)
        else:
            boxes = np.empty((0, 4))
            confs = np.empty(0)
            class_ids = np.empty(0)

        tracking_input = np.zeros((len(boxes), 6))
        if len(boxes) > 0:
            # Format for tracking: [x1, y1, x2, y2, conf, class_id]
            tracking_input[:, :4] = boxes
            tracking_input[:, 4] = confs
            tracking_input[:, 5] = class_ids
            
        # Run tracking
        try:
            if self.tracker_type in ["deepocsort", "strongsort", "boosttrack"]:
                # These trackers need the original frame for appearance features
                tracks = self.tracker.update(tracking_input, frame)
            else:
                # These trackers only need detection boxes
                tracks = self.tracker.update(tracking_input)
                
            # Update statistics
            self.total_tracks += len(tracks)
            
        except Exception as e:
            print(f"Error during tracking: {e}")
            tracks = np.empty((0, 5))
        
        # Update trajectories
        self._update_trajectories(tracks)
        
        # Visualize tracks
        annotated_frame = draw_tracks(
            frame, 
            tracks, 
            self.trajectories, 
            self.colors, 
            self.show_trajectories,
            self.frame_count,
            self.tracker_type
        )
        
        return annotated_frame, tracks
    
    def _update_trajectories(self, tracks):
        """Update trajectory history for each track"""
        for track in tracks:
            if len(track) >= 5:
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])
                
                # Generate consistent color if needed
                if track_id not in self.colors:
                    self.colors[track_id] = (
                        int((track_id * 43) % 255),
                        int((track_id * 97) % 255),
                        int((track_id * 29) % 255)
                    )
                
                # Initialize trajectory if needed
                if track_id not in self.trajectories:
                    self.trajectories[track_id] = deque(maxlen=self.trajectory_length)
                
                # Update trajectory with center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                self.trajectories[track_id].append((center_x, center_y))
            
    def process_video(self, video_path, output_path=None, show=True):
        """
        Process an entire video
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (None for no output)
            show: Whether to show video during processing
        """
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {frame_count} frames")
        
        # Initialize video writer if needed
        writer = None
        if output_path:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
            
        # Process frames
        processing_times = []
        start_time = time.time()
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_start = time.time()
                annotated_frame, tracks = self.process_frame(frame)
                frame_time = time.time() - frame_start
                processing_times.append(frame_time)
                
                # Write to output video
                if writer:
                    writer.write(annotated_frame)
                    
                # Show frame
                if show:
                    # Add processing time
                    cv2.putText(annotated_frame, f"Time: {frame_time*1000:.1f}ms", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Show frame
                    cv2.imshow("Human Tracking", annotated_frame)
                    
                    # Handle key press (ESC or q to quit)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):
                        break
                    
                # Print progress every 30 frames
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
                    avg_time = np.mean(processing_times[-30:]) * 1000 if processing_times else 0
                    print(f"Frame {self.frame_count}/{frame_count}, {avg_fps:.1f} FPS, {avg_time:.1f}ms per frame")
                    
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        
        finally:
            # Clean up
            if writer:
                writer.release()
            cap.release()
            cv2.destroyAllWindows()
            
            # Print summary
            total_time = time.time() - start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            avg_time = np.mean(processing_times) * 1000 if processing_times else 0
            print(f"\nProcessing finished:")
            print(f"Total frames: {self.frame_count}")
            print(f"Total time: {total_time:.2f} seconds")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Average processing time: {avg_time:.1f}ms per frame")
            print(f"Total detections: {self.total_detections}")
            print(f"Total tracks: {self.total_tracks}")