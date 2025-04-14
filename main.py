import argparse
from pathlib import Path
from tracker import HumanTracker

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Human Tracking with BoxMOT")
    parser.add_argument("--video", type=str, default="input_video_examples/1.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="input_video_examples/1_out.mp4", help="Path to output video")
    parser.add_argument("--tracker", type=str, default="deepocsort", 
                        choices=["deepocsort", "boosttrack", "strongsort", "botsort", "bytetrack"],
                        help="Tracker type")
    parser.add_argument("--model", type=str, default="yolo12n.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use (cuda or cpu)")
    parser.add_argument("--no-show", action="store_true", help="Don't show video during processing")
    parser.add_argument("--no-trajectory", action="store_true", help="Don't show trajectories")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create tracker
    tracker = HumanTracker(
        model_path=args.model,
        tracker_type=args.tracker,
        device=args.device,
        conf_threshold=args.conf,
        show_trajectories=not args.no_trajectory
    )
    
    # Process video
    tracker.process_video(
        video_path=args.video,
        output_path=args.output,
        show=not args.no_show
    )