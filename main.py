import argparse
from pathlib import Path
import torch
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
    
    # Add mobile optimization arguments
    parser.add_argument("--optimize-mobile", action="store_true", help="Optimize and export model for mobile devices")
    parser.add_argument("--export-format", type=str, default="onnx", 
                        choices=["onnx", "tflite", "coreml", "torchscript"],
                        help="Export format for mobile deployment")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantization")
    parser.add_argument("--optimize-only", action="store_true", 
                        help="Only optimize and export model without running tracking")
    parser.add_argument("--export-path", type=str, default="mobile_models", 
                        help="Directory to save exported model")
    
    return parser.parse_args()

def optimize_for_mobile(model_path, format="onnx", quantize=False, export_path="mobile_models"):
    """
    Optimize and export model for mobile devices
    
    Args:
        model_path: Path to YOLO model
        format: Export format (onnx, tflite, coreml, torchscript)
        quantize: Whether to use quantization
        export_path: Directory to save exported model
    
    Returns:
        Path to exported model
    """
    from ultralytics import YOLO
    import os
    import glob
    from pathlib import Path
    import shutil
    
    # Create export directory
    export_path = Path(export_path)
    export_path.mkdir(exist_ok=True, parents=True)
    
    # Construct file name based on format and model name
    model_stem = Path(model_path).stem
    if format == "onnx":
        output_file = export_path / f"{model_stem}_mobile.onnx"
    elif format == "tflite":
        output_file = export_path / f"{model_stem}_mobile.tflite"
    elif format == "coreml":
        output_file = export_path / f"{model_stem}_mobile.mlmodel"
    elif format == "torchscript":
        output_file = export_path / f"{model_stem}_mobile.torchscript"
    else:
        output_file = export_path / f"{model_stem}_mobile.{format}"
    
    # Delete the output file if it already exists
    if output_file.exists():
        os.remove(output_file)
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Export model using updated syntax
    print(f"Exporting to {format.upper()} format...")
    
    # Export with proper arguments
    model.export(
        format=format,
        imgsz=320,  # smaller for mobile
        half=not quantize,  # Use FP16 if not using INT8
        int8=quantize,
        simplify=True,
        opset=12 if format == "onnx" else None,
        verbose=True
    )
    
    # The exported file will be in the current directory with the same name as the model
    # but with the appropriate extension
    exported_file = None
    
    # Look for the exported file in the current directory
    if format == "onnx":
        file_pattern = f"{model_stem}.onnx"
    elif format == "tflite":
        file_pattern = f"{model_stem}.tflite"
    elif format == "coreml":
        file_pattern = f"{model_stem}.mlmodel"
    elif format == "torchscript":
        file_pattern = f"{model_stem}.torchscript"
    else:
        file_pattern = f"{model_stem}.{format}"
    
    # Find the exported file
    exported_files = list(Path(".").glob(file_pattern))
    
    if not exported_files:
        # Try to find it in the runs/export directory as a fallback
        export_dir = Path("runs") / "export"
        exported_files = list(export_dir.glob(file_pattern))
        
    if not exported_files:
        # Still not found, try a more general search
        print("Searching for exported file...")
        if format == "onnx":
            exported_files = list(Path(".").glob("*.onnx"))
        elif format == "tflite":
            exported_files = list(Path(".").glob("*.tflite"))
        elif format == "coreml":
            exported_files = list(Path(".").glob("*.mlmodel"))
        elif format == "torchscript":
            exported_files = list(Path(".").glob("*.torchscript"))
        
        if not exported_files and Path("runs/export").exists():
            # Try even more broadly in the runs/export directory
            export_dir = Path("runs") / "export"
            if format == "onnx":
                exported_files = list(export_dir.glob("*.onnx"))
            elif format == "tflite":
                exported_files = list(export_dir.glob("*.tflite"))
            elif format == "coreml":
                exported_files = list(export_dir.glob("*.mlmodel"))
            elif format == "torchscript":
                exported_files = list(export_dir.glob("*.torchscript"))
    
    if exported_files:
        exported_file = exported_files[0]
        print(f"Found exported file: {exported_file}")
        
        # Copy the file to the target location
        shutil.copy2(exported_file, output_file)
        print(f"Model successfully exported to {output_file}")
        print(f"Optimizations applied: {'INT8 quantization' if quantize else 'FP16 precision'}")
        return output_file
    else:
        print(f"WARNING: Exported file not found. The model was exported but could not be located automatically.")
        print(f"Check the working directory and the 'runs/export' directory for the exported model.")
        print(f"Expected output format: {format}")
        
        # Return a best guess at where the file might be
        best_guess = Path(f"{model_stem}.{format}")
        print(f"Best guess for exported file location: {best_guess}")
        return best_guess

if __name__ == "__main__":
    args = parse_args()
    
    # Default image size for regular models
    model_imgsz = 640
    
    # Handle model optimization and export if requested
    if args.optimize_mobile:
        # Mobile models use a smaller image size
        model_imgsz = 320
        
        optimized_model_path = optimize_for_mobile(
            model_path=args.model,
            format=args.export_format,
            quantize=args.int8,
            export_path=args.export_path
        )
        
        print(f"Mobile-optimized model saved to: {optimized_model_path}")
        
        # If only optimizing, exit here
        if args.optimize_only:
            print("Optimization completed. Exiting as --optimize-only was specified.")
            exit(0)
        
        # Otherwise use the optimized model
        args.model = str(optimized_model_path)
        print(f"Using optimized model for tracking: {args.model}")
    
    # Create tracker with proper image size
    tracker = HumanTracker(
        model_path=args.model,
        tracker_type=args.tracker,
        device=args.device,
        conf_threshold=args.conf,
        show_trajectories=not args.no_trajectory,
        imgsz=model_imgsz  # Pass the appropriate image size
    )
    
    # Process video
    tracker.process_video(
        video_path=args.video,
        output_path=args.output,
        show=not args.no_show
    )