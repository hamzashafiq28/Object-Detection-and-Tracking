import cv2

def draw_tracks(frame, tracks, trajectories, colors, show_trajectories, frame_count, tracker_type):
    """
    Draw tracks and trajectories on the frame
    
    Args:
        frame: Original frame
        tracks: List of tracks
        trajectories: Dictionary of track trajectories
        colors: Dictionary of track colors
        show_trajectories: Whether to show trajectories
        frame_count: Current frame count
        tracker_type: Type of tracker used
        
    Returns:
        Annotated frame
    """
    vis_frame = frame.copy()
    
    # Draw each track
    for track in tracks:
        # Get track info - format depends on tracker type
        if len(track) >= 5:  # Standard format: [x1, y1, x2, y2, id, ...]
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            
            # Get confidence if available
            conf = track[5] if len(track) > 5 else 1.0
        else:
            # Skip invalid tracks
            continue
        
        # Get color for this track
        color = colors.get(track_id, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Add label with track ID and confidence
        label = f"ID:{track_id} {conf:.2f}"
        cv2.putText(vis_frame, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw trajectory
        if show_trajectories and track_id in trajectories:
            points = trajectories[track_id]
            for i in range(1, len(points)):
                # Get previous and current points
                prev_pt = points[i-1]
                curr_pt = points[i]
                
                # Draw line
                cv2.line(vis_frame, prev_pt, curr_pt, color, 2)
    
    # Add stats
    cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis_frame, f"Active Tracks: {len(tracks)}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(vis_frame, f"Tracker: {tracker_type}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
    return vis_frame