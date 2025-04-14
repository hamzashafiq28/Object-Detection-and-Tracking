import numpy as np
import cv2

def filter_detections(boxes, confs, class_ids):
    """
    Filter detections to get only persons and remove small or invalid detections
    
    Args:
        boxes: Bounding boxes [x1, y1, x2, y2]
        confs: Confidence scores
        class_ids: Class IDs
        
    Returns:
        Filtered boxes, confidences, and class IDs
    """
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0)
        
    # Ensure we're only tracking persons (class 0)
    person_indices = np.where(class_ids == 0)[0]
    if len(person_indices) > 0:
        boxes = boxes[person_indices]
        confs = confs[person_indices]
        class_ids = class_ids[person_indices]
    else:
        return np.empty((0, 4)), np.empty(0), np.empty(0)
        
    # Filter by minimum size and aspect ratio
    valid_indices = []
    for i, box in enumerate(boxes):
        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height
        
        # Minimum size criteria
        if width > 20 and height > 50 and area > 1000:
            valid_indices.append(i)
            
    if not valid_indices:
        return np.empty((0, 4)), np.empty(0), np.empty(0)
        
    filtered_boxes = boxes[valid_indices]
    filtered_confs = confs[valid_indices]
    filtered_class_ids = class_ids[valid_indices]
    
    return filtered_boxes, filtered_confs, filtered_class_ids