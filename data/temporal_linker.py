#!/usr/bin/env python3
"""
Temporal Data Linker for Video Sequences
Creates frame-to-frame temporal links for EndoVis 2018 sequences.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalLinker:
    """Link frames in video sequences for temporal CoT reasoning."""
    
    def __init__(self, sequence_dir: Path):
        """
        Initialize temporal linker.
        
        Args:
            sequence_dir: Directory containing sequence frames
        """
        self.sequence_dir = Path(sequence_dir)
        self.sequences: Dict[str, List[Dict]] = {}
    
    def load_sequence_frames(
        self,
        sequence_id: str,
        frame_pattern: str = "*.png"
    ) -> List[Dict]:
        """
        Load all frames for a sequence.
        
        Args:
            sequence_id: Sequence identifier (e.g., "seq_1")
            frame_pattern: Pattern to match frame files
            
        Returns:
            List of frame metadata dictionaries
        """
        seq_dir = self.sequence_dir / sequence_id
        if not seq_dir.exists():
            logger.warning(f"Sequence directory not found: {seq_dir}")
            return []
        
        # Find all frame files
        frame_files = sorted(seq_dir.glob(frame_pattern))
        
        frames = []
        for idx, frame_file in enumerate(frame_files):
            frame_id = frame_file.stem
            frame_num = self._extract_frame_number(frame_id)
            
            frames.append({
                "sequence_id": sequence_id,
                "frame_id": frame_id,
                "frame_number": frame_num,
                "frame_index": idx,
                "image_path": str(frame_file),
                "previous_frame_id": frames[-1]["frame_id"] if frames else None,
                "next_frame_id": None  # Will be set in next iteration
            })
            
            # Set next_frame_id for previous frame
            if len(frames) > 1:
                frames[-2]["next_frame_id"] = frame_id
        
        logger.info(f"Loaded {len(frames)} frames for sequence {sequence_id}")
        return frames
    
    def _extract_frame_number(self, frame_id: str) -> int:
        """Extract frame number from frame ID."""
        # Try to extract number from frame ID (e.g., "frame_001" -> 1)
        import re
        match = re.search(r'(\d+)', frame_id)
        if match:
            return int(match.group(1))
        return 0
    
    def compute_motion_features(
        self,
        frame1_path: str,
        frame2_path: str
    ) -> Dict:
        """
        Compute motion features between two consecutive frames.
        
        Args:
            frame1_path: Path to first frame
            frame2_path: Path to second frame
            
        Returns:
            Dictionary with motion features and description
        """
        try:
            # Load images
            img1 = cv2.imread(frame1_path)
            img2 = cv2.imread(frame2_path)
            
            if img1 is None or img2 is None:
                return {"description": "Unable to load frames", "motion_vector": None}
            
            # Resize if needed for efficiency
            if img1.shape[0] > 512 or img1.shape[1] > 512:
                scale = 512 / max(img1.shape[0], img1.shape[1])
                new_size = (int(img1.shape[1] * scale), int(img1.shape[0] * scale))
                img1 = cv2.resize(img1, new_size)
                img2 = cv2.resize(img2, new_size)
            
            # Convert to grayscale for optical flow
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            # Compute motion magnitude and direction
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            mean_magnitude = np.mean(magnitude)
            max_magnitude = np.max(magnitude)
            
            # Compute dominant direction
            angle = np.arctan2(flow[..., 1], flow[..., 0])
            mean_angle = np.mean(angle)
            
            # Describe motion
            motion_desc = self._describe_motion(mean_magnitude, max_magnitude, mean_angle)
            
            # Detect significant change regions
            change_regions = self._detect_change_regions(img1, img2)
            
            return {
                "motion_vector": {
                    "mean_magnitude": float(mean_magnitude),
                    "max_magnitude": float(max_magnitude),
                    "mean_angle": float(mean_angle)
                },
                "description": motion_desc,
                "change_regions": change_regions,
                "has_significant_motion": mean_magnitude > 2.0
            }
        except Exception as e:
            logger.warning(f"Error computing motion: {e}")
            return {
                "description": "Motion computation failed",
                "motion_vector": None,
                "has_significant_motion": False
            }
    
    def _describe_motion(
        self,
        mean_magnitude: float,
        max_magnitude: float,
        mean_angle: float
    ) -> str:
        """Generate human-readable motion description."""
        if mean_magnitude < 1.0:
            return "Minimal camera movement, stable view"
        elif mean_magnitude < 3.0:
            direction = self._angle_to_direction(mean_angle)
            return f"Small camera movement ({direction}), slight adjustment"
        elif mean_magnitude < 10.0:
            direction = self._angle_to_direction(mean_angle)
            return f"Moderate camera movement ({direction}), repositioning"
        else:
            direction = self._angle_to_direction(mean_angle)
            return f"Significant camera movement ({direction}), major repositioning"
    
    def _angle_to_direction(self, angle: float) -> str:
        """Convert angle to direction description."""
        # Convert angle to degrees
        degrees = np.degrees(angle)
        degrees = (degrees + 360) % 360  # Normalize to 0-360
        
        if 337.5 <= degrees or degrees < 22.5:
            return "right"
        elif 22.5 <= degrees < 67.5:
            return "down-right"
        elif 67.5 <= degrees < 112.5:
            return "down"
        elif 112.5 <= degrees < 157.5:
            return "down-left"
        elif 157.5 <= degrees < 202.5:
            return "left"
        elif 202.5 <= degrees < 247.5:
            return "up-left"
        elif 247.5 <= degrees < 292.5:
            return "up"
        else:
            return "up-right"
    
    def _detect_change_regions(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> List[Dict]:
        """Detect regions with significant changes."""
        # Compute absolute difference
        diff = cv2.absdiff(img1, img2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find significant changes
        _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        change_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small changes
                x, y, w, h = cv2.boundingRect(contour)
                change_regions.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "area": int(area)
                })
        
        return change_regions
    
    def create_temporal_structure(
        self,
        sequence_id: str,
        qa_pairs: List[Dict],
        compute_motion: bool = True
    ) -> List[Dict]:
        """
        Create temporal structure for a sequence with QA pairs.
        
        Args:
            sequence_id: Sequence identifier
            qa_pairs: List of QA pairs for this sequence
            compute_motion: Whether to compute motion between frames
            
        Returns:
            List of frame data with temporal links and QA pairs
        """
        # Load frames
        frames = self.load_sequence_frames(sequence_id)
        
        if not frames:
            return []
        
        # Group QA pairs by frame
        qa_by_frame: Dict[str, List[Dict]] = {}
        for qa in qa_pairs:
            frame_id = qa.get('frame_id') or qa.get('image_id')
            if frame_id:
                if frame_id not in qa_by_frame:
                    qa_by_frame[frame_id] = []
                qa_by_frame[frame_id].append(qa)
        
        # Create temporal structure
        temporal_data = []
        prev_frame_data = None
        
        for frame in frames:
            frame_id = frame["frame_id"]
            frame_qa_pairs = qa_by_frame.get(frame_id, [])
            
            # Compute motion if previous frame exists
            motion_info = None
            if compute_motion and prev_frame_data:
                motion_info = self.compute_motion_features(
                    prev_frame_data["image_path"],
                    frame["image_path"]
                )
            
            frame_data = {
                "sequence_id": sequence_id,
                "frame_id": frame_id,
                "frame_number": frame["frame_number"],
                "frame_index": frame["frame_index"],
                "image_path": frame["image_path"],
                "previous_frame_id": frame["previous_frame_id"],
                "qa_pairs": frame_qa_pairs,
                "motion_info": motion_info,
                "previous_frame_predictions": None  # Will be populated during inference
            }
            
            temporal_data.append(frame_data)
            prev_frame_data = frame_data
        
        logger.info(
            f"Created temporal structure for {sequence_id}: "
            f"{len(temporal_data)} frames, {sum(len(f['qa_pairs']) for f in temporal_data)} QA pairs"
        )
        
        return temporal_data
    
    def save_temporal_structure(
        self,
        temporal_data: List[Dict],
        output_path: Path
    ):
        """Save temporal structure to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(temporal_data, f, indent=2)
        logger.info(f"Saved temporal structure to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create temporal structure for video sequences")
    parser.add_argument("--sequence-dir", required=True, help="Directory containing sequences")
    parser.add_argument("--qa-file", required=True, help="JSON file with QA pairs")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--sequence-id", required=True, help="Sequence ID to process")
    parser.add_argument("--no-motion", action="store_true", help="Skip motion computation")
    
    args = parser.parse_args()
    
    # Load QA pairs
    with open(args.qa_file, 'r') as f:
        qa_pairs = json.load(f)
    
    # Create temporal structure
    linker = TemporalLinker(args.sequence_dir)
    temporal_data = linker.create_temporal_structure(
        args.sequence_id,
        qa_pairs,
        compute_motion=not args.no_motion
    )
    
    # Save
    linker.save_temporal_structure(temporal_data, Path(args.output))














