#!/usr/bin/env python3
"""
Data Loaders for Surgical VQA with Temporal Support
Supports both Kvasir-VQA (single frame) and EndoVis 2018 (video sequences).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SurgicalVQADataset(Dataset):
    """
    Dataset for surgical VQA with support for temporal context.
    Handles both single-frame (Kvasir) and multi-frame (EndoVis) scenarios.
    """
    
    def __init__(
        self,
        data_file: str,
        image_base_path: str,
        is_temporal: bool = False,
        sequence_data: Optional[Dict] = None,
        max_length: int = 512,
        image_size: int = 448
    ):
        """
        Initialize the dataset.
        
        Args:
            data_file: Path to JSON file with QA pairs
            image_base_path: Base path for images
            is_temporal: Whether this is a temporal dataset (video sequences)
            sequence_data: Optional temporal structure data
            max_length: Maximum sequence length
            image_size: Image size for resizing
        """
        self.image_base_path = Path(image_base_path)
        self.is_temporal = is_temporal
        self.max_length = max_length
        self.image_size = image_size
        
        # Load data - handle both JSON and JSONL files
        with open(data_file, 'r', encoding='utf-8') as f:
            if data_file.endswith('.jsonl'):
                # JSONL format: one JSON object per line
                self.data = []
                for line in f:
                    line = line.strip()
                    if line:
                        self.data.append(json.loads(line))
            else:
                # Regular JSON format
                self.data = json.load(f)
                # If it's a dict, try to extract list
                if isinstance(self.data, dict):
                    # Try common keys
                    if 'data' in self.data:
                        self.data = self.data['data']
                    elif 'qa_pairs' in self.data:
                        self.data = self.data['qa_pairs']
                    elif 'questions' in self.data:
                        self.data = self.data['questions']
        
        # Load temporal structure if available
        self.sequence_data = sequence_data or {}
        if is_temporal and sequence_data:
            self._build_temporal_index()
        
        logger.info(f"Loaded {len(self.data)} samples from {data_file}")
        if is_temporal:
            logger.info(f"Temporal mode: {len(self.sequence_data)} sequences")
    
    def _build_temporal_index(self):
        """Build index for fast temporal lookups."""
        self.temporal_index = {}
        for seq_id, frames in self.sequence_data.items():
            for frame in frames:
                frame_id = frame.get('frame_id')
                if frame_id:
                    self.temporal_index[frame_id] = frame
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        item = self.data[idx]
        
        # Load image
        image_path = self._get_image_path(item)
        image = self._load_image(image_path)
        
        # Get question and answer
        question = item.get('question', '')
        answer = item.get('answer', '')
        category = item.get('category', item.get('stage', 1))
        
        # Get temporal context if available
        temporal_context = None
        if self.is_temporal:
            temporal_context = self._get_temporal_context(item)
        
        return {
            'image': image,
            'image_path': str(image_path),
            'question': question,
            'answer': answer,
            'category': category,
            'stage': item.get('stage', 1),
            'temporal_context': temporal_context,
            'item_id': item.get('image_id') or item.get('frame_id', str(idx))
        }
    
    def _get_image_path(self, item: Dict) -> Path:
        """Get image path from item."""
        if 'image_path' in item:
            return Path(item['image_path'])
        elif 'image_filename' in item:
            return self.image_base_path / item['image_filename']
        elif 'image_id' in item:
            # Try common extensions
            for ext in ['.jpg', '.jpeg', '.png']:
                path = self.image_base_path / f"{item['image_id']}{ext}"
                if path.exists():
                    return path
            return self.image_base_path / f"{item['image_id']}.jpg"
        else:
            raise ValueError(f"Cannot determine image path for item: {item}")
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """Load and preprocess image."""
        try:
            image = Image.open(image_path).convert('RGB')
            # Resize if needed
            if max(image.size) > self.image_size:
                image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Return blank image as fallback
            return Image.new('RGB', (self.image_size, self.image_size), color='black')
    
    def _get_temporal_context(self, item: Dict) -> Optional[Dict]:
        """Get temporal context for an item."""
        # Check if temporal_index exists (only built if is_temporal and sequence_data)
        if not hasattr(self, 'temporal_index') or self.temporal_index is None:
            return None
        
        frame_id = item.get('frame_id') or item.get('image_id')
        if not frame_id or frame_id not in self.temporal_index:
            return None
        
        frame_data = self.temporal_index[frame_id]
        
        # Get previous frame info
        prev_frame_id = frame_data.get('previous_frame_id')
        prev_predictions = frame_data.get('previous_frame_predictions')
        motion_info = frame_data.get('motion_info')
        
        context = {}
        
        if prev_predictions:
            context['observations'] = prev_predictions
        
        if motion_info:
            if isinstance(motion_info, dict):
                context['motion_description'] = motion_info.get('description')
                context['motion_info'] = motion_info
            else:
                context['motion_description'] = str(motion_info)
        
        return context if context else None


class TemporalSurgicalVQADataset(Dataset):
    """
    Dataset specifically for temporal sequences (EndoVis 2018).
    Processes frames in sequence order with temporal dependencies.
    """
    
    def __init__(
        self,
        temporal_data_file: str,
        image_base_path: str,
        max_length: int = 512,
        image_size: int = 448
    ):
        """
        Initialize temporal dataset.
        
        Args:
            temporal_data_file: Path to JSON file with temporal structure
            image_base_path: Base path for images
            max_length: Maximum sequence length
            image_size: Image size for resizing
        """
        self.image_base_path = Path(image_base_path)
        self.max_length = max_length
        self.image_size = image_size
        
        # Load temporal structure
        with open(temporal_data_file, 'r') as f:
            self.temporal_data = json.load(f)
        
        # Flatten to list of (frame, qa_pair) tuples
        self.samples = []
        for frame_data in self.temporal_data:
            frame_id = frame_data['frame_id']
            for qa_pair in frame_data.get('qa_pairs', []):
                self.samples.append({
                    'frame_data': frame_data,
                    'qa_pair': qa_pair
                })
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.temporal_data)} frames")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with temporal context."""
        sample = self.samples[idx]
        frame_data = sample['frame_data']
        qa_pair = sample['qa_pair']
        
        # Load image
        image_path = Path(frame_data['image_path'])
        image = self._load_image(image_path)
        
        # Get temporal context
        temporal_context = {
            'previous_frame_id': frame_data.get('previous_frame_id'),
            'previous_frame_predictions': frame_data.get('previous_frame_predictions'),
            'motion_info': frame_data.get('motion_info')
        }
        
        return {
            'image': image,
            'image_path': str(image_path),
            'question': qa_pair.get('question', ''),
            'answer': qa_pair.get('answer', ''),
            'category': qa_pair.get('category', qa_pair.get('stage', 1)),
            'stage': qa_pair.get('stage', 1),
            'temporal_context': temporal_context,
            'frame_id': frame_data['frame_id'],
            'sequence_id': frame_data['sequence_id'],
            'frame_number': frame_data.get('frame_number', 0)
        }
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """Load and preprocess image."""
        try:
            image = Image.open(image_path).convert('RGB')
            if max(image.size) > self.image_size:
                image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return Image.new('RGB', (self.image_size, self.image_size), color='black')
    
    def get_sequence_frames(self, sequence_id: str) -> List[Dict]:
        """Get all frames for a sequence in order."""
        return [
            frame for frame in self.temporal_data
            if frame['sequence_id'] == sequence_id
        ]


def create_data_loader(
    data_file: str,
    image_base_path: str,
    batch_size: int = 4,
    shuffle: bool = True,
    is_temporal: bool = False,
    temporal_data_file: Optional[str] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for surgical VQA data.
    
    Args:
        data_file: Path to JSON file with QA pairs
        image_base_path: Base path for images
        batch_size: Batch size
        shuffle: Whether to shuffle
        is_temporal: Whether this is temporal data
        temporal_data_file: Path to temporal structure file
        **kwargs: Additional arguments for dataset
        
    Returns:
        DataLoader instance
    """
    # Load temporal data if needed
    sequence_data = None
    if is_temporal and temporal_data_file:
        with open(temporal_data_file, 'r') as f:
            temporal_data = json.load(f)
            # Convert to dict by sequence_id
            sequence_data = {}
            for frame in temporal_data:
                seq_id = frame.get('sequence_id')
                if seq_id:
                    if seq_id not in sequence_data:
                        sequence_data[seq_id] = []
                    sequence_data[seq_id].append(frame)
    
    # Extract num_workers and pin_memory from kwargs BEFORE passing to dataset
    # These are DataLoader parameters, not dataset parameters
    num_workers = kwargs.pop('num_workers', 4)  # Default to 4 workers for parallel data loading
    pin_memory = kwargs.pop('pin_memory', True)  # Default to True for faster CPU->GPU transfer
    
    # Create dataset (kwargs now only contains dataset-relevant parameters)
    if is_temporal and temporal_data_file:
        dataset = TemporalSurgicalVQADataset(
            temporal_data_file=temporal_data_file,
            image_base_path=image_base_path,
            **kwargs
        )
    else:
        dataset = SurgicalVQADataset(
            data_file=data_file,
            image_base_path=image_base_path,
            is_temporal=is_temporal,
            sequence_data=sequence_data,
            **kwargs
        )
    
    # Create collator (simple for now, can be enhanced)
    def collate_fn(batch):
        """Simple collation function."""
        return {
            'images': [item['image'] for item in batch],
            'questions': [item['question'] for item in batch],
            'answers': [item['answer'] for item in batch],
            'categories': [item['category'] for item in batch],
            'stages': [item['stage'] for item in batch],
            'temporal_contexts': [item.get('temporal_context') for item in batch],
            'item_ids': [item.get('item_id') for item in batch]
        }
    
    # Only use num_workers if batch_size > 1 to avoid overhead
    if batch_size == 1:
        num_workers = 0
        pin_memory = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)  # Keep workers alive between epochs
    )




