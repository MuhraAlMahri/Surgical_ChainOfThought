#!/usr/bin/env python3
"""
Training script for multi-head temporal CoT.

Training strategy:
1. Load fine-tuned checkpoint
2. Add multi-head wrapper (freeze base)
3. Train heads with sequential context passing:
   - Process Stage 1 questions → store outputs
   - Process Stage 2 questions with Stage 1 context
   - Process Stage 3 questions with Stage 1+2 context
4. For EndoVis: Add temporal context between frames
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from multihead_model import create_multihead_model
from cot_prompts import build_cot_prompt, format_prompt_for_model
from data.vqa_data_loader import create_data_loader
from data.temporal_linker import TemporalLinker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_question_categories(categories_file: str) -> Dict[str, Dict[str, str]]:
    """Load question categories mapping."""
    with open(categories_file, 'r') as f:
        return json.load(f)


def train_kvasir_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    categories: Dict[str, str],
    processor,
    device: str,
    criterion: nn.Module,
    grad_accum: int = 16,
    use_bf16: bool = True,
    batch_size: int = 1
):
    """Training for single-frame dataset (Kvasir)"""
    
    model.train()
    total_loss = 0.0
    total_samples = 0
    accumulation_step = 0
    
    # Set dtype for mixed precision
    if use_bf16 and torch.cuda.is_bf16_supported():
        scaler = torch.cuda.amp.GradScaler(enabled=False)  # bf16 doesn't need scaler
        dtype = torch.bfloat16
    else:
        scaler = None
        dtype = torch.float32
    
    logger.info(f"Starting epoch with {len(dataloader)} batches (grad_accum={grad_accum}, batch_size={batch_size}, dtype={dtype})")
    
    import time
    batch_times = []
    
    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()
        if batch_idx % 100 == 0:
            if batch_times:
                avg_time = sum(batch_times[-100:]) / len(batch_times[-100:])
                logger.info(f"Processing batch {batch_idx}/{len(dataloader)} (avg: {avg_time:.2f}s/batch)")
            else:
                logger.info(f"Processing batch {batch_idx}/{len(dataloader)}")
        # Group questions by stage
        stage1_qa = []
        stage2_qa = []
        stage3_qa = []
        
        images = batch.get('images', [])
        questions = batch.get('questions', [])
        answers = batch.get('answers', [])
        question_categories = batch.get('categories', [])
        
        for i, (q, cat) in enumerate(zip(questions, question_categories)):
            category = categories.get(q, cat)
            if category == 'abnormality_detection' or category == 1:
                stage1_qa.append({'question': q, 'answer': answers[i], 'image': images[i] if i < len(images) else None})
            elif category == 'characteristics' or category == 2:
                stage2_qa.append({'question': q, 'answer': answers[i], 'image': images[i] if i < len(images) else None})
            elif category == 'treatment' or category == 3:
                stage3_qa.append({'question': q, 'answer': answers[i], 'image': images[i] if i < len(images) else None})
        
        # Stage 1: Process abnormality detection questions
        if len(stage1_qa) == 0:
            logger.debug(f"Batch {batch_idx}: No Stage 1 questions, skipping")
            continue
            
        stage1_context = None
        stage1_count = 0
        for qa in stage1_qa:
            if qa['image'] is None:
                continue
            
            prompt = build_cot_prompt(qa['question'], 'abnormality_detection', model_type="qwen3vl")
            
            # For qwen3vl, prompt is a messages list - add image to messages
            if isinstance(prompt, list):
                # Add image to the user message content
                for msg in prompt:
                    if msg.get('role') == 'user' and isinstance(msg.get('content'), str):
                        msg['content'] = [
                            {"type": "image", "image": qa['image']},
                            {"type": "text", "text": msg['content']}
                        ]
                text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            else:
                text = format_prompt_for_model(prompt, model_type="qwen3vl", processor=processor)
            
            inputs = processor(text=[text], images=[qa['image']], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Extract image_grid_thw if present (required for Qwen3-VL)
            model_kwargs = {
                'pixel_values': inputs['pixel_values'],
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs.get('attention_mask'),
                'category': 'abnormality_detection'
            }
            if 'image_grid_thw' in inputs:
                model_kwargs['image_grid_thw'] = inputs['image_grid_thw']
            
            output = model(**model_kwargs)
            
            # Compute loss: tokenize answer and compute cross-entropy
            logits = output['logits']  # Shape: [batch_size, vocab_size]
            
            # Tokenize answer
            answer_text = qa['answer']
            if isinstance(answer_text, list):
                answer_text = '; '.join(answer_text)  # Handle multi-label answers
            
            answer_tokens = processor.tokenizer(
                answer_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=64  # Limit answer length
            )
            answer_ids = answer_tokens['input_ids'].to(device)  # Shape: [1, answer_len]
            
            # Get first token as target (for next-token prediction)
            if answer_ids.shape[1] > 0:
                target_token = answer_ids[0, 0].long()  # First token of answer (ensure long dtype)
                # Ensure logits is [1, vocab_size] and target is [1]
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)  # [vocab_size] -> [1, vocab_size]
                target = target_token.unsqueeze(0)  # scalar -> [1]
                # Compute loss: cross-entropy between logits and target token
                loss = criterion(logits, target)
            else:
                # Fallback: use a dummy loss if answer is empty
                loss = torch.tensor(0.0, requires_grad=True).to(device)
            
            # Scale loss by accumulation steps
            loss = loss / grad_accum
            loss.backward()
            accumulation_step += 1
            stage1_count += 1
            
            # Store context for Stage 2 (detach to avoid graph reuse)
            if stage1_context is None:
                stage1_context = output['hidden_state'].detach()
            else:
                stage1_context = (stage1_context + output['hidden_state'].detach()) / 2
        
        # Stage 2: Process characteristics WITH Stage 1 context
        if len(stage2_qa) == 0:
            logger.debug(f"Batch {batch_idx}: No Stage 2 questions, skipping")
            stage2_context = None
        else:
            stage2_context = None
            stage2_count = 0
            for qa in stage2_qa:
                if qa['image'] is None:
                    continue
            
            prompt = build_cot_prompt(qa['question'], 'characteristics', model_type="qwen3vl")
            
            # For qwen3vl, prompt is a messages list - add image to messages
            if isinstance(prompt, list):
                # Add image to the user message content
                for msg in prompt:
                    if msg.get('role') == 'user' and isinstance(msg.get('content'), str):
                        msg['content'] = [
                            {"type": "image", "image": qa['image']},
                            {"type": "text", "text": msg['content']}
                        ]
                text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            else:
                text = format_prompt_for_model(prompt, model_type="qwen3vl", processor=processor)
            
            inputs = processor(text=[text], images=[qa['image']], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            model_kwargs = {
                'pixel_values': inputs['pixel_values'],
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs.get('attention_mask'),
                'category': 'characteristics',
                'previous_context': stage1_context
            }
            if 'image_grid_thw' in inputs:
                model_kwargs['image_grid_thw'] = inputs['image_grid_thw']
            
            output = model(**model_kwargs)
            
            logits = output['logits']
            
            # Compute loss: tokenize answer
            answer_text = qa['answer']
            if isinstance(answer_text, list):
                answer_text = '; '.join(answer_text)
            
            answer_tokens = processor.tokenizer(
                answer_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=64
            )
            answer_ids = answer_tokens['input_ids'].to(device)
            
            if answer_ids.shape[1] > 0:
                target_token = answer_ids[0, 0].long()  # Ensure long dtype
                # Ensure logits is [1, vocab_size] and target is [1]
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)  # [vocab_size] -> [1, vocab_size]
                target = target_token.unsqueeze(0)  # scalar -> [1]
                loss = criterion(logits, target)
            else:
                loss = torch.tensor(0.0, requires_grad=True).to(device)
            
            # Scale loss by accumulation steps
            loss = loss / grad_accum
            loss.backward()
            accumulation_step += 1
            
            if stage2_context is None:
                stage2_context = output['hidden_state'].detach()
            else:
                stage2_context = (stage2_context + output['hidden_state'].detach()) / 2
        
        # Stage 3: Process treatment WITH Stage 1+2 context (already detached)
        combined_context = None
        if stage1_context is not None and stage2_context is not None:
            combined_context = (stage1_context + stage2_context) / 2
        elif stage1_context is not None:
            combined_context = stage1_context
        elif stage2_context is not None:
            combined_context = stage2_context
        
        if len(stage3_qa) == 0:
            logger.debug(f"Batch {batch_idx}: No Stage 3 questions, skipping")
        else:
            stage3_count = 0
            for qa in stage3_qa:
                if qa['image'] is None:
                    continue
                
                prompt = build_cot_prompt(qa['question'], 'treatment', model_type="qwen3vl")
                
                # For qwen3vl, prompt is a messages list - add image to messages
                if isinstance(prompt, list):
                    # Add image to the user message content
                    for msg in prompt:
                        if msg.get('role') == 'user' and isinstance(msg.get('content'), str):
                            msg['content'] = [
                                {"type": "image", "image": qa['image']},
                                {"type": "text", "text": msg['content']}
                            ]
                    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                else:
                    text = format_prompt_for_model(prompt, model_type="qwen3vl", processor=processor)
                
                inputs = processor(text=[text], images=[qa['image']], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                model_kwargs = {
                    'pixel_values': inputs['pixel_values'],
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs.get('attention_mask'),
                    'category': 'treatment',
                    'previous_context': combined_context
                }
                if 'image_grid_thw' in inputs:
                    model_kwargs['image_grid_thw'] = inputs['image_grid_thw']
                
                output = model(**model_kwargs)
                
                logits = output['logits']
                
                # Compute loss: tokenize answer
                answer_text = qa['answer']
                if isinstance(answer_text, list):
                    answer_text = '; '.join(answer_text)
                
                answer_tokens = processor.tokenizer(
                    answer_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=64
                )
                answer_ids = answer_tokens['input_ids'].to(device)
                
                if answer_ids.shape[1] > 0:
                    target_token = answer_ids[0, 0].long()  # Ensure long dtype
                    # Ensure logits is [1, vocab_size] and target is [1]
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)  # [vocab_size] -> [1, vocab_size]
                    target = target_token.unsqueeze(0)  # scalar -> [1]
                    loss = criterion(logits, target)
                else:
                    loss = torch.tensor(0.0, requires_grad=True).to(device)
                
                # Scale loss by accumulation steps
                loss = loss / grad_accum
                loss.backward()
                accumulation_step += 1
                stage3_count += 1
        
        # Update optimizer after accumulation
        if accumulation_step >= grad_accum:
            optimizer.step()
            optimizer.zero_grad()
            accumulation_step = 0
        
        # Track loss (accumulate unscaled for logging)
        if accumulation_step == 0:  # After optimizer step
            total_loss += loss.item() * grad_accum if isinstance(loss, torch.Tensor) else 0.0
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / max((batch_idx + 1) // grad_accum, 1)
            avg_batch_time = sum(batch_times[-100:]) / min(len(batch_times), 100)
            samples_per_sec = batch_size / avg_batch_time if avg_batch_time > 0 else 0.0
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)} Loss: {avg_loss:.4f} (accum_step: {accumulation_step}/{grad_accum}, {avg_batch_time:.2f}s/batch, {samples_per_sec:.2f} samples/s)")
    
    # Final optimizer step if there are remaining accumulated gradients
    if accumulation_step > 0:
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / max(len(dataloader) // grad_accum, 1)
    logger.info(f"Epoch loss: {avg_loss:.4f}")


def compute_motion_description(prev_frame, curr_frame) -> str:
    """Simple motion description - can be enhanced with optical flow"""
    # Placeholder - actual implementation would use optical flow
    return "Camera moved slightly with minimal rotation"


def train_endovis_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    categories: Dict[str, str],
    processor,
    device: str,
    criterion: nn.Module,
    temporal_linker: Optional[TemporalLinker] = None,
    grad_accum: int = 16,
    use_bf16: bool = True
):
    """Training for video sequences (EndoVis) with temporal context"""
    
    model.train()
    total_loss = 0.0
    accumulation_step = 0
    
    # Set dtype for mixed precision
    if use_bf16 and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    logger.info(f"Starting EndoVis epoch with grad_accum={grad_accum}, dtype={dtype}")
    
    # Group by sequence
    sequences = {}
    for batch in dataloader:
        frame_ids = batch.get('frame_ids', [])
        sequence_ids = batch.get('sequence_ids', [])
        
        for i, (seq_id, frame_id) in enumerate(zip(sequence_ids, frame_ids)):
            if seq_id not in sequences:
                sequences[seq_id] = []
            
            sequences[seq_id].append({
                'frame_id': frame_id,
                'image': batch['images'][i] if i < len(batch['images']) else None,
                'question': batch['questions'][i] if i < len(batch['questions']) else None,
                'answer': batch['answers'][i] if i < len(batch['answers']) else None
            })
    
    # Process each sequence
    for seq_id, frames in sequences.items():
        prev_frame_context = None
        
        # Sort frames by frame_id
        frames = sorted(frames, key=lambda x: x['frame_id'])
        
        for frame_idx, frame in enumerate(frames):
            # Compute motion between frames
            if frame_idx > 0 and temporal_linker:
                prev_frame = frames[frame_idx - 1]
                motion_desc = compute_motion_description(
                    prev_frame.get('image'),
                    frame.get('image')
                )
            else:
                motion_desc = None
            
            # Build temporal context info
            temporal_info = None
            if prev_frame_context is not None:
                temporal_info = {
                    'summary': "Observations from previous frame",  # Could be more sophisticated
                    'motion': motion_desc
                }
            
            # Group questions by stage
            question = frame.get('question', '')
            category = categories.get(question, 'abnormality_detection')
            
            stage1_qa = []
            stage2_qa = []
            stage3_qa = []
            
            if category == 'abnormality_detection' or category == 1:
                stage1_qa.append(frame)
            elif category == 'characteristics' or category == 2:
                stage2_qa.append(frame)
            elif category == 'treatment' or category == 3:
                stage3_qa.append(frame)
            
            # Process Stage 1 with temporal context
            stage1_outputs = []
            for qa in stage1_qa:
                if qa.get('image') is None:
                    continue
                
                prompt = build_cot_prompt(
                    qa['question'],
                    'abnormality_detection',
                    previous_frame_info=temporal_info,
                    model_type="qwen3vl"
                )
                
                # For qwen3vl, prompt is a messages list - add image to messages
                if isinstance(prompt, list):
                    # Add image to the user message content
                    for msg in prompt:
                        if msg.get('role') == 'user' and isinstance(msg.get('content'), str):
                            msg['content'] = [
                                {"type": "image", "image": qa['image']},
                                {"type": "text", "text": msg['content']}
                            ]
                    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                else:
                    text = format_prompt_for_model(prompt, model_type="qwen3vl", processor=processor)
                
                inputs = processor(text=[text], images=[qa['image']], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                model_kwargs = {
                    'pixel_values': inputs['pixel_values'],
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs.get('attention_mask'),
                    'category': 'abnormality_detection',
                    'previous_context': prev_frame_context
                }
                if 'image_grid_thw' in inputs:
                    model_kwargs['image_grid_thw'] = inputs['image_grid_thw']
                
                output = model(**model_kwargs)
                logits = output['logits']
                
                # Compute loss: tokenize answer
                answer_text = qa.get('answer', '')
                if isinstance(answer_text, list):
                    answer_text = '; '.join(answer_text)
                
                answer_tokens = processor.tokenizer(
                    answer_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=64
                )
                answer_ids = answer_tokens['input_ids'].to(device)
                
                if answer_ids.shape[1] > 0:
                    target_token = answer_ids[0, 0].long()  # Ensure long dtype
                    # Ensure logits is [1, vocab_size] and target is [1]
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)  # [vocab_size] -> [1, vocab_size]
                    target = target_token.unsqueeze(0)  # scalar -> [1]
                    loss = criterion(logits, target)
                else:
                    loss = torch.tensor(0.0, requires_grad=True).to(device)
                
                loss = loss / grad_accum
                loss.backward()
                accumulation_step += 1
                stage1_outputs.append(output['hidden_state'].detach())
            
            # Process Stage 2 with Stage 1 + temporal context
            stage1_context = torch.stack(stage1_outputs).mean(dim=0) if stage1_outputs else None
            
            stage2_outputs = []
            for qa in stage2_qa:
                if qa.get('image') is None:
                    continue
                
                prompt = build_cot_prompt(
                    qa['question'],
                    'characteristics',
                    previous_frame_info=temporal_info,
                    model_type="qwen3vl"
                )
                
                # For qwen3vl, prompt is a messages list - add image to messages
                if isinstance(prompt, list):
                    # Add image to the user message content
                    for msg in prompt:
                        if msg.get('role') == 'user' and isinstance(msg.get('content'), str):
                            msg['content'] = [
                                {"type": "image", "image": qa['image']},
                                {"type": "text", "text": msg['content']}
                            ]
                    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                else:
                    text = format_prompt_for_model(prompt, model_type="qwen3vl", processor=processor)
                
                inputs = processor(text=[text], images=[qa['image']], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Combine temporal context + stage 1 context
                combined_context = None
                if prev_frame_context is not None and stage1_context is not None:
                    combined_context = (prev_frame_context + stage1_context) / 2
                elif prev_frame_context is not None:
                    combined_context = prev_frame_context
                elif stage1_context is not None:
                    combined_context = stage1_context
                
                model_kwargs = {
                    'pixel_values': inputs['pixel_values'],
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs.get('attention_mask'),
                    'category': 'characteristics',
                    'previous_context': combined_context
                }
                if 'image_grid_thw' in inputs:
                    model_kwargs['image_grid_thw'] = inputs['image_grid_thw']
                
                output = model(**model_kwargs)
                logits = output['logits']
                
                # Compute loss: tokenize answer
                answer_text = qa.get('answer', '')
                if isinstance(answer_text, list):
                    answer_text = '; '.join(answer_text)
                
                answer_tokens = processor.tokenizer(
                    answer_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=64
                )
                answer_ids = answer_tokens['input_ids'].to(device)
                
                if answer_ids.shape[1] > 0:
                    target_token = answer_ids[0, 0].long()  # Ensure long dtype
                    # Ensure logits is [1, vocab_size] and target is [1]
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)  # [vocab_size] -> [1, vocab_size]
                    target = target_token.unsqueeze(0)  # scalar -> [1]
                    loss = criterion(logits, target)
                else:
                    loss = torch.tensor(0.0, requires_grad=True).to(device)
                
                loss = loss / grad_accum
                loss.backward()
                accumulation_step += 1
                stage2_outputs.append(output['hidden_state'].detach())
            
            # Similar for Stage 3...
            stage3_outputs = []
            combined_stage_context = None
            if stage1_context is not None and stage2_outputs:
                stage2_context = torch.stack(stage2_outputs).mean(dim=0) if stage2_outputs else None
                if stage2_context is not None:
                    combined_stage_context = (stage1_context + stage2_context) / 2
                else:
                    combined_stage_context = stage1_context
            
            for qa in stage3_qa:
                if qa.get('image') is None:
                    continue
                
                prompt = build_cot_prompt(
                    qa['question'],
                    'treatment',
                    previous_frame_info=temporal_info,
                    model_type="qwen3vl"
                )
                
                # For qwen3vl, prompt is a messages list - add image to messages
                if isinstance(prompt, list):
                    # Add image to the user message content
                    for msg in prompt:
                        if msg.get('role') == 'user' and isinstance(msg.get('content'), str):
                            msg['content'] = [
                                {"type": "image", "image": qa['image']},
                                {"type": "text", "text": msg['content']}
                            ]
                    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                else:
                    text = format_prompt_for_model(prompt, model_type="qwen3vl", processor=processor)
                
                inputs = processor(text=[text], images=[qa['image']], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                final_context = combined_stage_context
                if prev_frame_context is not None and final_context is not None:
                    final_context = (prev_frame_context + final_context) / 2
                elif prev_frame_context is not None:
                    final_context = prev_frame_context
                
                model_kwargs = {
                    'pixel_values': inputs['pixel_values'],
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs.get('attention_mask'),
                    'category': 'treatment',
                    'previous_context': final_context
                }
                if 'image_grid_thw' in inputs:
                    model_kwargs['image_grid_thw'] = inputs['image_grid_thw']
                
                output = model(**model_kwargs)
                logits = output['logits']
                
                # Compute loss: tokenize answer
                answer_text = qa.get('answer', '')
                if isinstance(answer_text, list):
                    answer_text = '; '.join(answer_text)
                
                answer_tokens = processor.tokenizer(
                    answer_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=64
                )
                answer_ids = answer_tokens['input_ids'].to(device)
                
                if answer_ids.shape[1] > 0:
                    target_token = answer_ids[0, 0].long()  # Ensure long dtype
                    # Ensure logits is [1, vocab_size] and target is [1]
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)  # [vocab_size] -> [1, vocab_size]
                    target = target_token.unsqueeze(0)  # scalar -> [1]
                    loss = criterion(logits, target)
                else:
                    loss = torch.tensor(0.0, requires_grad=True).to(device)
                
                loss = loss / grad_accum
                loss.backward()
                accumulation_step += 1
                stage3_outputs.append(output['hidden_state'].detach())
            
            # Store aggregated context for next frame
            all_outputs = stage1_outputs + stage2_outputs + stage3_outputs
            if all_outputs:
                prev_frame_context = torch.stack(all_outputs).mean(dim=0)
            
            # Update optimizer after accumulation
            if accumulation_step >= grad_accum:
                optimizer.step()
                optimizer.zero_grad()
                accumulation_step = 0
                total_loss += loss.item() * grad_accum if isinstance(loss, torch.Tensor) else 0.0
    
    # Final optimizer step if there are remaining accumulated gradients
    if accumulation_step > 0:
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / max(len(sequences) // grad_accum, 1)
    logger.info(f"EndoVis epoch loss: {avg_loss:.4f}")
    
    avg_loss = total_loss / max(len(sequences), 1)
    logger.info(f"Epoch loss: {avg_loss:.4f}")


def train_multihead_cot(args):
    """Main training function."""
    
    # Load question categories
    categories = {}
    if args.question_categories and Path(args.question_categories).exists():
        all_categories = load_question_categories(args.question_categories)
        categories = all_categories.get(args.dataset, {})
    else:
        logger.warning(f"Question categories file not found: {args.question_categories}, continuing without categories")
    
    # Load model with correct LoRA settings
    model = create_multihead_model(
        base_checkpoint=args.base_checkpoint,
        model_type=args.model_type,
        freeze_base=True,  # Only train heads
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        # The model structure: MultiHeadCoT_Model -> model (actual model) -> base_model
        if hasattr(model, 'model') and hasattr(model.model, 'base_model'):
            if hasattr(model.model.base_model, 'gradient_checkpointing_enable'):
                model.model.base_model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
        elif hasattr(model, 'base_model'):
            if hasattr(model.base_model, 'gradient_checkpointing_enable'):
                model.base_model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
    
    # Get HuggingFace token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Get processor
    if args.model_type == "qwen3vl":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True, token=hf_token)
    elif args.model_type == "medgemma":
        processor = AutoProcessor.from_pretrained("google/medgemma-4b-it", trust_remote_code=True, token=hf_token)
    else:  # llava_med
        processor = AutoProcessor.from_pretrained("microsoft/llava-med-v1.5-mistral-7b", trust_remote_code=True, token=hf_token)
    
    # Optimizer (only trainable parameters) with weight decay
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Criterion
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    logger.info(f"Image base path: {args.image_base_path}")
    # Get num_workers and pin_memory from args if available, otherwise use defaults
    num_workers = getattr(args, 'dataloader_num_workers', 4)
    pin_memory = getattr(args, 'dataloader_pin_memory', True)
    train_loader = create_data_loader(
        data_file=args.data_path,
        image_base_path=args.image_base_path,
        batch_size=args.batch_size,
        shuffle=True,
        is_temporal=(args.dataset == "endovis"),
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    logger.info(f"Dataset loaded: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    
    # Temporal linker for EndoVis
    temporal_linker = None
    if args.dataset == "endovis":
        temporal_linker = TemporalLinker(args.image_base_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if resuming from checkpoint
    start_epoch = 0
    if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint:
        checkpoint_path = Path(args.resume_from_checkpoint)
        if checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint_epoch = checkpoint.get('epoch', 0)
            logger.info(f"Loaded checkpoint from epoch {checkpoint_epoch}")
            
            # Load model state (before moving to device/DataParallel)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            # Handle DataParallel wrapper if present
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            logger.info("✓ Model weights loaded from checkpoint")
            
            # Start from next epoch
            start_epoch = checkpoint_epoch  # checkpoint_epoch=1 means we start from epoch 2
            logger.info(f"Resuming training from epoch {start_epoch + 1}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}, starting from scratch")
    
    # Move model to device after loading checkpoint
    # Multi-GPU support: Use DataParallel if multiple GPUs available
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        logger.info(f"Using {num_gpus} GPUs with DataParallel")
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        # Adjust gradient accumulation: with 2 GPUs, effective batch size doubles
        # So we can halve grad_accum to maintain same effective batch size
        effective_grad_accum = max(1, args.grad_accum // num_gpus)
        logger.info(f"Adjusted gradient accumulation: {args.grad_accum} → {effective_grad_accum} (with {num_gpus} GPUs)")
        args.grad_accum = effective_grad_accum
    else:
        model = model.to(device)
    
    # Load optimizer state after model is on device
    if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint:
        checkpoint_path = Path(args.resume_from_checkpoint)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("✓ Optimizer state loaded from checkpoint")
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs (from epoch {start_epoch + 1})")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"=" * 60)
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"=" * 60)
        
        if args.dataset == "kvasir":
            # Single frame dataset
            train_kvasir_epoch(
                model, train_loader, optimizer, categories,
                processor, device, criterion,
                grad_accum=args.grad_accum,
                use_bf16=args.bf16,
                batch_size=args.batch_size
            )
        else:  # endovis
            # Video sequences with temporal context
            train_endovis_epoch(
                model, train_loader, optimizer, categories,
                processor, device, criterion, temporal_linker,
                grad_accum=args.grad_accum,
                use_bf16=args.bf16
            )
        
        # Save checkpoint
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            checkpoint_path = output_path / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["qwen3vl", "medgemma", "llava_med"], required=True)
    parser.add_argument("--dataset", choices=["kvasir", "endovis"], required=True)
    parser.add_argument("--base_checkpoint", required=True, help="Path to fine-tuned checkpoint")
    parser.add_argument("--question_categories", default="question_categories.json")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_base_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_len", type=int, default=3072)
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing (slows training but saves memory)")
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of DataLoader workers (0 to disable)")
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=True, help="Use pinned memory for faster CPU->GPU transfer")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from (e.g., checkpoint_epoch_1.pt)")
    
    args = parser.parse_args()
    train_multihead_cot(args)



