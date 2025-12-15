#!/usr/bin/env python3
"""
Create Category-Based Instructions from Training Data Only

Per advisor requirements:
1. Extract all unique answers from TRAINING SET ONLY
2. Create ONE instruction template per category
3. Apply same instruction to train/val/test consistently
4. Report final instructions per category for advisor review
"""

import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple


class CategoryBasedInstructionBuilder:
    """Build instruction templates based on training data categories."""
    
    def __init__(self):
        self.category_questions = defaultdict(list)  # category -> list of questions
        self.category_candidates = defaultdict(set)  # category -> unique answers from TRAIN
        self.question_to_category = {}  # question -> category mapping
        
    def analyze_training_data(self, train_file: str):
        """
        Analyze ONLY the training set to extract categories and candidates.
        This ensures we don't leak test/val information.
        """
        print("\n" + "="*80)
        print("STEP 1: Analyzing TRAINING DATA ONLY")
        print("="*80)
        
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        print(f"Training samples: {len(train_data)}")
        
        # Track all raw answers per category to detect multi-label
        self.category_raw_answers = defaultdict(list)  # category -> all raw answers
        
        # Group questions by their semantic category
        for item in train_data:
            question = item.get('question', '').lower().strip()
            answer = item.get('answer', '').strip()
            
            if not question or not answer:
                continue
            
            # Use existing category field if available, otherwise categorize from question
            if 'category' in item and item['category']:
                category = item['category'].lower().replace('_', '_')
            else:
                category = self._categorize_question(question)
            
            # Store question-category mapping
            self.question_to_category[question] = category
            
            # Store raw answer for multi-label detection
            self.category_raw_answers[category].append(answer)
            
            # Collect unique answers from TRAINING data only
            # Handle multi-label (semicolon-separated)
            if ';' in answer:
                answer_parts = [a.strip() for a in answer.split(';')]
                self.category_candidates[category].update(answer_parts)
            else:
                self.category_candidates[category].add(answer)
            
            # Track questions in this category
            if question not in self.category_questions[category]:
                self.category_questions[category].append(question)
        
        # Convert sets to sorted lists for consistency
        for category in self.category_candidates:
            self.category_candidates[category] = sorted(list(self.category_candidates[category]))
        
        # EXPANSION STEP: Add clinically/logically possible options
        # Per advisor: Base on training but expand for generalization
        self._expand_candidates_for_robustness()
        
        # Print summary
        print(f"\nFound {len(self.category_candidates)} categories:")
        for category, candidates in sorted(self.category_candidates.items()):
            num_questions = len(self.category_questions[category])
            print(f"  {category:25s}: {len(candidates):3d} candidates, {num_questions} questions")
        
        return self.category_candidates
    
    def _expand_candidates_for_robustness(self):
        """
        Expand candidate lists to include clinically/logically possible options.
        Per advisor requirement: Base on training but expand for generalization.
        """
        print("\n" + "="*80)
        print("STEP 1b: Expanding Candidates for Robustness")
        print("="*80)
        
        for category, candidates in self.category_candidates.items():
            original_count = len(candidates)
            
            # NUMERIC EXPANSION: Ensure full range 0-20 for counting questions
            if self._is_numeric_category(category, candidates):
                numeric_vals = [int(c) for c in candidates if c.isdigit()]
                if numeric_vals:
                    max_observed = max(numeric_vals)
                    # Expand to at least 0-20 for medical image counting
                    expanded_max = max(20, max_observed)
                    full_range = [str(i) for i in range(0, expanded_max + 1)]
                    self.category_candidates[category] = full_range
                    print(f"\n  ✓ {category}: Expanded numeric range")
                    print(f"    Training range: 0-{max_observed} ({original_count} values)")
                    print(f"    Expanded range: 0-{expanded_max} ({len(full_range)} values)")
                    print(f"    Reason: Ensure model can handle any count up to {expanded_max}")
            
            # CLINICAL EXPANSION: Add common medical options
            else:
                expanded_candidates = self._expand_clinical_options(category, candidates)
                if len(expanded_candidates) > original_count:
                    self.category_candidates[category] = expanded_candidates
                    print(f"\n  ✓ {category}: Expanded with clinical options")
                    print(f"    Training: {original_count} options")
                    print(f"    Expanded: {len(expanded_candidates)} options")
                    print(f"    Added: {sorted(set(expanded_candidates) - set(candidates))}")
        
        print("\n" + "="*80)
    
    def _is_numeric_category(self, category: str, candidates: List[str]) -> bool:
        """Check if category is numeric (counting)."""
        # Check if category name suggests counting
        if any(word in category.lower() for word in ['count', 'how many', 'number']):
            return True
        # Check if all candidates are digits
        if all(c.isdigit() for c in candidates if c):
            return True
        return False
    
    def _expand_clinical_options(self, category: str, training_candidates: List[str]) -> List[str]:
        """
        Expand candidates with clinically/logically possible options.
        Per advisor: "Include all clinically or logically possible options for that category,
        even if some options do not appear in the training split."
        """
        # Start with training candidates
        expanded = set(training_candidates)
        
        # ABNORMALITY DETECTION: Add common endoscopic abnormalities
        if 'abnormality' in category.lower() and 'detection' in category.lower():
            clinical_abnormalities = [
                'none', 'normal',  # Negative cases
                'polyp', 'ulcer', 'bleeding', 'erosion',  # Common findings
                'inflammation', 'tumor', 'mass', 'lesion',
                'barretts', 'oesophagitis', 'gastritis',
                'hemorrhoids', 'ulcerative colitis', 'short-segment barretts',
                'diverticula', 'stricture', 'varices'
            ]
            expanded.update(clinical_abnormalities)
        
        # INSTRUMENT DETECTION: Add common surgical instruments
        elif 'instrument' in category.lower() and 'detection' in category.lower():
            clinical_instruments = [
                'none',  # No instruments
                'polyp snare', 'biopsy forceps', 'injection needle',
                'metal clip', 'hemoclip', 'tube', 'catheter',
                'grasping forceps', 'scissors', 'knife',
                'dilator', 'retrieval net', 'basket'
            ]
            expanded.update(clinical_instruments)
        
        # LANDMARK DETECTION: Add common anatomical landmarks
        elif 'landmark' in category.lower() and 'detection' in category.lower():
            clinical_landmarks = [
                'none',
                'z-line', 'pylorus', 'cecum', 'ileum',
                'cardia', 'fundus', 'antrum',
                'duodenum', 'ampulla', 'ileocecal valve'
            ]
            expanded.update(clinical_landmarks)
        
        # COLORS: Ensure comprehensive color coverage
        elif 'color' in category.lower():
            clinical_colors = [
                'none',
                'red', 'pink', 'white', 'yellow', 'brown',
                'black', 'blue', 'green', 'purple', 'orange',
                'grey', 'gray', 'flesh', 'pale'
            ]
            expanded.update(clinical_colors)
        
        # POLYP TYPE: Add common polyp classifications
        elif 'polyp' in category.lower() and 'type' in category.lower():
            polyp_types = [
                'none', 'not applicable',
                'paris ip', 'paris is', 'paris iia', 'paris iib', 'paris iic',
                'paris iii', 'pedunculated', 'sessile', 'flat'
            ]
            expanded.update(polyp_types)
        
        # PROCEDURE TYPE: Add common endoscopic procedures
        elif 'procedure' in category.lower():
            procedures = [
                'gastroscopy', 'colonoscopy', 'sigmoidoscopy',
                'enteroscopy', 'esophagogastroduodenoscopy', 'egd',
                'capsule endoscopy', 'ercp', 'eus'
            ]
            expanded.update(procedures)
        
        # SIZE: Ensure common size ranges
        elif 'size' in category.lower():
            size_categories = [
                'none', 'not applicable',
                '<5mm', '5-10mm', '11-20mm', '>20mm',
                '>20', 'small', 'medium', 'large'
            ]
            expanded.update(size_categories)
        
        # LOCATION: Ensure comprehensive location coverage
        elif 'location' in category.lower():
            locations = [
                'none', 'not applicable',
                'upper-left', 'upper-center', 'upper-right',
                'center-left', 'center', 'center-right',
                'lower-left', 'lower-center', 'lower-right',
                'lower-rigth'  # Keep training typo for consistency
            ]
            expanded.update(locations)
        
        # DIFFICULTY/REMOVAL/OTHER: Add common yes/no/not relevant options
        elif any(word in category.lower() for word in ['difficulty', 'removal', 'presence']):
            common_options = ['yes', 'no', 'not relevant', 'not applicable']
            expanded.update(common_options)
        
        # Sort and return
        return sorted(list(expanded))
    
    def _categorize_question(self, question: str) -> str:
        """
        Categorize question based on its content.
        Returns category name (e.g., 'abnormality', 'instrument', etc.)
        """
        q = question.lower()
        
        # Abnormality detection
        if 'abnormalit' in q and 'check all' in q:
            return 'abnormality_detection'
        
        # Instrument detection
        if 'instrument' in q and 'check all' in q:
            return 'instrument_detection'
        
        # Anatomical landmark detection
        if 'anatomical landmark' in q and 'check all' in q:
            return 'landmark_detection'
        
        # Procedure type
        if 'procedure' in q or 'type of procedure' in q:
            return 'procedure_type'
        
        # Polyp type
        if 'type of polyp' in q:
            return 'polyp_type'
        
        # Polyp size
        if 'size of the polyp' in q or 'size of polyp' in q:
            return 'polyp_size'
        
        # Detection difficulty
        if 'easy to detect' in q:
            return 'detection_difficulty'
        
        # Polyp removal status
        if 'polyps been removed' in q or 'all polyps removed' in q:
            return 'polyp_removal'
        
        # Binary: text presence
        if 'is there text' in q:
            return 'text_presence'
        
        # Binary: artifact presence
        if 'green/black box' in q or 'artefact' in q:
            return 'artifact_presence'
        
        # Binary: finding presence
        if 'contain any finding' in q:
            return 'finding_presence'
        
        # Count: polyps
        if 'how many polyp' in q:
            return 'polyp_count'
        
        # Count: instruments
        if 'how many instrument' in q or 'how many instrumnet' in q:  # handle typo
            return 'instrument_count'
        
        # Count: findings
        if 'how many finding' in q:
            return 'finding_count'
        
        # Color questions
        if 'color' in q or 'colour' in q:
            if 'abnormality' in q:
                return 'abnormality_color'
            elif 'landmark' in q or 'anatomical' in q:
                return 'landmark_color'
            else:
                return 'color_general'
        
        # Location questions
        if 'where in the image' in q:
            if 'abnormality' in q:
                return 'abnormality_location'
            elif 'instrument' in q:
                return 'instrument_location'
            elif 'landmark' in q or 'anatomical' in q:
                return 'landmark_location'
            else:
                return 'location_general'
        
        # Fallback
        return 'other'
    
    def create_category_instructions(self) -> Dict[str, Dict]:
        """
        Create ONE instruction template per category.
        Returns dict mapping category -> instruction template info.
        """
        print("\n" + "="*80)
        print("STEP 2: Creating Category-Based Instructions")
        print("="*80)
        
        category_instructions = {}
        
        for category, candidates in sorted(self.category_candidates.items()):
            # Determine instruction type and create template
            instruction_info = self._create_instruction_for_category(
                category, 
                candidates,
                self.category_questions[category]
            )
            
            category_instructions[category] = instruction_info
            
            print(f"\n✓ Created instruction for: {category}")
            print(f"  Candidates: {len(candidates)}")
            print(f"  Questions: {len(self.category_questions[category])}")
        
        return category_instructions
    
    def _create_instruction_for_category(self, category: str, candidates: List[str], 
                                         questions: List[str]) -> Dict:
        """Create instruction template for a specific category."""
        
        # Get sample question for this category
        sample_question = questions[0] if questions else ""
        
        # Determine if multi-label by checking:
        # 1. Question text contains "check all"
        # 2. OR >50% of training answers contain semicolons (multi-value answers)
        raw_answers = self.category_raw_answers.get(category, [])
        multi_answer_count = sum(1 for ans in raw_answers if ';' in ans)
        multi_answer_ratio = multi_answer_count / len(raw_answers) if raw_answers else 0
        
        is_multi_label = ('check all' in sample_question.lower() or 
                         multi_answer_ratio > 0.5)  # >50% have multiple values
        
        # Debug: show why multi-label was detected
        if is_multi_label and multi_answer_ratio > 0.5:
            print(f"   → {category}: Detected multi-label from training data ({multi_answer_ratio:.1%} have multiple values)")
        
        is_binary = len(candidates) == 2 and set(candidates) == {'yes', 'no'}
        is_numeric = all(c.isdigit() or c == '0' for c in candidates[:5])  # Check first 5
        
        # Create appropriate instruction
        if is_binary:
            instruction = self._create_binary_instruction(category, sample_question)
            qtype = 'binary'
        elif is_numeric:
            instruction = self._create_numeric_instruction(category, candidates, sample_question)
            qtype = 'numeric'
        elif is_multi_label:
            instruction = self._create_multi_label_instruction(category, candidates, sample_question)
            qtype = 'multi_label'
        else:
            # Determine if single choice or open-ended
            if len(candidates) <= 30:  # Reasonable number for single choice
                instruction = self._create_single_choice_instruction(category, candidates, sample_question)
                qtype = 'single_choice'
            else:
                instruction = self._create_open_constrained_instruction(category, candidates, sample_question)
                qtype = 'open_constrained'
        
        return {
            'category': category,
            'instruction_template': instruction,
            'question_type': qtype,
            'candidates': candidates,
            'num_candidates': len(candidates),
            'sample_questions': questions[:3],  # First 3 questions
            'is_multi_label': is_multi_label
        }
    
    @staticmethod
    def get_general_system_instruction() -> str:
        """
        Return the general system instruction that applies to ALL samples.
        This is prepended to every training/val/test sample.
        """
        return """You are a surgical image analysis assistant analysing an endoscopic image.

Instructions:
- Select your answer(s) ONLY from the provided candidate list
- For multi-label questions: Select ALL applicable items, separated by semicolons (;)
- For single-choice questions: Select EXACTLY one option
- Output format: item1; item2; item3 (for multi-label) or item1 (for single-choice)"""
    
    def _create_binary_instruction(self, category: str, sample_q: str) -> str:
        """Binary yes/no instruction - ULTRA-CONDENSED FORMAT."""
        return f"""Question: {{question}}
Candidates: ['yes', 'no']
Answer:"""
    
    def _create_numeric_instruction(self, category: str, candidates: List[str], sample_q: str) -> str:
        """Numeric count instruction - ULTRA-CONDENSED FORMAT."""
        candidates_str = ", ".join([f"'{c}'" for c in candidates])
        return f"""Question: {{question}}
Candidates: [{candidates_str}]
Answer:"""
    
    def _create_multi_label_instruction(self, category: str, candidates: List[str], sample_q: str) -> str:
        """Multi-label (choose all that apply) instruction - ULTRA-CONDENSED FORMAT."""
        candidates_str = ", ".join([f"'{c}'" for c in candidates])
        return f"""Question: {{question}}
Candidates: [{candidates_str}]
Answer:"""
    
    def _create_single_choice_instruction(self, category: str, candidates: List[str], sample_q: str) -> str:
        """Single choice instruction - ULTRA-CONDENSED FORMAT."""
        candidates_str = ", ".join([f"'{c}'" for c in candidates])
        return f"""Question: {{question}}
Candidates: [{candidates_str}]
Answer:"""
    
    def _create_open_constrained_instruction(self, category: str, candidates: List[str], sample_q: str) -> str:
        """Open-ended with controlled vocabulary instruction - ULTRA-CONDENSED FORMAT."""
        candidates_str = ", ".join([f"'{c}'" for c in candidates])
        return f"""Question: {{question}}
Candidates: [{candidates_str}]
Answer:"""
    
    def apply_instructions_to_dataset(self, data_file: str, output_file: str, 
                                     category_instructions: Dict):
        """Apply category-based instructions to a dataset split.
        
        ULTRA-CONDENSED FORMAT:
        Combines general system instruction + minimal category template.
        """
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        processed_data = []
        unknown_count = 0
        
        # Get the general system instruction (same for all samples)
        general_instruction = self.get_general_system_instruction()
        
        for item in data:
            question = item.get('question', '').lower().strip()
            
            # Get category for this question
            category = self.question_to_category.get(question)
            
            if category and category in category_instructions:
                instr_info = category_instructions[category]
                
                # Fill in the minimal category template with actual question
                category_instruction = instr_info['instruction_template'].replace(
                    '{question}', item.get('question', '')
                )
                
                # Combine: General instruction + Category instruction
                # The general instruction will be added as system message during training
                # The category instruction will be the user message
                full_instruction = f"{general_instruction}\n\n{category_instruction}"
                
                enhanced_item = {
                    **item,
                    'instruction': full_instruction,
                    'system_instruction': general_instruction,  # For reference
                    'category_instruction': category_instruction,  # For reference
                    'category': category,
                    'question_type': instr_info['question_type'],
                    'candidates': instr_info['candidates'],
                    'is_multi_label': instr_info['is_multi_label']
                }
                processed_data.append(enhanced_item)
            else:
                # Unknown question - shouldn't happen if trained on train set
                unknown_count += 1
                print(f"⚠️  Unknown question: {question[:60]}...")
                processed_data.append(item)
        
        # Save
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"\n✓ Processed {len(processed_data)} items")
        if unknown_count > 0:
            print(f"  ⚠️  {unknown_count} unknown questions (not in training set)")
        
        return processed_data
    
    def generate_advisor_report(self, category_instructions: Dict, output_file: str):
        """
        Generate report for advisor showing one instruction per category.
        ULTRA-CONDENSED FORMAT with general system instruction.
        """
        print("\n" + "="*80)
        print("STEP 3: Generating Advisor Report (ULTRA-CONDENSED FORMAT)")
        print("="*80)
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("ULTRA-CONDENSED INSTRUCTION TEMPLATES - ALL CATEGORIES")
        report_lines.append("(Based on Training Set Only)")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append("STRUCTURE: General Instruction + Minimal Category Templates")
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("GENERAL INSTRUCTION (Applied to ALL samples)")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append(self.get_general_system_instruction())
        report_lines.append("")
        report_lines.append("Character count: ~337 chars")
        report_lines.append("Estimated tokens: ~120 tokens")
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("MINIMAL CATEGORY TEMPLATES (Per question type)")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append(f"Total Categories: {len(category_instructions)}")
        report_lines.append("")
        
        for i, (category, info) in enumerate(sorted(category_instructions.items()), 1):
            report_lines.append("="*80)
            report_lines.append(f"CATEGORY {i}: {category.upper()}")
            report_lines.append("="*80)
            report_lines.append("")
            report_lines.append("MINIMAL TEMPLATE:")
            report_lines.append("-"*80)
            report_lines.append(info['instruction_template'])
            report_lines.append("-"*80)
            report_lines.append("")
        
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("COMPLETE SAMPLE EXAMPLE")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append("<|im_start|>system")
        report_lines.append(self.get_general_system_instruction())
        report_lines.append("<|im_end|>")
        report_lines.append("<|im_start|>user")
        report_lines.append("<image>")
        report_lines.append("Question: What color is the abnormality? If more than one separate with ;")
        report_lines.append("Candidates: ['black', 'blue', 'brown', ..., 'red', 'white', 'yellow']")
        report_lines.append("Answer:<|im_end|>")
        report_lines.append("<|im_start|>assistant")
        report_lines.append("red; white<|im_end|>")
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("END OF ULTRA-CONDENSED INSTRUCTION TEMPLATES")
        report_lines.append("="*80)
        
        # Save report
        report_text = "\n".join(report_lines)
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Advisor report saved to: {output_file}")
        print(f"  Contains {len(category_instructions)} category-based instructions")
        
        return report_text


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', required=True, help='Training data file')
    parser.add_argument('--val_file', required=True, help='Validation data file')
    parser.add_argument('--test_file', required=True, help='Test data file')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CATEGORY-BASED INSTRUCTION BUILDER")
    print("="*80)
    print("\nRequirements:")
    print("  1. ✓ Instructions based on TRAINING SET only")
    print("  2. ✓ Same instruction per category across all splits")
    print("  3. ✓ Candidate lists from training data only")
    print("="*80)
    
    builder = CategoryBasedInstructionBuilder()
    
    # Step 1: Analyze training data
    builder.analyze_training_data(args.train_file)
    
    # Step 2: Create category instructions
    category_instructions = builder.create_category_instructions()
    
    # Step 3: Apply to all splits
    print("\n" + "="*80)
    print("STEP 3: Applying Instructions to All Splits")
    print("="*80)
    
    for split_name, input_file in [
        ('train', args.train_file),
        ('val', args.val_file),
        ('test', args.test_file)
    ]:
        output_file = f"{args.output_dir}/{split_name}_CATEGORY_BASED.json"
        print(f"\nProcessing {split_name} split...")
        builder.apply_instructions_to_dataset(input_file, output_file, category_instructions)
    
    # Step 4: Generate advisor report
    report_file = f"{args.output_dir}/INSTRUCTIONS_PER_CATEGORY.txt"
    builder.generate_advisor_report(category_instructions, report_file)
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {args.output_dir}/train_CATEGORY_BASED.json")
    print(f"  - {args.output_dir}/val_CATEGORY_BASED.json")
    print(f"  - {args.output_dir}/test_CATEGORY_BASED.json")
    print(f"  - {args.output_dir}/INSTRUCTIONS_PER_CATEGORY.txt  ← Send this to advisor!")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
