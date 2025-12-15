#!/usr/bin/env python3
"""
Analyze Kvasir-VQA dataset to understand question-answer patterns.
This will help identify:
1. Questions that look binary but have categorical answers (MISMATCH!)
2. All possible answer values for each question
3. Which questions are truly binary vs multi-choice vs open-ended
4. What candidate lists we need to create
"""

import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import os


class KvasirAnalyzer:
    """Analyze Kvasir-VQA dataset to understand question-answer patterns."""
    
    def __init__(self):
        self.questions = defaultdict(list)  # question text -> list of answers
        self.question_patterns = defaultdict(set)  # question pattern -> unique answers
        self.binary_like_questions = []  # Questions that LOOK binary but aren't
        self.answer_vocabulary = defaultdict(Counter)  # question -> answer frequency
        
    def normalize_question(self, question: str) -> str:
        """Normalize question to find patterns."""
        # Convert to lowercase
        q = question.lower().strip()
        # Remove punctuation at end
        q = q.rstrip('?.,;!').strip()
        return q
    
    def looks_binary(self, question: str) -> bool:
        """Check if question looks like it should have yes/no answer."""
        q = question.lower()
        binary_patterns = [
            r'^is\s',
            r'^are\s',
            r'^does\s',
            r'^do\s',
            r'^has\s',
            r'^have\s',
            r'^can\s',
            r'^was\s',
            r'^were\s',
            r'^will\s',
            r'^should\s',
        ]
        return any(re.match(pattern, q) for pattern in binary_patterns)
    
    def is_truly_binary(self, answers: List[str]) -> bool:
        """Check if answers are actually yes/no."""
        unique_answers = set(a.lower().strip() for a in answers)
        return unique_answers.issubset({'yes', 'no'})
    
    def analyze_dataset(self, data_file: str):
        """Analyze a single dataset file."""
        print(f"\n{'='*80}")
        print(f"Analyzing: {data_file}")
        print(f"{'='*80}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        print(f"Total samples: {len(data)}")
        
        # Collect all question-answer pairs
        for item in data:
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            if question and answer:
                normalized_q = self.normalize_question(question)
                self.questions[normalized_q].append(answer)
                self.question_patterns[question].add(answer)
                self.answer_vocabulary[normalized_q][answer] += 1
        
        print(f"Unique questions: {len(self.questions)}")
    
    def find_mismatches(self):
        """Find questions that look binary but have non-binary answers."""
        print(f"\n{'='*80}")
        print("CRITICAL: INSTRUCTION-GT MISMATCHES")
        print(f"{'='*80}\n")
        
        mismatches = []
        
        for question, answers in self.questions.items():
            if self.looks_binary(question) and not self.is_truly_binary(answers):
                unique_answers = set(answers)
                mismatches.append({
                    'question': question,
                    'looks_like': 'binary (yes/no)',
                    'actual_answers': unique_answers,
                    'count': len(answers)
                })
        
        if mismatches:
            print(f"⚠️  Found {len(mismatches)} MISMATCHED questions!")
            print("These questions LOOK like yes/no but have categorical answers:\n")
            
            for i, mismatch in enumerate(mismatches[:10], 1):  # Show first 10
                print(f"{i}. Question: \"{mismatch['question']}\"")
                print(f"   Looks like: {mismatch['looks_like']}")
                print(f"   Actual answers: {sorted(mismatch['actual_answers'])}")
                print(f"   Occurrences: {mismatch['count']}")
                print()
            
            if len(mismatches) > 10:
                print(f"... and {len(mismatches) - 10} more mismatches\n")
        else:
            print("✓ No mismatches found!\n")
        
        return mismatches
    
    def analyze_answer_types(self):
        """Categorize questions by their answer patterns."""
        print(f"\n{'='*80}")
        print("ANSWER TYPE ANALYSIS")
        print(f"{'='*80}\n")
        
        categories = {
            'binary': [],
            'numeric': [],
            'categorical_small': [],  # 2-10 unique answers
            'categorical_medium': [],  # 11-30 unique answers
            'categorical_large': [],  # 30+ unique answers
            'open_ended': []  # Many unique answers, likely free text
        }
        
        for question, answers in self.questions.items():
            unique_answers = set(answers)
            num_unique = len(unique_answers)
            
            # Check if binary
            if unique_answers.issubset({'yes', 'no'}):
                categories['binary'].append(question)
            
            # Check if numeric
            elif all(self._is_numeric(a) for a in unique_answers):
                categories['numeric'].append(question)
            
            # Check if categorical
            elif num_unique <= 10:
                categories['categorical_small'].append({
                    'question': question,
                    'answers': sorted(unique_answers),
                    'count': num_unique
                })
            elif num_unique <= 30:
                categories['categorical_medium'].append({
                    'question': question,
                    'answers': sorted(unique_answers),
                    'count': num_unique
                })
            elif num_unique <= 100:
                categories['categorical_large'].append({
                    'question': question,
                    'answers': sorted(unique_answers),
                    'count': num_unique
                })
            else:
                categories['open_ended'].append({
                    'question': question,
                    'unique_answers': num_unique,
                    'sample_answers': sorted(unique_answers)[:10]
                })
        
        # Print summary
        print("Question Type Distribution:")
        print(f"  Binary (yes/no): {len(categories['binary'])}")
        print(f"  Numeric: {len(categories['numeric'])}")
        print(f"  Categorical (2-10 choices): {len(categories['categorical_small'])}")
        print(f"  Categorical (11-30 choices): {len(categories['categorical_medium'])}")
        print(f"  Categorical (30-100 choices): {len(categories['categorical_large'])}")
        print(f"  Open-ended (100+ unique): {len(categories['open_ended'])}")
        
        return categories
    
    def _is_numeric(self, answer: str) -> bool:
        """Check if answer is numeric."""
        answer = answer.strip()
        # Direct number
        if answer.isdigit():
            return True
        # Number with unit (e.g., "5mm", "2cm")
        if re.match(r'^\d+\s*(mm|cm|m)?$', answer):
            return True
        # Range (e.g., "5-10mm")
        if re.match(r'^\d+-\d+\s*(mm|cm|m)?$', answer):
            return True
        return False
    
    def generate_candidate_lists(self, categories: Dict) -> Dict[str, List[str]]:
        """Generate candidate lists for categorical questions."""
        print(f"\n{'='*80}")
        print("SUGGESTED CANDIDATE LISTS")
        print(f"{'='*80}\n")
        
        candidate_lists = {}
        
        # Process categorical_small (most important)
        print("Close-Ended Questions (Need Candidate Lists):\n")
        
        for i, item in enumerate(categories['categorical_small'][:20], 1):
            question = item['question']
            answers = item['answers']
            
            # Try to infer category name from question
            category_name = self._infer_category_name(question)
            
            print(f"{i}. Question: \"{question}\"")
            print(f"   Category: {category_name}")
            print(f"   Candidates ({len(answers)}): {answers}")
            print()
            
            candidate_lists[category_name] = answers
        
        return candidate_lists
    
    def _infer_category_name(self, question: str) -> str:
        """Infer category name from question text."""
        q = question.lower()
        
        # Common patterns
        if 'abnormalit' in q:
            return 'abnormality'
        elif 'instrument' in q or 'tool' in q:
            return 'instrument'
        elif 'procedure' in q or 'type' in q:
            return 'procedure'
        elif 'color' in q or 'colour' in q:
            return 'color'
        elif 'size' in q:
            return 'size'
        elif 'location' in q or 'position' in q:
            return 'location'
        elif 'quality' in q:
            return 'quality'
        elif 'polyp' in q:
            return 'polyp_related'
        elif 'detect' in q:
            return 'detection'
        else:
            # Use first significant word
            words = q.split()
            for word in words:
                if len(word) > 4 and word not in ['there', 'these', 'those', 'which', 'where']:
                    return word
            return 'unknown'
    
    def show_sample_qa_pairs(self, n: int = 20):
        """Show sample Q-A pairs for inspection."""
        print(f"\n{'='*80}")
        print(f"SAMPLE QUESTION-ANSWER PAIRS (First {n})")
        print(f"{'='*80}\n")
        
        for i, (question, answers) in enumerate(list(self.questions.items())[:n], 1):
            unique_answers = sorted(set(answers))
            print(f"{i}. Q: \"{question}\"")
            print(f"   A: {unique_answers[:10]}")  # Show first 10 unique answers
            if len(unique_answers) > 10:
                print(f"      ... and {len(unique_answers) - 10} more")
            print(f"   Total occurrences: {len(answers)}")
            print()
    
    def export_analysis(self, output_file: str):
        """Export analysis results to JSON."""
        results = {
            'total_unique_questions': len(self.questions),
            'mismatches': [],
            'question_types': {},
            'candidate_lists': {},
        }
        
        # Find mismatches
        for question, answers in self.questions.items():
            if self.looks_binary(question) and not self.is_truly_binary(answers):
                results['mismatches'].append({
                    'question': question,
                    'unique_answers': sorted(set(answers)),
                    'count': len(answers)
                })
        
        # Categorize questions
        for question, answers in self.questions.items():
            unique_answers = set(answers)
            num_unique = len(unique_answers)
            
            if unique_answers.issubset({'yes', 'no'}):
                qtype = 'binary'
            elif all(self._is_numeric(a) for a in unique_answers):
                qtype = 'numeric'
            elif num_unique <= 10:
                qtype = 'categorical_small'
            elif num_unique <= 30:
                qtype = 'categorical_medium'
            else:
                qtype = 'open_ended'
            
            if qtype not in results['question_types']:
                results['question_types'][qtype] = []
            
            results['question_types'][qtype].append({
                'question': question,
                'unique_answers': sorted(unique_answers),
                'count': len(answers)
            })
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Analysis exported to: {output_file}")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Kvasir-VQA dataset")
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing train.json, val.json, test.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='kvasir_analysis_report.json',
        help='Output file for analysis results'
    )
    
    args = parser.parse_args()
    
    analyzer = KvasirAnalyzer()
    
    # Analyze all splits
    for split in ['train.json', 'val.json', 'test.json']:
        data_file = os.path.join(args.data_dir, split)
        if os.path.exists(data_file):
            analyzer.analyze_dataset(data_file)
    
    print(f"\n{'='*80}")
    print("COMPLETE ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    # Show sample Q-A pairs
    analyzer.show_sample_qa_pairs(n=20)
    
    # Find critical mismatches
    mismatches = analyzer.find_mismatches()
    
    # Analyze answer types
    categories = analyzer.analyze_answer_types()
    
    # Generate candidate lists
    candidate_lists = analyzer.generate_candidate_lists(categories)
    
    # Export results
    analyzer.export_analysis(args.output)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total unique questions analyzed: {len(analyzer.questions)}")
    print(f"Critical mismatches found: {len(mismatches)}")
    print(f"Binary questions: {len(categories['binary'])}")
    print(f"Numeric questions: {len(categories['numeric'])}")
    print(f"Categorical questions: {len(categories['categorical_small']) + len(categories['categorical_medium'])}")
    print(f"Open-ended questions: {len(categories['open_ended'])}")
    print(f"\nDetailed results saved to: {args.output}")
    print(f"{'='*80}\n")
    
    # Critical warning if mismatches found
    if mismatches:
        print("⚠️  CRITICAL WARNING ⚠️")
        print(f"Found {len(mismatches)} questions with instruction-GT mismatches!")
        print("These MUST be fixed before training!")
        print("See detailed report above and in the output file.\n")


if __name__ == "__main__":
    main()
