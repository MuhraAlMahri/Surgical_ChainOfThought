import json, os, re

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip().replace(".", "").replace(",", "").replace(";", "")

def extract_answer(pred: str, question: str) -> str:
    p = pred.lower().strip(" .,")
    q = question.lower()

    # YES/NO
    if any(w in q for w in ["is", "are", "was", "does", "do", "have", "has", "can", "could"]):
        if "yes" in p: return "yes"
        if "no" in p: return "no"

    # COUNTING
    if "how many" in q or "number of" in q:
        nums = re.findall(r'\b\d+\b', p)
        if nums: return nums[0]
        # Fallback: words like "one", "two"
        word_nums = {"one":"1", "two":"2", "three":"3", "four":"4", "five":"5"}
        for word, digit in word_nums.items():
            if word in p: return digit

    # ANATOMICAL TERMS (common in surgical VQA)
    surgical_terms = ["polyp", "ulcer", "bleeding", "instrument", "colon", "stomach", "esophagus", "text", "artifact", "red", "green", "blue", "yellow", "white", "black", "pink", "orange", "brown", "purple", "gray"]
    for term in surgical_terms:
        if term in p: return term

    # FALLBACK: first alphanumeric word
    words = re.findall(r'\b\w+\b', p)
    return words[0] if words else p

# Process all experiments
results_dir = "./results"
experiments = ["exp1", "exp2", "exp4"] # Single-stage experiments
multi_stage_experiments = ["exp3_corrected", "exp4_corrected", "exp5"] # Multi-stage experiments

# Sample counts for weighted accuracy
stage_sample_counts = {
    "Stage 1": 3275,
    "Stage 2": 5703,
    "Stage 3": 6
}

print("================================================================================")
print("SURGICAL VQA - CONCISE ANSWER EXTRACTION & ACCURACY COMPUTATION")
print("================================================================================")
print(f"\nResults directory: {results_dir}")

print("\n================================================================================")
print("SINGLE-STAGE EXPERIMENTS")
print("================================================================================")

for exp in experiments:
    file = os.path.join(results_dir, f"{exp}_evaluation_results.json")
    if not os.path.exists(file):
        print(f"  Warning: {exp} not found at {file}")
        continue

    with open(file, 'r') as f:
        data = json.load(f)

    if 'predictions' not in data:
        print(f"  Warning: '{exp}' results file does not contain 'predictions' key. Skipping.")
        continue

    correct_concise = 0
    for item in data['predictions']:
        gt = normalize_text(item["ground_truth"])
        pred_concise = extract_answer(item["prediction"], item["question"])
        item["prediction_concise"] = pred_concise
        if pred_concise == gt:
            correct_concise += 1

    acc = (correct_concise / len(data['predictions'])) * 100 if data['predictions'] else 0
    print(f"  ✓ {exp}: {acc:.2f}% ({correct_concise}/{len(data['predictions'])})")

    # Save enhanced results
    data['accuracy_concise'] = acc
    data['correct_concise'] = correct_concise
    output_file = os.path.join(results_dir, f"{exp}_predictions_concise.json")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"    Saved to: {os.path.basename(output_file)}")

print("\n================================================================================")
print("MULTI-STAGE EXPERIMENTS")
print("================================================================================")

for exp in multi_stage_experiments:
    overall_correct = 0
    overall_total = 0
    all_predictions = []
    stage_accuracies = {}

    # Check if this is a single-file multi-stage result (like exp3_corrected, exp4_corrected)
    single_file = os.path.join(results_dir, f"{exp}_evaluation_results.json")
    
    if os.path.exists(single_file):
        # Single file contains all stages
        with open(single_file, 'r') as f:
            data = json.load(f)
        
        # Check if it uses 'all_predictions' or 'predictions'
        predictions_key = 'all_predictions' if 'all_predictions' in data else 'predictions'
        
        if 'by_stage' in data and predictions_key in data:
            # Process predictions grouped by stage
            for item in data[predictions_key]:
                gt = normalize_text(item["ground_truth"])
                pred_concise = extract_answer(item["prediction"], item["question"])
                item["prediction_concise"] = pred_concise
                if pred_concise == gt:
                    item["correct_concise"] = True
                else:
                    item["correct_concise"] = False
            
            # Recalculate stage accuracies with concise answers
            stage_correct = {"Stage 1": 0, "Stage 2": 0, "Stage 3": 0}
            stage_total = {"Stage 1": 0, "Stage 2": 0, "Stage 3": 0}
            
            for item in data[predictions_key]:
                stage = item.get('stage')
                # Convert stage number to "Stage X" format
                if isinstance(stage, int):
                    stage_name = f"Stage {stage}"
                else:
                    stage_name = stage
                    
                if stage_name in stage_correct:
                    stage_total[stage_name] += 1
                    if item.get("correct_concise", False):
                        stage_correct[stage_name] += 1
            
            for stage_name in ["Stage 1", "Stage 2", "Stage 3"]:
                if stage_total[stage_name] > 0:
                    stage_acc = (stage_correct[stage_name] / stage_total[stage_name]) * 100
                    stage_accuracies[stage_name] = {
                        "accuracy": stage_acc,
                        "correct": stage_correct[stage_name],
                        "total": stage_total[stage_name]
                    }
                    overall_correct += stage_correct[stage_name]
                    overall_total += stage_total[stage_name]
            
            all_predictions = data[predictions_key]
    else:
        # Multiple files (exp5_stage1, exp5_stage2, exp5_stage3)
        for stage_num in range(1, 4):
            stage_file = os.path.join(results_dir, f"{exp}_stage{stage_num}_evaluation_results.json")
            if not os.path.exists(stage_file):
                print(f"  Warning: No predictions found for {exp} Stage {stage_num}")
                continue

            with open(stage_file, 'r') as f:
                stage_data = json.load(f)

            if 'predictions' not in stage_data:
                print(f"  Warning: '{exp}' Stage {stage_num} results file does not contain 'predictions' key. Skipping.")
                continue

            stage_correct = 0
            for item in stage_data['predictions']:
                gt = normalize_text(item["ground_truth"])
                # Handle both "prediction" and "final_prediction" keys
                pred_text = item.get("final_prediction", item.get("prediction", ""))
                pred_concise = extract_answer(pred_text, item["question"])
                item["prediction_concise"] = pred_concise
                if pred_concise == gt:
                    stage_correct += 1
            
            stage_total = len(stage_data['predictions'])
            stage_acc = (stage_correct / stage_total) * 100 if stage_total else 0
            stage_accuracies[f"Stage {stage_num}"] = {"accuracy": stage_acc, "correct": stage_correct, "total": stage_total}

            overall_correct += stage_correct
            overall_total += stage_total
            all_predictions.extend(stage_data['predictions'])

    if overall_total > 0:
        overall_acc = (overall_correct / overall_total) * 100
        print(f"\n  ✓ {exp} Overall: {overall_acc:.2f}% ({overall_correct}/{overall_total})")

        # Calculate weighted accuracy
        weighted_acc_sum = 0
        total_weighted_samples = 0
        for stage_name, stage_info in stage_accuracies.items():
            print(f"    {stage_name}: {stage_info['accuracy']:.2f}% ({stage_info['correct']}/{stage_info['total']})")
            if stage_name in stage_sample_counts:
                weight = stage_sample_counts[stage_name]
                weighted_acc_sum += (stage_info['accuracy'] / 100) * weight
                total_weighted_samples += weight
        
        weighted_overall_acc = (weighted_acc_sum / total_weighted_samples) * 100 if total_weighted_samples > 0 else 0
        print(f"    Weighted Accuracy: {weighted_overall_acc:.2f}%")

        # Save enhanced results for all stages combined
        output_file = os.path.join(results_dir, f"{exp}_predictions_concise.json")
        with open(output_file, "w") as f:
            json.dump({"overall_accuracy": overall_acc, "weighted_accuracy": weighted_overall_acc, "by_stage": stage_accuracies, "predictions": all_predictions}, f, indent=2)
        print(f"    Saved to: {os.path.basename(output_file)}")
    else:
        print(f"\n  ✓ {exp} Overall: 0.00%")
        print(f"    Weighted Accuracy: 0.00%")

print("\n================================================================================")
print("SUMMARY - CONCISE ANSWER ACCURACY")
print("================================================================================")

# Collect final summary
final_summary = {}

for exp in experiments:
    file = os.path.join(results_dir, f"{exp}_predictions_concise.json")
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
            final_summary[exp] = data.get('accuracy_concise', 0.0)
    else:
        final_summary[exp] = 0.0

for exp in multi_stage_experiments:
    file = os.path.join(results_dir, f"{exp}_predictions_concise.json")
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
            final_summary[exp] = data.get('weighted_accuracy', 0.0)
    else:
        final_summary[exp] = 0.0

for exp_name, acc_val in final_summary.items():
    print(f"  {exp_name:<20}: {acc_val:.2f}%")

print("\n================================================================================")
print("✓ Post-processing complete!")
print("================================================================================")

