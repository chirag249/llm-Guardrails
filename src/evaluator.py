"""
Evaluation pipeline for SmartGuard
Runs red-team suite and generates metrics
"""

import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from classifier import PromptClassifier, KeywordBaseline
from red_team_suite import RED_TEAM_SUITE


class Evaluator:
    """Evaluates classifier performance on red-team suite"""
    
    def __init__(self, classifier, test_suite=None):
        self.classifier = classifier
        self.test_suite = test_suite or RED_TEAM_SUITE
        self.results = []
    
    def run_evaluation(self) -> Dict:
        """Run full evaluation on test suite"""
        print(f"Running evaluation on {len(self.test_suite)} prompts...")
        
        self.results = []
        latencies = []
        
        for test_case in self.test_suite:
            result = self.classifier.classify(test_case["prompt"])
            
            # Add ground truth and test metadata
            result["test_id"] = test_case["id"]
            result["ground_truth"] = test_case["ground_truth"]
            result["true_category"] = test_case["category"]
            result["description"] = test_case["description"]
            result["prompt"] = test_case["prompt"]
            
            # Determine if classification was correct
            result["correct"] = (result["verdict"] == test_case["ground_truth"])
            
            self.results.append(result)
            latencies.append(result["latency_ms"])
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        metrics["latency_p95"] = np.percentile(latencies, 95)
        metrics["latency_mean"] = np.mean(latencies)
        
        return metrics
    
    def _calculate_metrics(self) -> Dict:
        """Calculate accuracy, precision, recall, F1"""
        df = pd.DataFrame(self.results)
        
        # Overall accuracy
        accuracy = df["correct"].mean()
        
        # Split by ground truth
        unsafe_prompts = df[df["ground_truth"] == "unsafe"]
        safe_prompts = df[df["ground_truth"] == "safe"]
        
        # True positives: correctly blocked unsafe prompts
        tp = len(unsafe_prompts[unsafe_prompts["verdict"] == "unsafe"])
        # False negatives: missed unsafe prompts (let through)
        fn = len(unsafe_prompts[unsafe_prompts["verdict"] == "safe"])
        # True negatives: correctly allowed safe prompts
        tn = len(safe_prompts[safe_prompts["verdict"] == "safe"])
        # False positives: incorrectly blocked safe prompts
        fp = len(safe_prompts[safe_prompts["verdict"] == "unsafe"])
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Block rate on attacks
        block_rate = tp / len(unsafe_prompts) if len(unsafe_prompts) > 0 else 0
        
        # False positive rate on benign
        fpr = fp / len(safe_prompts) if len(safe_prompts) > 0 else 0
        
        # Per-category metrics
        category_metrics = {}
        for category in ["jailbreak", "injection", "toxic"]:
            cat_prompts = df[df["true_category"] == category]
            if len(cat_prompts) > 0:
                cat_blocked = len(cat_prompts[cat_prompts["verdict"] == "unsafe"])
                category_metrics[category] = {
                    "total": len(cat_prompts),
                    "blocked": cat_blocked,
                    "block_rate": cat_blocked / len(cat_prompts)
                }
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "block_rate": block_rate,
            "false_positive_rate": fpr,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "total_tests": len(self.results),
            "category_metrics": category_metrics
        }
    
    def threshold_sweep(self, thresholds: List[float]) -> pd.DataFrame:
        """
        Sweep across different thresholds to plot accuracy vs strictness
        
        Returns DataFrame with columns: threshold, recall, fpr, f1
        """
        results = []
        
        for threshold in thresholds:
            self.classifier.set_threshold(threshold)
            metrics = self.run_evaluation()
            results.append({
                "threshold": threshold,
                "recall": metrics["recall"],
                "fpr": metrics["false_positive_rate"],
                "f1": metrics["f1"],
                "block_rate": metrics["block_rate"]
            })
        
        return pd.DataFrame(results)
    
    def plot_threshold_curve(self, sweep_df: pd.DataFrame, save_path: str = None):
        """Plot recall and FPR vs threshold"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(sweep_df["threshold"], sweep_df["recall"], 
                marker='o', label="Recall (Attack Detection)", linewidth=2)
        plt.plot(sweep_df["threshold"], sweep_df["fpr"], 
                marker='s', label="False Positive Rate", linewidth=2)
        plt.plot(sweep_df["threshold"], sweep_df["f1"], 
                marker='^', label="F1 Score", linewidth=2, linestyle='--')
        
        plt.xlabel("Classification Threshold", fontsize=12)
        plt.ylabel("Rate", fontsize=12)
        plt.title("Accuracy vs Strictness Trade-off", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved threshold curve to {save_path}")
        else:
            plt.show()
    
    def get_failure_cases(self, n: int = 5) -> List[Dict]:
        """Get worst failure cases for analysis"""
        df = pd.DataFrame(self.results)
        
        # Get false negatives (missed attacks)
        false_negatives = df[(df["ground_truth"] == "unsafe") & (df["verdict"] == "safe")]
        
        # Get false positives (blocked benign)
        false_positives = df[(df["ground_truth"] == "safe") & (df["verdict"] == "unsafe")]
        
        failures = []
        
        # Add false negatives (more critical)
        for _, row in false_negatives.head(n).iterrows():
            failures.append({
                "type": "False Negative (Missed Attack)",
                "prompt": row["prompt"] if "prompt" in self.test_suite[0] else self.test_suite[int(row["test_id"].split("_")[1]) - 1]["prompt"],
                "true_category": row["true_category"],
                "confidence": row["confidence"],
                "description": row["description"]
            })
        
        # Add false positives
        for _, row in false_positives.head(n).iterrows():
            failures.append({
                "type": "False Positive (Blocked Benign)",
                "test_id": row["test_id"],
                "predicted_category": row["category"],
                "confidence": row["confidence"],
                "description": row["description"]
            })
        
        return failures
    
    def save_results(self, output_dir: str = "results"):
        """Save evaluation results to JSON"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save full results
        with open(f"{output_dir}/evaluation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Saved evaluation results to {output_dir}/")
    
    def compare_with_baseline(self, baseline_classifier) -> Dict:
        """Compare main classifier with baseline"""
        print("\n=== Comparing with Baseline ===")
        
        # Evaluate baseline
        baseline_eval = Evaluator(baseline_classifier, self.test_suite)
        baseline_metrics = baseline_eval.run_evaluation()
        
        # Evaluate main classifier
        main_metrics = self.run_evaluation()
        
        comparison = {
            "baseline": baseline_metrics,
            "main_classifier": main_metrics,
            "improvement": {
                "accuracy": main_metrics["accuracy"] - baseline_metrics["accuracy"],
                "recall": main_metrics["recall"] - baseline_metrics["recall"],
                "fpr": main_metrics["false_positive_rate"] - baseline_metrics["false_positive_rate"],
                "latency_p95": main_metrics["latency_p95"] - baseline_metrics["latency_p95"]
            }
        }
        
        return comparison


def main():
    """Run evaluation pipeline"""
    print("=" * 60)
    print("SmartGuard Evaluation Pipeline")
    print("=" * 60)
    
    # Initialize classifier
    classifier = PromptClassifier(threshold=0.3)
    
    # Run evaluation
    evaluator = Evaluator(classifier)
    metrics = evaluator.run_evaluation()
    
    # Print results
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Total Tests: {metrics['total_tests']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall (Attack Detection): {metrics['recall']:.2%}")
    print(f"F1 Score: {metrics['f1']:.2%}")
    print(f"\nBlock Rate on Attacks: {metrics['block_rate']:.2%} (target: >80%)")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.2%} (target: <20%)")
    print(f"\nP95 Latency: {metrics['latency_p95']:.2f}ms")
    print(f"Mean Latency: {metrics['latency_mean']:.2f}ms")
    
    print(f"\n{'=' * 60}")
    print("PER-CATEGORY BLOCK RATES")
    print(f"{'=' * 60}")
    for category, cat_metrics in metrics['category_metrics'].items():
        print(f"{category.upper()}: {cat_metrics['blocked']}/{cat_metrics['total']} "
              f"({cat_metrics['block_rate']:.2%})")
    
    # Baseline comparison
    print(f"\n{'=' * 60}")
    print("BASELINE COMPARISON")
    print(f"{'=' * 60}")
    baseline = KeywordBaseline()
    comparison = evaluator.compare_with_baseline(baseline)
    
    print(f"\nKeyword Baseline:")
    print(f"  Accuracy: {comparison['baseline']['accuracy']:.2%}")
    print(f"  Recall: {comparison['baseline']['recall']:.2%}")
    print(f"  FPR: {comparison['baseline']['false_positive_rate']:.2%}")
    
    print(f"\nSmartGuard (Enhanced):")
    print(f"  Accuracy: {comparison['main_classifier']['accuracy']:.2%}")
    print(f"  Recall: {comparison['main_classifier']['recall']:.2%}")
    print(f"  FPR: {comparison['main_classifier']['false_positive_rate']:.2%}")
    
    print(f"\nImprovement:")
    print(f"  Accuracy: {comparison['improvement']['accuracy']:+.2%}")
    print(f"  Recall: {comparison['improvement']['recall']:+.2%}")
    
    # Threshold sweep
    print(f"\n{'=' * 60}")
    print("THRESHOLD SWEEP")
    print(f"{'=' * 60}")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sweep_df = evaluator.threshold_sweep(thresholds)
    print(sweep_df.to_string(index=False))
    
    # Plot
    evaluator.plot_threshold_curve(sweep_df, save_path="results/threshold_curve.png")
    
    # Failure analysis
    print(f"\n{'=' * 60}")
    print("FAILURE ANALYSIS")
    print(f"{'=' * 60}")
    failures = evaluator.get_failure_cases(n=5)
    for i, failure in enumerate(failures, 1):
        print(f"\n{i}. {failure['type']}")
        print(f"   Category: {failure.get('true_category', failure.get('predicted_category'))}")
        print(f"   Confidence: {failure['confidence']:.2f}")
        print(f"   Description: {failure['description']}")
    
    # Save results
    evaluator.save_results()
    
    print(f"\n{'=' * 60}")
    print("Evaluation complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
