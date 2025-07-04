"""MMLU (Massive Multitask Language Understanding) benchmark implementation."""

import json
import random
from pathlib import Path
from typing import Optional, List, Any, Dict
import asyncio
import aiohttp
from collections import defaultdict
import pandas as pd
import io

from benchmarks.base_benchmark import BaseBenchmark, BenchmarkSample


# MMLU subjects and their categories
MMLU_SUBJECTS = {
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


class MMLUBenchmark(BaseBenchmark):
    """MMLU benchmark for general knowledge and reasoning.
    
    Tests broad knowledge across multiple domains.
    Target: 75%+ accuracy at <$0.01 per problem.
    """
    
    # HuggingFace MMLU dataset API URL
    HF_API_URL = "https://datasets-server.huggingface.co/rows?dataset=cais/mmlu&config={subject}&split=test&offset=0&length=100"
    
    def __init__(self, data_path: Optional[Path] = None, subjects: Optional[List[str]] = None):
        super().__init__(name="MMLU", data_path=data_path)
        self.subjects = subjects or list(MMLU_SUBJECTS.keys())
        self.subject_results = defaultdict(list)
        
    async def load_data(self, num_samples: Optional[int] = None) -> None:
        """Load MMLU test data from selected subjects."""
        self.samples = []
        
        # If num_samples specified, distribute across subjects
        if num_samples:
            samples_per_subject = max(1, num_samples // len(self.subjects))
            remaining = num_samples - (samples_per_subject * len(self.subjects))
        else:
            samples_per_subject = 10  # Default samples per subject
            remaining = 0
        
        print(f"Loading MMLU data from {len(self.subjects)} subjects...")
        
        async with aiohttp.ClientSession() as session:
            for i, subject in enumerate(self.subjects):
                subject_samples = await self._load_subject(
                    session, 
                    subject, 
                    samples_per_subject + (1 if i < remaining else 0)
                )
                self.samples.extend(subject_samples)
                
                if num_samples and len(self.samples) >= num_samples:
                    break
        
        # Shuffle to mix subjects
        random.shuffle(self.samples)
        
        # Limit to requested number
        if num_samples:
            self.samples = self.samples[:num_samples]
            
        print(f"Loaded {len(self.samples)} MMLU samples")
    
    async def _load_subject(self, session: aiohttp.ClientSession, subject: str, max_samples: int) -> List[BenchmarkSample]:
        """Load samples from a specific subject using HuggingFace API."""
        samples = []
        
        # HuggingFace API URL for the subject
        url = self.HF_API_URL.format(subject=subject)
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    rows = data.get('rows', [])
                    
                    for i, row in enumerate(rows[:max_samples]):
                        sample = self._parse_hf_row(row, subject, i)
                        if sample:
                            samples.append(sample)
                else:
                    print(f"Failed to load {subject}: HTTP {response.status}")
        except Exception as e:
            print(f"Error loading {subject}: {e}")
        
        return samples
    
    def _parse_hf_row(self, row: Dict[str, Any], subject: str, index: int) -> Optional[BenchmarkSample]:
        """Parse a HuggingFace row into a benchmark sample."""
        try:
            row_data = row.get('row', {})
            question = row_data.get('question', '')
            choices = row_data.get('choices', [])
            answer_index = row_data.get('answer', 0)
            
            if not question or not choices or len(choices) < 4:
                return None
            
            # Convert answer index to letter (0=A, 1=B, 2=C, 3=D)
            answer_letter = chr(ord('A') + answer_index)
            
            return BenchmarkSample(
                id=f"mmlu_{subject}_{index}",
                question=question,
                expected_answer=answer_letter,
                metadata={
                    'subject': subject,
                    'category': MMLU_SUBJECTS.get(subject, 'other'),
                    'choices': choices,
                    'answer_index': answer_index,
                    'answer_text': choices[answer_index] if 0 <= answer_index < len(choices) else ""
                }
            )
        except Exception as e:
            print(f"Error parsing row {index} for {subject}: {e}")
            return None
    
    def _parse_csv_line(self, line: str, subject: str, index: int) -> Optional[BenchmarkSample]:
        """Parse a CSV line into a benchmark sample."""
        # Simple CSV parsing (MMLU format: question,A,B,C,D,answer)
        parts = line.split(',')
        if len(parts) < 6:
            return None
            
        question = parts[0].strip()
        choices = [parts[i].strip() for i in range(1, 5)]
        answer_letter = parts[5].strip()
        
        # Convert answer letter to index (A=0, B=1, C=2, D=3)
        answer_index = ord(answer_letter.upper()) - ord('A')
        
        return BenchmarkSample(
            id=f"mmlu_{subject}_{index}",
            question=question,
            expected_answer=answer_letter.upper(),
            metadata={
                'subject': subject,
                'category': MMLU_SUBJECTS.get(subject, 'other'),
                'choices': choices,
                'answer_index': answer_index,
                'answer_text': choices[answer_index] if 0 <= answer_index < 4 else ""
            }
        )
    
    def format_question(self, sample: BenchmarkSample) -> str:
        """Format the multiple choice question for the agent."""
        choices = sample.metadata['choices']
        question_text = (
            f"Subject: {sample.metadata['subject'].replace('_', ' ').title()}\n\n"
            f"Question: {sample.question}\n\n"
            f"A) {choices[0]}\n"
            f"B) {choices[1]}\n"
            f"C) {choices[2]}\n"
            f"D) {choices[3]}\n\n"
            f"Think step by step and select the best answer. "
            f"Respond with just the letter (A, B, C, or D) of your choice."
        )
        return question_text
    
    def evaluate_answer(self, predicted: Any, expected: Any) -> bool:
        """Check if the predicted answer matches expected."""
        if not predicted:
            return False
            
        # Extract just the letter from the prediction
        pred_letter = self._extract_answer_letter(str(predicted))
        exp_letter = str(expected).upper()
        
        return pred_letter == exp_letter
    
    def extract_answer(self, agent_response: str) -> Any:
        """Extract the answer choice from agent response."""
        return self._extract_answer_letter(agent_response)
    
    def _extract_answer_letter(self, text: str) -> Optional[str]:
        """Extract answer letter (A, B, C, or D) from text."""
        # Clean the text
        text = text.strip().upper()
        
        # Direct match
        if text in ['A', 'B', 'C', 'D']:
            return text
        
        # Look for patterns
        import re
        
        # Pattern: "answer is X", "choose X", "select X", etc.
        patterns = [
            r'(?:answer|choice|select|choose)\s*(?:is|:)?\s*([ABCD])',
            r'([ABCD])\s*(?:is|seems|appears)\s*(?:correct|right|best)',
            r'^\s*([ABCD])\s*[\.:\)]\s*',  # Starting with letter
            r'\b([ABCD])\b(?![\w])',  # Isolated letter
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # Last resort: find any A, B, C, or D
        for letter in ['A', 'B', 'C', 'D']:
            if letter in text:
                return letter
                
        return None
    
    async def run_benchmark(self, *args, **kwargs) -> "BenchmarkReport":
        """Override to track per-subject results."""
        report = await super().run_benchmark(*args, **kwargs)
        
        # Calculate per-subject accuracy
        for result in report.results:
            subject = self.samples[int(result.sample_id.split('_')[-1])].metadata['subject']
            self.subject_results[subject].append(result.is_correct)
        
        # Add subject breakdown to metadata
        subject_accuracy = {}
        for subject, results in self.subject_results.items():
            if results:
                subject_accuracy[subject] = sum(results) / len(results)
        
        report.metadata['subject_accuracy'] = subject_accuracy
        report.metadata['category_accuracy'] = self._calculate_category_accuracy(subject_accuracy)
        
        return report
    
    def _calculate_category_accuracy(self, subject_accuracy: Dict[str, float]) -> Dict[str, float]:
        """Calculate accuracy by category (STEM, Humanities, etc.)."""
        category_scores = defaultdict(list)
        
        for subject, accuracy in subject_accuracy.items():
            category = MMLU_SUBJECTS.get(subject, 'other')
            category_scores[category].append(accuracy)
        
        return {
            category: sum(scores) / len(scores) 
            for category, scores in category_scores.items()
            if scores
        }
    
    def print_summary(self, report: "BenchmarkReport") -> None:
        """Override to include subject breakdown."""
        super().print_summary(report)
        
        # Print category breakdown
        if 'category_accuracy' in report.metadata:
            print("\nAccuracy by Category:")
            for category, accuracy in sorted(report.metadata['category_accuracy'].items()):
                print(f"  {category.title()}: {accuracy:.1%}")


async def run_mmlu_evaluation(
    agent,
    num_samples: int = 100,
    subjects: Optional[List[str]] = None,
    strategy: Optional[str] = None,
    output_dir: Path = Path("benchmark_results")
):
    """Run MMLU evaluation with an agent."""
    # Default to a diverse set of subjects if none specified
    if not subjects:
        subjects = [
            "elementary_mathematics",
            "high_school_physics", 
            "college_computer_science",
            "world_history",
            "moral_scenarios",
            "business_ethics",
            "clinical_knowledge",
            "sociology"
        ]
    
    benchmark = MMLUBenchmark(subjects=subjects)
    
    # Run benchmark
    report = await benchmark.run_benchmark(
        agent=agent,
        num_samples=num_samples,
        strategy=strategy,
        use_tools=False,  # Generally no tools needed for MMLU
        parallel=True,
        max_concurrent=10  # Higher concurrency for simple questions
    )
    
    # Print summary
    benchmark.print_summary(report)
    
    # Save detailed results
    benchmark.save_report(report, output_dir)
    
    # Check if we meet target
    print(f"\nTarget: 75%+ accuracy at <$0.01 per problem")
    print(f"Achieved: {report.accuracy:.1%} at ${report.average_cost_per_sample:.3f} per problem")
    
    if report.accuracy >= 0.75 and report.average_cost_per_sample < 0.01:
        print("✅ Target met!")
    else:
        print("❌ Target not met")
    
    return report


if __name__ == "__main__":
    # Example usage
    async def main():
        # This would normally import your agent
        # from agents.adaptive_agent import AdaptiveAgent
        # agent = AdaptiveAgent()
        
        print("MMLU benchmark module ready.")
        print("Usage: await run_mmlu_evaluation(agent, num_samples=100)")
        print(f"Available subjects: {len(MMLU_SUBJECTS)}")
        
    asyncio.run(main())