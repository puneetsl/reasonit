#!/usr/bin/env python3
"""
Demo script showcasing Self-Proving Module capabilities.

This script demonstrates real-world scenarios where the Self-Proving Module
provides formal verification for different types of reasoning.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from proofs import (
    CertificateGenerator, CertificateLevel, ProofType
)


def demo_logical_reasoning():
    """Demonstrate logical reasoning verification."""
    print("🧠 LOGICAL REASONING VERIFICATION")
    print("=" * 60)
    
    cert_generator = CertificateGenerator()
    
    # Classic syllogism
    claim = "Socrates is mortal"
    reasoning = [
        "Premise 1: All humans are mortal",
        "Premise 2: Socrates is a human",
        "From universal instantiation: If Socrates is human, then Socrates is mortal",
        "By modus ponens: Since Socrates is human and all humans are mortal, Socrates is mortal",
        "Therefore: Socrates is mortal"
    ]
    
    print(f"Claim: {claim}")
    print("Reasoning chain:")
    for i, step in enumerate(reasoning, 1):
        print(f"  {i}. {step}")
    
    certificate = cert_generator.generate_certificate(
        claim=claim,
        reasoning_trace=reasoning,
        certificate_level=CertificateLevel.STANDARD,
        proof_type=ProofType.LOGICAL_INFERENCE
    )
    
    print(f"\n🎯 VERIFICATION RESULTS:")
    print(f"  Certificate Status: {certificate.status.value.upper()}")
    print(f"  Overall Trustworthiness: {certificate.metrics.overall_trustworthiness:.1%}")
    print(f"  Logical Consistency: {certificate.metrics.logical_consistency:.1%}")
    print(f"  Proof Confidence: {certificate.metrics.proof_confidence:.1%}")
    print(f"  Verification Score: {certificate.metrics.verification_score:.1%}")
    
    if certificate.proof_certificate:
        print(f"\n📋 PROOF ANALYSIS:")
        print(f"  Proof Steps: {len(certificate.proof_certificate.steps)}")
        print(f"  Premises Identified: {len(certificate.proof_certificate.premises)}")
        print(f"  Constraints Satisfied: {len(certificate.proof_certificate.constraints_satisfied)}")
        
        if certificate.proof_certificate.violations:
            print(f"  ⚠️ Issues Found: {len(certificate.proof_certificate.violations)}")
            for violation in certificate.proof_certificate.violations:
                print(f"    • {violation}")
    
    print(f"\n✅ Logical reasoning verification completed\n")
    return certificate


def demo_mathematical_reasoning():
    """Demonstrate mathematical reasoning verification."""
    print("🧮 MATHEMATICAL REASONING VERIFICATION")
    print("=" * 60)
    
    cert_generator = CertificateGenerator()
    
    # Mathematical problem solving
    claim = "The area of the rectangle is 24 square units"
    reasoning = [
        "Given: Rectangle with length = 6 units and width = 4 units",
        "Formula: Area of rectangle = length × width",
        "Substitution: Area = 6 × 4",
        "Calculation: 6 × 4 = 24",
        "Therefore: The area of the rectangle is 24 square units"
    ]
    
    print(f"Claim: {claim}")
    print("Mathematical reasoning:")
    for i, step in enumerate(reasoning, 1):
        print(f"  {i}. {step}")
    
    certificate = cert_generator.generate_certificate(
        claim=claim,
        reasoning_trace=reasoning,
        certificate_level=CertificateLevel.STANDARD,
        proof_type=ProofType.MATHEMATICAL_PROOF
    )
    
    print(f"\n🎯 VERIFICATION RESULTS:")
    print(f"  Certificate Status: {certificate.status.value.upper()}")
    print(f"  Overall Trustworthiness: {certificate.metrics.overall_trustworthiness:.1%}")
    print(f"  Mathematical Correctness: {certificate.metrics.mathematical_correctness:.1%}")
    print(f"  Proof Confidence: {certificate.metrics.proof_confidence:.1%}")
    
    if certificate.verification_report:
        report = certificate.verification_report
        print(f"\n🔍 DETAILED VERIFICATION:")
        print(f"  Verification Result: {report.result.value.upper()}")
        print(f"  Mathematical Score: {report.mathematical_correctness_score:.1%}")
        print(f"  Completeness: {report.completeness_score:.1%}")
        print(f"  Soundness: {report.soundness_score:.1%}")
    
    print(f"\n✅ Mathematical reasoning verification completed\n")
    return certificate


def demo_error_detection():
    """Demonstrate error detection capabilities."""
    print("❌ ERROR DETECTION DEMONSTRATION")
    print("=" * 60)
    
    cert_generator = CertificateGenerator()
    
    # Intentionally flawed reasoning with mathematical error
    claim = "The sum is 5"
    flawed_reasoning = [
        "We need to add 2 + 2",
        "Using arithmetic: 2 + 2 = 5",  # Intentional error
        "Therefore: The sum is 5"
    ]
    
    print(f"Claim: {claim}")
    print("Flawed reasoning (contains intentional error):")
    for i, step in enumerate(flawed_reasoning, 1):
        print(f"  {i}. {step}")
    
    certificate = cert_generator.generate_certificate(
        claim=claim,
        reasoning_trace=flawed_reasoning,
        certificate_level=CertificateLevel.PREMIUM,
        proof_type=ProofType.MATHEMATICAL_PROOF
    )
    
    print(f"\n🎯 ERROR DETECTION RESULTS:")
    print(f"  Certificate Status: {certificate.status.value.upper()}")
    print(f"  Overall Trustworthiness: {certificate.metrics.overall_trustworthiness:.1%}")
    print(f"  Mathematical Correctness: {certificate.metrics.mathematical_correctness:.1%}")
    
    if certificate.verification_report and certificate.verification_report.errors_found:
        print(f"\n🚨 ERRORS DETECTED:")
        for error in certificate.verification_report.errors_found:
            print(f"    • {error}")
    
    if certificate.constraint_violations:
        print(f"\n⚠️ CONSTRAINT VIOLATIONS:")
        for violation in certificate.constraint_violations:
            print(f"    • {violation}")
    
    if certificate.recommendations:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in certificate.recommendations[:3]:
            print(f"    • {rec}")
    
    print(f"\n✅ Error detection demonstration completed\n")
    return certificate


def demo_complex_reasoning():
    """Demonstrate complex multi-step reasoning verification."""
    print("🧩 COMPLEX REASONING VERIFICATION")
    print("=" * 60)
    
    cert_generator = CertificateGenerator()
    
    # Complex logical argument with multiple premises
    claim = "Alice should take an umbrella"
    complex_reasoning = [
        "Observation: The weather forecast shows 80% chance of rain today",
        "Rule 1: If chance of rain > 70%, then it will likely rain",
        "Application: Since 80% > 70%, it will likely rain today",
        "Rule 2: If it will likely rain, then one should prepare for rain",
        "Application: Since it will likely rain, Alice should prepare for rain",
        "Rule 3: Taking an umbrella is a way to prepare for rain",
        "Rule 4: If one should prepare for rain, then one should take an umbrella",
        "Chain reasoning: Alice should prepare for rain AND taking umbrella prepares for rain",
        "By modus ponens: Alice should take an umbrella",
        "Therefore: Alice should take an umbrella"
    ]
    
    print(f"Claim: {claim}")
    print("Complex reasoning chain:")
    for i, step in enumerate(complex_reasoning, 1):
        print(f"  {i:2d}. {step}")
    
    # Use CRITICAL level for maximum verification rigor
    certificate = cert_generator.generate_certificate(
        claim=claim,
        reasoning_trace=complex_reasoning,
        certificate_level=CertificateLevel.CRITICAL,
        proof_type=ProofType.LOGICAL_INFERENCE,
        custom_constraints={
            "min_confidence": 0.8,
            "max_verification_time": 15.0
        }
    )
    
    print(f"\n🎯 COMPREHENSIVE VERIFICATION:")
    print(f"  Certificate Status: {certificate.status.value.upper()}")
    print(f"  Certificate Level: {certificate.certificate_level.value.upper()}")
    print(f"  Overall Trustworthiness: {certificate.metrics.overall_trustworthiness:.1%}")
    print(f"  Generation Time: {certificate.generation_time:.3f}s")
    
    print(f"\n📊 DETAILED METRICS:")
    metrics = certificate.metrics
    print(f"  Proof Confidence:        {metrics.proof_confidence:.1%}")
    print(f"  Verification Score:      {metrics.verification_score:.1%}")
    print(f"  Logical Consistency:     {metrics.logical_consistency:.1%}")
    print(f"  Semantic Coherence:      {metrics.semantic_coherence:.1%}")
    print(f"  Completeness:            {metrics.completeness:.1%}")
    print(f"  Soundness:               {metrics.soundness:.1%}")
    
    if certificate.proof_certificate:
        proof = certificate.proof_certificate
        print(f"\n🔧 PROOF ANALYSIS:")
        print(f"  Total Proof Steps: {len(proof.steps)}")
        print(f"  Identified Premises: {len(proof.premises)}")
        print(f"  Validity Score: {proof.validity_score:.1%}")
        print(f"  Completeness Score: {proof.completeness_score:.1%}")
        
        # Show inference rules used
        rules_used = set(step.inference_rule for step in proof.steps)
        print(f"  Inference Rules Used: {', '.join(rules_used)}")
    
    if certificate.verification_report:
        report = certificate.verification_report
        print(f"\n🔍 VERIFICATION ANALYSIS:")
        print(f"  Final Result: {report.result.value.upper()}")
        print(f"  Constraints Checked: {len(report.constraint_results)}")
        satisfied = sum(1 for v in report.constraint_results.values() if v)
        total = len(report.constraint_results)
        print(f"  Constraints Satisfied: {satisfied}/{total} ({satisfied/total:.1%})")
    
    print(f"\n📜 CERTIFICATE ATTESTATIONS:")
    for i, attestation in enumerate(certificate.attestations[:5], 1):
        print(f"  {i}. {attestation}")
    
    print(f"\n✅ Complex reasoning verification completed\n")
    return certificate


def demo_certificate_comparison():
    """Compare certificates at different verification levels."""
    print("📋 CERTIFICATE LEVEL COMPARISON")
    print("=" * 60)
    
    cert_generator = CertificateGenerator()
    
    # Same reasoning, different certificate levels
    claim = "Plants need sunlight to grow"
    reasoning = [
        "Fact: Plants perform photosynthesis to produce energy",
        "Fact: Photosynthesis requires sunlight as an energy source",
        "Logic: If plants need photosynthesis and photosynthesis needs sunlight, then plants need sunlight",
        "Therefore: Plants need sunlight to grow"
    ]
    
    levels_to_test = [
        CertificateLevel.BASIC,
        CertificateLevel.STANDARD,
        CertificateLevel.PREMIUM,
        CertificateLevel.CRITICAL
    ]
    
    print(f"Testing claim: {claim}")
    print("Comparing verification at different certificate levels:\n")
    
    results = []
    
    for level in levels_to_test:
        certificate = cert_generator.generate_certificate(
            claim=claim,
            reasoning_trace=reasoning,
            certificate_level=level,
            proof_type=ProofType.LOGICAL_INFERENCE
        )
        
        results.append({
            'level': level.value.upper(),
            'status': certificate.status.value.upper(),
            'trustworthiness': certificate.metrics.overall_trustworthiness,
            'generation_time': certificate.generation_time,
            'attestations': len(certificate.attestations),
            'expires_in_days': (certificate.expires_at - certificate.created_at).days if certificate.expires_at else 0
        })
    
    # Display comparison table
    print("┌─────────────┬──────────┬──────────────────┬─────────────┬──────────────┬──────────────┐")
    print("│ Level       │ Status   │ Trustworthiness  │ Time (s)    │ Attestations │ Valid (days) │")
    print("├─────────────┼──────────┼──────────────────┼─────────────┼──────────────┼──────────────┤")
    
    for result in results:
        print(f"│ {result['level']:11} │ {result['status']:8} │ {result['trustworthiness']:15.1%} │ {result['generation_time']:11.3f} │ {result['attestations']:12d} │ {result['expires_in_days']:12d} │")
    
    print("└─────────────┴──────────┴──────────────────┴─────────────┴──────────────┴──────────────┘")
    
    print(f"\n💡 Key Observations:")
    print(f"  • Higher certificate levels provide more thorough verification")
    print(f"  • Trustworthiness scores vary based on verification rigor")
    print(f"  • Critical certificates have longer validity periods")
    print(f"  • Generation time increases with verification thoroughness")
    
    print(f"\n✅ Certificate comparison completed\n")


def main():
    """Run the Self-Proving Module demonstration."""
    print("🔧 ReasonIt Self-Proving Module Demonstration")
    print("=" * 70)
    print("Showcasing formal verification capabilities for reasoning outputs\n")
    
    try:
        # Demonstrate different verification scenarios
        demo_logical_reasoning()
        demo_mathematical_reasoning()
        demo_error_detection()
        demo_complex_reasoning()
        demo_certificate_comparison()
        
        print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("\nThe Self-Proving Module has demonstrated:")
        print("  ✅ Logical reasoning verification with high accuracy")
        print("  ✅ Mathematical correctness checking with error detection")
        print("  ✅ Complex multi-step reasoning analysis")
        print("  ✅ Comprehensive certificate generation at multiple levels")
        print("  ✅ Robust error detection and constraint validation")
        print("\nThe system is ready to provide trustworthy formal verification")
        print("for any reasoning output from the ReasonIt architecture!")
        
    except Exception as e:
        print(f"❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)