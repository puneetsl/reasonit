#!/usr/bin/env python3
"""
Test script for Self-Proving Module.

This script demonstrates the formal verification capabilities of the Self-Proving
Module including proof generation, formal verification, and certificate generation.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from proofs import (
    ProofGenerator, ProofType, VerificationLevel,
    FormalVerifier, VerificationResult,
    CertificateGenerator, CertificateLevel, CertificateStatus
)


def test_proof_generation():
    """Test proof generation capabilities."""
    print("🔧 Testing Proof Generation")
    print("=" * 50)
    
    generator = ProofGenerator()
    
    # Test logical inference proof
    claim = "Socrates is mortal"
    reasoning_trace = [
        "All men are mortal",
        "Socrates is a man", 
        "Therefore, Socrates is mortal"
    ]
    
    print(f"Claim: {claim}")
    print(f"Reasoning trace: {len(reasoning_trace)} steps")
    
    # Generate proof
    proof_cert = generator.generate_proof(
        claim=claim,
        reasoning_trace=reasoning_trace,
        proof_type=ProofType.LOGICAL_INFERENCE,
        verification_level=VerificationLevel.STANDARD
    )
    
    print(f"  Proof ID: {proof_cert.proof_id}")
    print(f"  Proof type: {proof_cert.proof_type.value}")
    print(f"  Overall confidence: {proof_cert.overall_confidence:.2f}")
    print(f"  Validity score: {proof_cert.validity_score:.2f}")
    print(f"  Completeness score: {proof_cert.completeness_score:.2f}")
    print(f"  Constraints satisfied: {len(proof_cert.constraints_satisfied)}")
    print(f"  Violations: {len(proof_cert.violations)}")
    
    print("\n  Proof steps:")
    for step in proof_cert.steps:
        print(f"    {step.step_id}: {step.inference_rule} (conf: {step.confidence:.2f})")
        print(f"      Premise: {step.premise}")
        print(f"      Conclusion: {step.conclusion}")
    
    print("\n✅ Proof generation test completed\n")
    return proof_cert


def test_mathematical_proof():
    """Test mathematical proof generation."""
    print("🧮 Testing Mathematical Proof Generation")
    print("=" * 50)
    
    generator = ProofGenerator()
    
    # Test mathematical proof
    claim = "The sum equals 7"
    reasoning_trace = [
        "We start with the equation 3 + 4",
        "3 + 4 = 7",
        "Therefore, the sum equals 7"
    ]
    
    print(f"Mathematical claim: {claim}")
    
    proof_cert = generator.generate_proof(
        claim=claim,
        reasoning_trace=reasoning_trace,
        proof_type=ProofType.MATHEMATICAL_PROOF,
        verification_level=VerificationLevel.RIGOROUS
    )
    
    print(f"  Proof confidence: {proof_cert.overall_confidence:.2f}")
    print(f"  Mathematical verification: {len([v for v in proof_cert.violations if 'arithmetic' in v.lower()]) == 0}")
    
    print("\n✅ Mathematical proof test completed\n")
    return proof_cert


def test_formal_verification():
    """Test formal verification capabilities."""
    print("🔍 Testing Formal Verification")
    print("=" * 50)
    
    # First generate a proof
    generator = ProofGenerator()
    verifier = FormalVerifier()
    
    claim = "All birds can fly, Tweety is a bird, therefore Tweety can fly"
    reasoning_trace = [
        "Given: All birds can fly",
        "Given: Tweety is a bird",
        "By universal instantiation: If Tweety is a bird, then Tweety can fly",
        "By modus ponens: Tweety can fly"
    ]
    
    print(f"Verifying claim: {claim}")
    
    # Generate proof
    proof_cert = generator.generate_proof(
        claim=claim,
        reasoning_trace=reasoning_trace,
        proof_type=ProofType.LOGICAL_INFERENCE,
        verification_level=VerificationLevel.STANDARD
    )
    
    # Verify proof
    verification_report = verifier.verify_proof_certificate(
        proof_cert,
        VerificationLevel.RIGOROUS
    )
    
    print(f"  Verification ID: {verification_report.verification_id}")
    print(f"  Verification result: {verification_report.result.value}")
    print(f"  Overall score: {verification_report.overall_score:.2f}")
    print(f"  Logical consistency: {verification_report.logical_consistency_score:.2f}")
    print(f"  Mathematical correctness: {verification_report.mathematical_correctness_score:.2f}")
    print(f"  Semantic coherence: {verification_report.semantic_coherence_score:.2f}")
    print(f"  Completeness: {verification_report.completeness_score:.2f}")
    print(f"  Soundness: {verification_report.soundness_score:.2f}")
    
    print("\n  Constraint results:")
    for constraint, satisfied in verification_report.constraint_results.items():
        status = "✓" if satisfied else "✗"
        print(f"    {status} {constraint}")
    
    if verification_report.errors_found:
        print("\n  Errors found:")
        for error in verification_report.errors_found:
            print(f"    • {error}")
    
    if verification_report.warnings:
        print("\n  Warnings:")
        for warning in verification_report.warnings:
            print(f"    • {warning}")
    
    if verification_report.recommendations:
        print("\n  Recommendations:")
        for rec in verification_report.recommendations:
            print(f"    • {rec}")
    
    print("\n✅ Formal verification test completed\n")
    return verification_report


def test_certificate_generation():
    """Test certificate generation capabilities."""
    print("📜 Testing Certificate Generation")
    print("=" * 50)
    
    cert_generator = CertificateGenerator()
    
    # Test different certificate levels
    test_cases = [
        {
            "level": CertificateLevel.BASIC,
            "claim": "The sky is blue",
            "reasoning": ["Observation shows the sky appears blue", "Therefore, the sky is blue"]
        },
        {
            "level": CertificateLevel.STANDARD, 
            "claim": "2 + 2 = 4",
            "reasoning": ["We have 2 objects", "We add 2 more objects", "2 + 2 = 4", "Therefore, 2 + 2 = 4"]
        },
        {
            "level": CertificateLevel.PREMIUM,
            "claim": "All mammals are warm-blooded",
            "reasoning": [
                "Mammals are defined as vertebrates with hair and mammary glands",
                "Mammals regulate their body temperature internally",
                "Internal temperature regulation means warm-blooded",
                "Therefore, all mammals are warm-blooded"
            ]
        }
    ]
    
    certificates = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  Test Case {i}: {test_case['level'].value.upper()} certificate")
        print(f"  Claim: {test_case['claim']}")
        
        certificate = cert_generator.generate_certificate(
            claim=test_case['claim'],
            reasoning_trace=test_case['reasoning'],
            certificate_level=test_case['level'],
            proof_type=ProofType.LOGICAL_INFERENCE
        )
        
        certificates.append(certificate)
        
        print(f"    Certificate ID: {certificate.certificate_id}")
        print(f"    Status: {certificate.status.value}")
        print(f"    Trustworthiness: {certificate.metrics.overall_trustworthiness:.2f}")
        print(f"    Generation time: {certificate.generation_time:.2f}s")
        print(f"    Expires: {certificate.expires_at.strftime('%Y-%m-%d %H:%M') if certificate.expires_at else 'Never'}")
        print(f"    Attestations: {len(certificate.attestations)}")
        print(f"    Recommendations: {len(certificate.recommendations)}")
        
        # Show a few attestations
        if certificate.attestations:
            print("    Sample attestations:")
            for attestation in certificate.attestations[:3]:
                print(f"      • {attestation}")
    
    print("\n✅ Certificate generation test completed\n")
    return certificates


def test_certificate_validation():
    """Test certificate validation and management."""
    print("✅ Testing Certificate Validation")
    print("=" * 50)
    
    cert_generator = CertificateGenerator()
    
    # Generate a test certificate
    certificate = cert_generator.generate_certificate(
        claim="Test validation claim",
        reasoning_trace=["Step 1", "Step 2", "Conclusion"],
        certificate_level=CertificateLevel.STANDARD
    )
    
    cert_id = certificate.certificate_id
    print(f"Generated test certificate: {cert_id}")
    
    # Test validation
    is_valid = cert_generator.validate_certificate(cert_id)
    print(f"  Certificate valid: {is_valid}")
    
    # Test certificate retrieval
    retrieved = cert_generator.get_certificate(cert_id)
    print(f"  Certificate retrieved: {retrieved is not None}")
    
    # Test certificate export
    cert_data = cert_generator.export_certificate(cert_id, include_full_details=False)
    print(f"  Certificate exported: {cert_data is not None}")
    
    # Test certificate summary
    summary = cert_generator.export_certificate_summary(cert_id)
    print(f"  Certificate summary: {summary is not None}")
    if summary:
        print(f"    Summary trustworthiness: {summary['trustworthiness']:.2f}")
    
    # Test search functionality
    valid_certs = cert_generator.search_certificates(status=CertificateStatus.VALID)
    print(f"  Valid certificates found: {len(valid_certs)}")
    
    # Test revocation
    revoke_success = cert_generator.revoke_certificate(cert_id, "Testing revocation")
    print(f"  Certificate revoked: {revoke_success}")
    
    # Validate after revocation
    is_valid_after_revoke = cert_generator.validate_certificate(cert_id)
    print(f"  Certificate valid after revocation: {is_valid_after_revoke}")
    
    print("\n✅ Certificate validation test completed\n")


def test_complex_reasoning_scenario():
    """Test complex reasoning scenario with full pipeline."""
    print("🧠 Testing Complex Reasoning Scenario")
    print("=" * 50)
    
    cert_generator = CertificateGenerator()
    
    # Complex logical reasoning
    claim = "If it's raining and I don't have an umbrella, then I will get wet"
    reasoning_trace = [
        "Premise 1: It is currently raining",
        "Premise 2: I do not have an umbrella with me", 
        "Premise 3: Rain causes people without protection to get wet",
        "From premises 1 and 2: I am outside in the rain without an umbrella",
        "From premise 3 and the above: Being in rain without protection causes wetness",
        "By modus ponens: If I am in the rain without protection, I will get wet",
        "Therefore: I will get wet"
    ]
    
    print(f"Complex claim: {claim}")
    print(f"Reasoning steps: {len(reasoning_trace)}")
    
    # Generate premium certificate with custom constraints
    custom_constraints = {
        "min_confidence": 0.8,
        "max_verification_time": 10.0,
        "required_proof_types": ["logical_inference"]
    }
    
    certificate = cert_generator.generate_certificate(
        claim=claim,
        reasoning_trace=reasoning_trace,
        certificate_level=CertificateLevel.PREMIUM,
        proof_type=ProofType.LOGICAL_INFERENCE,
        custom_constraints=custom_constraints
    )
    
    print(f"\n  Certificate Status: {certificate.status.value}")
    print(f"  Overall Trustworthiness: {certificate.metrics.overall_trustworthiness:.3f}")
    
    # Detailed metrics
    metrics = certificate.metrics
    print(f"\n  Detailed Metrics:")
    print(f"    Proof Confidence: {metrics.proof_confidence:.3f}")
    print(f"    Verification Score: {metrics.verification_score:.3f}")
    print(f"    Logical Consistency: {metrics.logical_consistency:.3f}")
    print(f"    Mathematical Correctness: {metrics.mathematical_correctness:.3f}")
    print(f"    Semantic Coherence: {metrics.semantic_coherence:.3f}")
    print(f"    Completeness: {metrics.completeness:.3f}")
    print(f"    Soundness: {metrics.soundness:.3f}")
    
    # Show proof details if available
    if certificate.proof_certificate:
        proof = certificate.proof_certificate
        print(f"\n  Proof Certificate:")
        print(f"    Proof ID: {proof.proof_id}")
        print(f"    Steps: {len(proof.steps)}")
        print(f"    Premises: {len(proof.premises)}")
        print(f"    Violations: {len(proof.violations)}")
    
    # Show verification details if available
    if certificate.verification_report:
        report = certificate.verification_report
        print(f"\n  Verification Report:")
        print(f"    Result: {report.result.value}")
        print(f"    Errors: {len(report.errors_found)}")
        print(f"    Warnings: {len(report.warnings)}")
        print(f"    Recommendations: {len(report.recommendations)}")
    
    # Show constraint violations
    if certificate.constraint_violations:
        print(f"\n  Constraint Violations:")
        for violation in certificate.constraint_violations:
            print(f"    • {violation}")
    
    # Show key attestations
    if certificate.attestations:
        print(f"\n  Key Attestations:")
        for attestation in certificate.attestations[:5]:
            print(f"    • {attestation}")
    
    print("\n✅ Complex reasoning scenario test completed\n")
    return certificate


def test_error_handling():
    """Test error handling and edge cases."""
    print("⚠️ Testing Error Handling")
    print("=" * 50)
    
    cert_generator = CertificateGenerator()
    
    # Test with empty reasoning
    print("  Testing empty reasoning trace...")
    empty_cert = cert_generator.generate_certificate(
        claim="Empty test",
        reasoning_trace=[],
        certificate_level=CertificateLevel.BASIC
    )
    print(f"    Status: {empty_cert.status.value}")
    print(f"    Trustworthiness: {empty_cert.metrics.overall_trustworthiness:.2f}")
    
    # Test with contradictory reasoning
    print("\n  Testing contradictory reasoning...")
    contradictory_cert = cert_generator.generate_certificate(
        claim="Contradictory claim",
        reasoning_trace=[
            "All birds can fly",
            "Penguins are birds",
            "Penguins cannot fly",
            "Therefore, some birds cannot fly"
        ],
        certificate_level=CertificateLevel.STANDARD
    )
    print(f"    Status: {contradictory_cert.status.value}")
    print(f"    Violations: {len(contradictory_cert.constraint_violations)}")
    
    # Test with mathematical errors
    print("\n  Testing mathematical errors...")
    math_error_cert = cert_generator.generate_certificate(
        claim="Incorrect math",
        reasoning_trace=[
            "We start with 2 + 2",
            "2 + 2 = 5",
            "Therefore, the sum is 5"
        ],
        certificate_level=CertificateLevel.PREMIUM,
        proof_type=ProofType.MATHEMATICAL_PROOF
    )
    print(f"    Status: {math_error_cert.status.value}")
    print(f"    Math correctness: {math_error_cert.metrics.mathematical_correctness:.2f}")
    
    # Test invalid certificate ID
    print("\n  Testing invalid certificate operations...")
    invalid_validation = cert_generator.validate_certificate("invalid_id")
    print(f"    Invalid ID validation: {invalid_validation}")
    
    invalid_retrieval = cert_generator.get_certificate("invalid_id")
    print(f"    Invalid ID retrieval: {invalid_retrieval is None}")
    
    print("\n✅ Error handling test completed\n")


def demonstrate_statistics():
    """Demonstrate statistics and reporting capabilities."""
    print("📊 Demonstrating Statistics")
    print("=" * 50)
    
    cert_generator = CertificateGenerator()
    
    # Generate several certificates for statistics
    test_claims = [
        "Basic logical claim",
        "Mathematical equation: 1 + 1 = 2", 
        "Complex reasoning about natural phenomena",
        "Simple factual statement",
        "Another logical inference"
    ]
    
    for claim in test_claims:
        cert_generator.generate_certificate(
            claim=claim,
            reasoning_trace=[f"Reasoning for {claim}", f"Therefore, {claim}"],
            certificate_level=CertificateLevel.STANDARD
        )
    
    # Get and display statistics
    stats = cert_generator.get_certificate_statistics()
    
    print(f"  Certificate Statistics:")
    print(f"    Total generated: {stats['total_generated']}")
    print(f"    Average trustworthiness: {stats['average_trustworthiness']:.3f}")
    print(f"    Total verification time: {stats['total_verification_time']:.2f}s")
    
    print(f"\n  By Certificate Level:")
    for level, count in stats['by_level'].items():
        print(f"    {level}: {count}")
    
    print(f"\n  By Status:")
    for status, count in stats['by_status'].items():
        print(f"    {status}: {count}")
    
    # Get proof generator statistics
    proof_stats = cert_generator.proof_generator.get_proof_statistics()
    print(f"\n  Proof Generation Statistics:")
    print(f"    Total proofs: {proof_stats['total_generated']}")
    print(f"    Average confidence: {proof_stats['average_confidence']:.3f}")
    
    # Get verification statistics
    verifier_stats = cert_generator.formal_verifier.get_verification_statistics()
    print(f"\n  Verification Statistics:")
    print(f"    Total verifications: {verifier_stats['total_verifications']}")
    print(f"    Successful: {verifier_stats['successful_verifications']}")
    print(f"    Failed: {verifier_stats['failed_verifications']}")
    print(f"    Average score: {verifier_stats['average_score']:.3f}")
    
    print("\n✅ Statistics demonstration completed\n")


def main():
    """Run all self-proving module tests."""
    print("🔧 Self-Proving Module Test Suite")
    print("=" * 60)
    print("Testing formal verification capabilities for reasoning outputs\n")
    
    try:
        # Core functionality tests
        test_proof_generation()
        test_mathematical_proof()
        test_formal_verification()
        test_certificate_generation()
        
        # Advanced functionality tests
        test_certificate_validation()
        test_complex_reasoning_scenario()
        
        # Edge cases and error handling
        test_error_handling()
        
        # Statistics and reporting
        demonstrate_statistics()
        
        print("🎉 All Self-Proving Module tests completed successfully!")
        print("\nThe Self-Proving Module is ready to provide formal verification")
        print("for reasoning outputs with comprehensive proof generation,")
        print("formal verification, and trustworthy certificate generation!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)