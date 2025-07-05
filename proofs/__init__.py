"""
Self-Proving Module for ReasonIt.

This module provides formal verification capabilities for reasoning outputs,
including proof generation, formal verification, and certificate generation
for trustworthy reasoning attestation.
"""

from .proof_generator import (
    ProofGenerator,
    ProofCertificate,
    ProofStep,
    ProofType,
    VerificationLevel,
    VerificationConstraint
)

from .formal_verifier import (
    FormalVerifier,
    VerificationReport,
    VerificationResult,
    ConstraintType,
    LogicalFormula
)

from .certificate_generator import (
    CertificateGenerator,
    ReasoningCertificate,
    CertificateLevel,
    CertificateStatus,
    CertificateMetrics
)

__all__ = [
    # Proof Generation
    "ProofGenerator",
    "ProofCertificate", 
    "ProofStep",
    "ProofType",
    "VerificationLevel",
    "VerificationConstraint",
    
    # Formal Verification
    "FormalVerifier",
    "VerificationReport",
    "VerificationResult",
    "ConstraintType",
    "LogicalFormula",
    
    # Certificate Generation
    "CertificateGenerator",
    "ReasoningCertificate",
    "CertificateLevel",
    "CertificateStatus",
    "CertificateMetrics"
]

# Version information
__version__ = "1.0.0"
__author__ = "ReasonIt Development Team"
__description__ = "Self-Proving Module for formal verification of reasoning outputs"