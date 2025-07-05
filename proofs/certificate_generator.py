"""
Certificate Generation Module for Self-Proving System.

This module generates comprehensive verification certificates that combine
proof generation with formal verification to create trustworthy reasoning attestations.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import uuid

from .proof_generator import ProofGenerator, ProofCertificate, ProofType, VerificationLevel
from .formal_verifier import FormalVerifier, VerificationReport, VerificationResult


class CertificateLevel(Enum):
    """Levels of certificate assurance."""
    BASIC = "basic"           # Basic verification
    STANDARD = "standard"     # Standard verification with proof
    PREMIUM = "premium"       # Comprehensive verification
    CRITICAL = "critical"     # Maximum security verification


class CertificateStatus(Enum):
    """Status of certificate generation."""
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"


@dataclass
class CertificateMetrics:
    """Metrics for certificate quality assessment."""
    proof_confidence: float = 0.0
    verification_score: float = 0.0
    logical_consistency: float = 0.0
    mathematical_correctness: float = 0.0
    semantic_coherence: float = 0.0
    completeness: float = 0.0
    soundness: float = 0.0
    overall_trustworthiness: float = 0.0


@dataclass
class ReasoningCertificate:
    """Comprehensive certificate for reasoning verification."""
    certificate_id: str
    certificate_level: CertificateLevel
    status: CertificateStatus
    
    # Core content
    original_claim: str
    reasoning_trace: List[str]
    final_conclusion: str
    
    # Verification components
    proof_certificate: Optional[ProofCertificate] = None
    verification_report: Optional[VerificationReport] = None
    
    # Certificate metrics
    metrics: CertificateMetrics = field(default_factory=CertificateMetrics)
    
    # Certificate metadata
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    issuer: str = "ReasonIt Self-Proving System"
    version: str = "1.0"
    
    # Cryptographic signature (placeholder for future implementation)
    signature: Optional[str] = None
    
    # Additional metadata
    generation_time: float = 0.0
    constraint_violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    attestations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CertificateGenerator:
    """
    Generates comprehensive verification certificates for reasoning outputs.
    
    Combines proof generation and formal verification to create trustworthy
    certificates that attest to the validity of reasoning processes.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.proof_generator = ProofGenerator()
        self.formal_verifier = FormalVerifier()
        
        # Certificate storage
        self.certificates: Dict[str, ReasoningCertificate] = {}
        
        # Certificate settings
        self.certificate_settings = {
            CertificateLevel.BASIC: {
                "verification_level": VerificationLevel.BASIC,
                "proof_required": False,
                "validity_duration_hours": 24,
                "min_confidence_threshold": 0.5
            },
            CertificateLevel.STANDARD: {
                "verification_level": VerificationLevel.STANDARD,
                "proof_required": True,
                "validity_duration_hours": 168,  # 1 week
                "min_confidence_threshold": 0.7
            },
            CertificateLevel.PREMIUM: {
                "verification_level": VerificationLevel.RIGOROUS,
                "proof_required": True,
                "validity_duration_hours": 720,  # 1 month
                "min_confidence_threshold": 0.8
            },
            CertificateLevel.CRITICAL: {
                "verification_level": VerificationLevel.EXHAUSTIVE,
                "proof_required": True,
                "validity_duration_hours": 8760,  # 1 year
                "min_confidence_threshold": 0.9
            }
        }
        
        # Certificate statistics
        self.certificate_stats = {
            "total_generated": 0,
            "by_level": {},
            "by_status": {},
            "average_trustworthiness": 0.0,
            "total_verification_time": 0.0
        }
    
    def generate_certificate(self,
                           claim: str,
                           reasoning_trace: List[str],
                           certificate_level: CertificateLevel = CertificateLevel.STANDARD,
                           proof_type: ProofType = ProofType.LOGICAL_INFERENCE,
                           additional_premises: Optional[List[str]] = None,
                           custom_constraints: Optional[Dict[str, Any]] = None) -> ReasoningCertificate:
        """
        Generate a comprehensive reasoning certificate.
        
        Args:
            claim: The main claim being verified
            reasoning_trace: List of reasoning steps
            certificate_level: Level of certificate assurance
            proof_type: Type of proof to generate
            additional_premises: Additional premises for proof
            custom_constraints: Custom verification constraints
            
        Returns:
            ReasoningCertificate with comprehensive verification
        """
        start_time = datetime.now()
        
        self.logger.info(f"Generating {certificate_level.value} certificate for claim: {claim[:100]}...")
        
        # Generate unique certificate ID
        certificate_id = self._generate_certificate_id(claim, reasoning_trace)
        
        # Get certificate settings
        settings = self.certificate_settings[certificate_level]
        verification_level = settings["verification_level"]
        
        # Initialize certificate
        certificate = ReasoningCertificate(
            certificate_id=certificate_id,
            certificate_level=certificate_level,
            status=CertificateStatus.PENDING,
            original_claim=claim,
            reasoning_trace=reasoning_trace,
            final_conclusion=self._extract_final_conclusion(reasoning_trace),
            expires_at=datetime.now() + timedelta(hours=settings["validity_duration_hours"])
        )
        
        try:
            # Step 1: Generate proof certificate if required
            if settings["proof_required"]:
                self.logger.debug("Generating proof certificate...")
                proof_cert = self.proof_generator.generate_proof(
                    claim=claim,
                    reasoning_trace=reasoning_trace,
                    proof_type=proof_type,
                    verification_level=verification_level,
                    additional_premises=additional_premises
                )
                certificate.proof_certificate = proof_cert
            
            # Step 2: Perform formal verification
            self.logger.debug("Performing formal verification...")
            if certificate.proof_certificate:
                verification_report = self.formal_verifier.verify_proof_certificate(
                    certificate.proof_certificate,
                    verification_level
                )
            else:
                # Create a basic proof for verification
                basic_proof = self.proof_generator.generate_proof(
                    claim=claim,
                    reasoning_trace=reasoning_trace,
                    proof_type=ProofType.CONSISTENCY_CHECK,
                    verification_level=VerificationLevel.BASIC
                )
                verification_report = self.formal_verifier.verify_proof_certificate(
                    basic_proof,
                    verification_level
                )
            
            certificate.verification_report = verification_report
            
            # Step 3: Calculate certificate metrics
            certificate.metrics = self._calculate_certificate_metrics(certificate)
            
            # Step 4: Apply custom constraints if provided
            if custom_constraints:
                self._apply_custom_constraints(certificate, custom_constraints)
            
            # Step 5: Determine certificate status
            certificate.status = self._determine_certificate_status(certificate, settings)
            
            # Step 6: Generate attestations
            certificate.attestations = self._generate_attestations(certificate)
            
            # Step 7: Generate recommendations
            certificate.recommendations = self._generate_certificate_recommendations(certificate)
            
            # Step 8: Generate cryptographic signature (placeholder)
            certificate.signature = self._generate_certificate_signature(certificate)
            
        except Exception as e:
            self.logger.error(f"Certificate generation failed: {e}")
            certificate.status = CertificateStatus.INVALID
            certificate.constraint_violations.append(f"Generation error: {str(e)}")
        
        # Finalize certificate
        end_time = datetime.now()
        certificate.generation_time = (end_time - start_time).total_seconds()
        
        # Store certificate
        self.certificates[certificate_id] = certificate
        self._update_certificate_stats(certificate)
        
        self.logger.info(f"Certificate {certificate_id} generated with status: {certificate.status.value}")
        
        return certificate
    
    def _generate_certificate_id(self, claim: str, reasoning_trace: List[str]) -> str:
        """Generate unique certificate ID."""
        content = claim + "".join(reasoning_trace) + str(datetime.now().timestamp())
        hash_obj = hashlib.sha256(content.encode())
        return f"cert_{hash_obj.hexdigest()[:16]}"
    
    def _extract_final_conclusion(self, reasoning_trace: List[str]) -> str:
        """Extract the final conclusion from reasoning trace."""
        if not reasoning_trace:
            return "No conclusion provided"
        
        # Look for conclusion indicators in the last few steps
        conclusion_indicators = ["therefore", "thus", "hence", "in conclusion", "finally"]
        
        for step in reversed(reasoning_trace[-3:]):  # Check last 3 steps
            step_lower = step.lower()
            for indicator in conclusion_indicators:
                if indicator in step_lower:
                    # Extract text after the indicator
                    parts = step.split(indicator, 1)
                    if len(parts) > 1:
                        return parts[1].strip()
        
        # If no explicit conclusion indicator, use the last step
        return reasoning_trace[-1].strip()
    
    def _calculate_certificate_metrics(self, certificate: ReasoningCertificate) -> CertificateMetrics:
        """Calculate comprehensive metrics for the certificate."""
        metrics = CertificateMetrics()
        
        # Extract metrics from proof certificate
        if certificate.proof_certificate:
            proof_cert = certificate.proof_certificate
            metrics.proof_confidence = proof_cert.overall_confidence
            
        # Extract metrics from verification report
        if certificate.verification_report:
            report = certificate.verification_report
            metrics.verification_score = report.overall_score
            metrics.logical_consistency = report.logical_consistency_score
            metrics.mathematical_correctness = report.mathematical_correctness_score
            metrics.semantic_coherence = report.semantic_coherence_score
            metrics.completeness = report.completeness_score
            metrics.soundness = report.soundness_score
        
        # Calculate overall trustworthiness
        metrics.overall_trustworthiness = self._calculate_trustworthiness(metrics)
        
        return metrics
    
    def _calculate_trustworthiness(self, metrics: CertificateMetrics) -> float:
        """Calculate overall trustworthiness score."""
        # Weighted combination of all metrics
        weights = {
            "proof_confidence": 0.2,
            "verification_score": 0.25,
            "logical_consistency": 0.2,
            "mathematical_correctness": 0.15,
            "semantic_coherence": 0.1,
            "completeness": 0.05,
            "soundness": 0.05
        }
        
        trustworthiness = (
            metrics.proof_confidence * weights["proof_confidence"] +
            metrics.verification_score * weights["verification_score"] +
            metrics.logical_consistency * weights["logical_consistency"] +
            metrics.mathematical_correctness * weights["mathematical_correctness"] +
            metrics.semantic_coherence * weights["semantic_coherence"] +
            metrics.completeness * weights["completeness"] +
            metrics.soundness * weights["soundness"]
        )
        
        return min(trustworthiness, 1.0)
    
    def _apply_custom_constraints(self, certificate: ReasoningCertificate, constraints: Dict[str, Any]) -> None:
        """Apply custom verification constraints."""
        violations = []
        
        # Apply minimum confidence threshold
        if "min_confidence" in constraints:
            min_confidence = constraints["min_confidence"]
            if certificate.metrics.overall_trustworthiness < min_confidence:
                violations.append(f"Trustworthiness {certificate.metrics.overall_trustworthiness:.2f} below minimum {min_confidence}")
        
        # Apply maximum verification time
        if "max_verification_time" in constraints:
            max_time = constraints["max_verification_time"]
            if certificate.generation_time > max_time:
                violations.append(f"Generation time {certificate.generation_time:.2f}s exceeds maximum {max_time}s")
        
        # Apply required proof types
        if "required_proof_types" in constraints:
            required_types = constraints["required_proof_types"]
            if certificate.proof_certificate:
                proof_type = certificate.proof_certificate.proof_type.value
                if proof_type not in required_types:
                    violations.append(f"Proof type {proof_type} not in required types: {required_types}")
        
        # Apply custom verification rules
        if "custom_rules" in constraints:
            custom_rules = constraints["custom_rules"]
            for rule_name, rule_func in custom_rules.items():
                try:
                    if not rule_func(certificate):
                        violations.append(f"Custom rule '{rule_name}' failed")
                except Exception as e:
                    violations.append(f"Custom rule '{rule_name}' error: {str(e)}")
        
        certificate.constraint_violations.extend(violations)
    
    def _determine_certificate_status(self, certificate: ReasoningCertificate, settings: Dict[str, Any]) -> CertificateStatus:
        """Determine the final status of the certificate."""
        # Check if there are any critical violations
        if certificate.constraint_violations:
            critical_violations = [v for v in certificate.constraint_violations if "error" in v.lower()]
            if critical_violations:
                return CertificateStatus.INVALID
        
        # Check minimum confidence threshold
        min_threshold = settings["min_confidence_threshold"]
        if certificate.metrics.overall_trustworthiness < min_threshold:
            return CertificateStatus.INVALID
        
        # Check verification result
        if certificate.verification_report:
            if certificate.verification_report.result == VerificationResult.FAILED:
                return CertificateStatus.INVALID
            elif certificate.verification_report.result == VerificationResult.UNKNOWN:
                return CertificateStatus.PENDING
        
        # Check if certificate has expired
        if certificate.expires_at and datetime.now() > certificate.expires_at:
            return CertificateStatus.EXPIRED
        
        return CertificateStatus.VALID
    
    def _generate_attestations(self, certificate: ReasoningCertificate) -> List[str]:
        """Generate attestations for the certificate."""
        attestations = []
        
        # Basic attestations
        attestations.append(f"Reasoning trace contains {len(certificate.reasoning_trace)} steps")
        attestations.append(f"Certificate level: {certificate.certificate_level.value}")
        
        # Proof attestations
        if certificate.proof_certificate:
            proof = certificate.proof_certificate
            attestations.append(f"Formal proof generated with {len(proof.steps)} proof steps")
            attestations.append(f"Proof type: {proof.proof_type.value}")
            attestations.append(f"Proof confidence: {proof.overall_confidence:.2f}")
        
        # Verification attestations
        if certificate.verification_report:
            report = certificate.verification_report
            attestations.append(f"Formal verification completed: {report.result.value}")
            attestations.append(f"Verification score: {report.overall_score:.2f}")
            
            if report.constraint_results:
                satisfied_count = sum(1 for v in report.constraint_results.values() if v)
                total_count = len(report.constraint_results)
                attestations.append(f"Constraints satisfied: {satisfied_count}/{total_count}")
        
        # Trustworthiness attestation
        trustworthiness = certificate.metrics.overall_trustworthiness
        if trustworthiness >= 0.9:
            attestations.append("High trustworthiness rating (≥90%)")
        elif trustworthiness >= 0.7:
            attestations.append("Good trustworthiness rating (≥70%)")
        elif trustworthiness >= 0.5:
            attestations.append("Acceptable trustworthiness rating (≥50%)")
        else:
            attestations.append("Low trustworthiness rating (<50%)")
        
        return attestations
    
    def _generate_certificate_recommendations(self, certificate: ReasoningCertificate) -> List[str]:
        """Generate recommendations for improving the certificate."""
        recommendations = []
        
        # Recommendations based on metrics
        metrics = certificate.metrics
        
        if metrics.logical_consistency < 0.8:
            recommendations.append("Improve logical consistency in reasoning steps")
        
        if metrics.mathematical_correctness < 0.9:
            recommendations.append("Verify all mathematical calculations")
        
        if metrics.semantic_coherence < 0.7:
            recommendations.append("Enhance semantic clarity and coherence")
        
        if metrics.completeness < 0.8:
            recommendations.append("Strengthen the connection between premises and conclusion")
        
        # Recommendations from verification report
        if certificate.verification_report and certificate.verification_report.recommendations:
            recommendations.extend(certificate.verification_report.recommendations)
        
        # Level-specific recommendations
        if certificate.certificate_level == CertificateLevel.CRITICAL:
            if metrics.overall_trustworthiness < 0.95:
                recommendations.append("For critical applications, consider additional verification steps")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_certificate_signature(self, certificate: ReasoningCertificate) -> str:
        """Generate cryptographic signature for the certificate (placeholder)."""
        # This is a placeholder for future cryptographic signature implementation
        # In a real system, this would use proper digital signatures
        
        content = f"{certificate.certificate_id}{certificate.original_claim}{certificate.final_conclusion}"
        hash_obj = hashlib.sha256(content.encode())
        return f"sig_{hash_obj.hexdigest()[:32]}"
    
    def _update_certificate_stats(self, certificate: ReasoningCertificate) -> None:
        """Update certificate generation statistics."""
        stats = self.certificate_stats
        
        stats["total_generated"] += 1
        
        # Update by level
        level = certificate.certificate_level.value
        stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
        
        # Update by status
        status = certificate.status.value
        stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
        
        # Update average trustworthiness
        total_trustworthiness = stats.get("total_trustworthiness", 0.0)
        total_trustworthiness += certificate.metrics.overall_trustworthiness
        stats["total_trustworthiness"] = total_trustworthiness
        stats["average_trustworthiness"] = total_trustworthiness / stats["total_generated"]
        
        # Update verification time
        stats["total_verification_time"] += certificate.generation_time
    
    def validate_certificate(self, certificate_id: str) -> bool:
        """Validate a certificate's current status."""
        certificate = self.certificates.get(certificate_id)
        if not certificate:
            return False
        
        # Check expiration
        if certificate.expires_at and datetime.now() > certificate.expires_at:
            certificate.status = CertificateStatus.EXPIRED
            return False
        
        # Check if revoked
        if certificate.status == CertificateStatus.REVOKED:
            return False
        
        # Check signature (placeholder)
        expected_signature = self._generate_certificate_signature(certificate)
        if certificate.signature != expected_signature:
            return False
        
        return certificate.status == CertificateStatus.VALID
    
    def revoke_certificate(self, certificate_id: str, reason: str = "") -> bool:
        """Revoke a certificate."""
        certificate = self.certificates.get(certificate_id)
        if not certificate:
            return False
        
        certificate.status = CertificateStatus.REVOKED
        certificate.metadata["revocation_reason"] = reason
        certificate.metadata["revoked_at"] = datetime.now().isoformat()
        
        self.logger.info(f"Certificate {certificate_id} revoked: {reason}")
        return True
    
    def get_certificate(self, certificate_id: str) -> Optional[ReasoningCertificate]:
        """Get certificate by ID."""
        return self.certificates.get(certificate_id)
    
    def get_certificate_statistics(self) -> Dict[str, Any]:
        """Get certificate generation statistics."""
        return self.certificate_stats.copy()
    
    def export_certificate(self, certificate_id: str, include_full_details: bool = True) -> Optional[Dict[str, Any]]:
        """Export certificate as JSON-serializable dictionary."""
        certificate = self.get_certificate(certificate_id)
        if not certificate:
            return None
        
        # Basic certificate info
        cert_data = {
            "certificate_id": certificate.certificate_id,
            "certificate_level": certificate.certificate_level.value,
            "status": certificate.status.value,
            "original_claim": certificate.original_claim,
            "final_conclusion": certificate.final_conclusion,
            "metrics": asdict(certificate.metrics),
            "created_at": certificate.created_at.isoformat(),
            "expires_at": certificate.expires_at.isoformat() if certificate.expires_at else None,
            "issuer": certificate.issuer,
            "version": certificate.version,
            "signature": certificate.signature,
            "generation_time": certificate.generation_time,
            "attestations": certificate.attestations,
            "recommendations": certificate.recommendations,
            "metadata": certificate.metadata
        }
        
        if include_full_details:
            # Include full reasoning trace
            cert_data["reasoning_trace"] = certificate.reasoning_trace
            cert_data["constraint_violations"] = certificate.constraint_violations
            
            # Include proof certificate if available
            if certificate.proof_certificate:
                cert_data["proof_certificate"] = self.proof_generator.export_proof_certificate(
                    certificate.proof_certificate.proof_id
                )
            
            # Include verification report if available
            if certificate.verification_report:
                cert_data["verification_report"] = self.formal_verifier.export_verification_report(
                    certificate.verification_report.verification_id
                )
        
        return cert_data
    
    def export_certificate_summary(self, certificate_id: str) -> Optional[Dict[str, Any]]:
        """Export a summary version of the certificate."""
        certificate = self.get_certificate(certificate_id)
        if not certificate:
            return None
        
        return {
            "certificate_id": certificate.certificate_id,
            "status": certificate.status.value,
            "level": certificate.certificate_level.value,
            "claim": certificate.original_claim[:100] + "..." if len(certificate.original_claim) > 100 else certificate.original_claim,
            "trustworthiness": certificate.metrics.overall_trustworthiness,
            "created_at": certificate.created_at.isoformat(),
            "valid_until": certificate.expires_at.isoformat() if certificate.expires_at else None,
            "attestations_count": len(certificate.attestations),
            "recommendations_count": len(certificate.recommendations)
        }
    
    def search_certificates(self, 
                          status: Optional[CertificateStatus] = None,
                          level: Optional[CertificateLevel] = None,
                          min_trustworthiness: Optional[float] = None,
                          created_after: Optional[datetime] = None) -> List[str]:
        """Search certificates by criteria."""
        matching_ids = []
        
        for cert_id, certificate in self.certificates.items():
            # Apply filters
            if status and certificate.status != status:
                continue
            
            if level and certificate.certificate_level != level:
                continue
            
            if min_trustworthiness and certificate.metrics.overall_trustworthiness < min_trustworthiness:
                continue
            
            if created_after and certificate.created_at < created_after:
                continue
            
            matching_ids.append(cert_id)
        
        return matching_ids
    
    def cleanup_expired_certificates(self) -> int:
        """Remove expired certificates and return count removed."""
        expired_ids = []
        current_time = datetime.now()
        
        for cert_id, certificate in self.certificates.items():
            if certificate.expires_at and current_time > certificate.expires_at:
                if certificate.status != CertificateStatus.EXPIRED:
                    certificate.status = CertificateStatus.EXPIRED
                
                # Remove certificates expired for more than 30 days
                if current_time > certificate.expires_at + timedelta(days=30):
                    expired_ids.append(cert_id)
        
        # Remove old expired certificates
        for cert_id in expired_ids:
            del self.certificates[cert_id]
        
        if expired_ids:
            self.logger.info(f"Cleaned up {len(expired_ids)} expired certificates")
        
        return len(expired_ids)