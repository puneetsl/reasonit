"""
Wikipedia Knowledge Detection System.

This module identifies when queries require encyclopedic knowledge and determines
what specific information needs to be retrieved from Wikipedia.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class KnowledgeType(Enum):
    """Types of knowledge that might be found on Wikipedia."""
    HISTORICAL_EVENTS = "historical_events"
    BIOGRAPHICAL = "biographical"
    GEOGRAPHIC = "geographic"
    SCIENTIFIC_CONCEPTS = "scientific_concepts"
    CULTURAL_TOPICS = "cultural_topics"
    STATISTICAL_DATA = "statistical_data"
    DEFINITIONS = "definitions"
    CURRENT_EVENTS = "current_events"
    TECHNICAL_TOPICS = "technical_topics"
    ARTISTIC_WORKS = "artistic_works"
    ORGANIZATIONS = "organizations"
    NONE = "none"


@dataclass
class KnowledgeNeed:
    """Represents a detected need for encyclopedic knowledge."""
    knowledge_type: KnowledgeType
    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    confidence: float = 0.0
    search_terms: List[str] = field(default_factory=list)
    reasoning: str = ""
    urgency: str = "medium"  # low, medium, high


@dataclass
class DetectionResult:
    """Result of knowledge detection analysis."""
    needs_wikipedia: bool
    primary_need: Optional[KnowledgeNeed]
    secondary_needs: List[KnowledgeNeed] = field(default_factory=list)
    overall_confidence: float = 0.0
    detection_reasoning: List[str] = field(default_factory=list)


class WikipediaKnowledgeDetector:
    """
    Detects when queries require Wikipedia knowledge and what to search for.
    
    Uses pattern matching, entity recognition, and semantic analysis to identify
    factual questions that would benefit from encyclopedic knowledge.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_detection_patterns()
        self._init_entity_patterns()
        self._init_knowledge_indicators()
    
    def _init_detection_patterns(self):
        """Initialize patterns for detecting different types of knowledge needs."""
        
        self.knowledge_patterns = {
            KnowledgeType.HISTORICAL_EVENTS: {
                "question_patterns": [
                    r"when\s+did\s+.+\s+(happen|occur|start|begin|end)",
                    r"what\s+year\s+.+\s+(war|revolution|battle|treaty)",
                    r"during\s+which\s+(period|era|century|decade)",
                    r"what\s+happened\s+(in|during|at)\s+\d{4}",
                    r"timeline\s+of\s+.+",
                    r"history\s+of\s+.+"
                ],
                "keywords": [
                    "war", "battle", "revolution", "empire", "dynasty", "era",
                    "century", "ancient", "medieval", "renaissance", "colonial",
                    "independence", "founding", "establishment", "conquest"
                ],
                "entities": ["dates", "historical_figures", "places", "events"]
            },
            
            KnowledgeType.BIOGRAPHICAL: {
                "question_patterns": [
                    r"who\s+(was|is)\s+.+",
                    r"biography\s+of\s+.+",
                    r"born\s+(in|on|at)\s+.+",
                    r"died\s+(in|on|at)\s+.+",
                    r"life\s+of\s+.+",
                    r"about\s+.+\s+(person|scientist|artist|leader)"
                ],
                "keywords": [
                    "born", "died", "biography", "life", "career", "achievements",
                    "famous", "inventor", "scientist", "artist", "politician",
                    "writer", "philosopher", "leader", "pioneer"
                ],
                "entities": ["people", "professions", "achievements"]
            },
            
            KnowledgeType.GEOGRAPHIC: {
                "question_patterns": [
                    r"where\s+is\s+.+\s+located",
                    r"capital\s+of\s+.+",
                    r"population\s+of\s+.+",
                    r"area\s+of\s+.+",
                    r"climate\s+of\s+.+",
                    r"geography\s+of\s+.+"
                ],
                "keywords": [
                    "capital", "population", "area", "location", "country",
                    "city", "continent", "ocean", "mountain", "river",
                    "climate", "geography", "border", "region", "territory"
                ],
                "entities": ["places", "countries", "cities", "landmarks"]
            },
            
            KnowledgeType.SCIENTIFIC_CONCEPTS: {
                "question_patterns": [
                    r"what\s+is\s+.+\s+(theory|law|principle|effect)",
                    r"how\s+does\s+.+\s+work",
                    r"explain\s+.+\s+(quantum|relativity|evolution|gravity)",
                    r"definition\s+of\s+.+",
                    r"scientific\s+explanation\s+of\s+.+"
                ],
                "keywords": [
                    "theory", "law", "principle", "quantum", "relativity",
                    "evolution", "gravity", "electromagnetic", "thermodynamics",
                    "chemistry", "biology", "physics", "molecular", "atomic",
                    "scientific", "research", "experiment", "hypothesis"
                ],
                "entities": ["scientific_terms", "theories", "phenomena"]
            },
            
            KnowledgeType.STATISTICAL_DATA: {
                "question_patterns": [
                    r"how\s+many\s+.+\s+(are\s+there|exist)",
                    r"percentage\s+of\s+.+",
                    r"statistics\s+(on|about)\s+.+",
                    r"data\s+(on|about)\s+.+",
                    r"rate\s+of\s+.+"
                ],
                "keywords": [
                    "statistics", "data", "percentage", "rate", "number",
                    "count", "frequency", "distribution", "average", "median",
                    "demographics", "survey", "census", "study"
                ],
                "entities": ["numbers", "measurements", "statistics"]
            },
            
            KnowledgeType.DEFINITIONS: {
                "question_patterns": [
                    r"what\s+(is|does)\s+.+\s+mean",
                    r"define\s+.+",
                    r"definition\s+of\s+.+",
                    r"meaning\s+of\s+.+",
                    r"what\s+is\s+(a|an)\s+.+"
                ],
                "keywords": [
                    "definition", "meaning", "term", "concept", "idea",
                    "terminology", "vocabulary", "glossary", "explain"
                ],
                "entities": ["terms", "concepts", "words"]
            },
            
            KnowledgeType.ORGANIZATIONS: {
                "question_patterns": [
                    r"what\s+is\s+.+\s+(organization|company|institution)",
                    r"founded\s+(in|by)\s+.+",
                    r"headquarters\s+of\s+.+",
                    r"mission\s+of\s+.+"
                ],
                "keywords": [
                    "organization", "company", "corporation", "institution",
                    "founded", "headquarters", "mission", "purpose", "structure",
                    "government", "agency", "department", "ministry"
                ],
                "entities": ["organizations", "companies", "institutions"]
            }
        }
    
    def _init_entity_patterns(self):
        """Initialize patterns for recognizing different types of entities."""
        
        self.entity_patterns = {
            "people": [
                r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # First Last names
                r"\b(Mr|Ms|Dr|Prof|President|King|Queen)\s+[A-Z][a-z]+",
                r"\b[A-Z][a-z]+\s+(the|von|van|de)\s+[A-Z][a-z]+\b"
            ],
            "places": [
                r"\b[A-Z][a-z]+\s+(City|County|State|Province|Country)\b",
                r"\b(Mount|Lake|River|Ocean|Sea|Bay)\s+[A-Z][a-z]+\b",
                r"\b[A-Z][a-z]+\s+(Mountains|Hills|Valley|Desert|Forest)\b"
            ],
            "organizations": [
                r"\b[A-Z][A-Z]+\b",  # Acronyms like NASA, FBI
                r"\b[A-Z][a-z]+\s+(University|College|Institute|Foundation)\b",
                r"\b(United\s+Nations|World\s+Bank|European\s+Union)\b"
            ],
            "dates": [
                r"\b\d{4}\b",  # Years
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}\b",
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
            ],
            "scientific_terms": [
                r"\b[a-z]+\s+(theory|law|principle|effect|equation|model)\b",
                r"\b(quantum|nuclear|molecular|electromagnetic|thermodynamic)\s+[a-z]+\b"
            ]
        }
    
    def _init_knowledge_indicators(self):
        """Initialize indicators that suggest need for factual knowledge."""
        
        self.factual_indicators = {
            "high_confidence": [
                "when did", "where is", "who was", "what year", "how many",
                "capital of", "population of", "born in", "died in",
                "founded in", "invented by", "discovered by"
            ],
            "medium_confidence": [
                "about", "regarding", "concerning", "information on",
                "facts about", "details of", "background on"
            ],
            "question_words": [
                "what", "who", "where", "when", "why", "how", "which"
            ],
            "factual_domains": [
                "history", "geography", "science", "biology", "chemistry",
                "physics", "astronomy", "mathematics", "politics", "economics",
                "culture", "art", "literature", "music", "sports"
            ]
        }
    
    def detect_knowledge_needs(self, query: str) -> DetectionResult:
        """
        Detect if a query needs Wikipedia knowledge and what type.
        
        Args:
            query: The user's query to analyze
            
        Returns:
            DetectionResult with detected knowledge needs
        """
        query_lower = query.lower()
        
        # Detect all possible knowledge needs
        detected_needs = []
        
        for knowledge_type, patterns in self.knowledge_patterns.items():
            confidence = self._calculate_knowledge_confidence(query_lower, patterns)
            
            if confidence > 0.1:  # Threshold for detection
                entities = self._extract_entities(query, patterns.get("entities", []))
                search_terms = self._generate_search_terms(query, entities, knowledge_type)
                
                need = KnowledgeNeed(
                    knowledge_type=knowledge_type,
                    entities=entities,
                    topics=self._extract_topics(query, knowledge_type),
                    confidence=confidence,
                    search_terms=search_terms,
                    reasoning=self._explain_detection(knowledge_type, confidence),
                    urgency=self._assess_urgency(confidence, knowledge_type)
                )
                
                detected_needs.append(need)
        
        # Sort by confidence
        detected_needs.sort(key=lambda n: n.confidence, reverse=True)
        
        # Determine if Wikipedia is needed
        needs_wikipedia = len(detected_needs) > 0 and detected_needs[0].confidence > 0.3
        
        # Calculate overall confidence
        overall_confidence = detected_needs[0].confidence if detected_needs else 0.0
        
        # Generate detection reasoning
        detection_reasoning = self._generate_detection_reasoning(query, detected_needs)
        
        return DetectionResult(
            needs_wikipedia=needs_wikipedia,
            primary_need=detected_needs[0] if detected_needs else None,
            secondary_needs=detected_needs[1:3] if len(detected_needs) > 1 else [],
            overall_confidence=overall_confidence,
            detection_reasoning=detection_reasoning
        )
    
    def _calculate_knowledge_confidence(self, query: str, patterns: Dict) -> float:
        """Calculate confidence score for a knowledge type."""
        score = 0.0
        
        # Check question patterns
        question_patterns = patterns.get("question_patterns", [])
        for pattern in question_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.6
                break  # Only count one pattern match
        
        # Check keywords
        keywords = patterns.get("keywords", [])
        keyword_matches = sum(1 for keyword in keywords if keyword in query)
        if keywords:
            keyword_score = min(keyword_matches / len(keywords), 1.0) * 0.4
            score += keyword_score
        
        return min(score, 1.0)
    
    def _extract_entities(self, query: str, entity_types: List[str]) -> List[str]:
        """Extract entities from the query based on entity types."""
        entities = []
        
        for entity_type in entity_types:
            if entity_type in self.entity_patterns:
                patterns = self.entity_patterns[entity_type]
                for pattern in patterns:
                    matches = re.findall(pattern, query)
                    entities.extend(matches)
        
        # Remove duplicates and clean up
        unique_entities = []
        for entity in entities:
            if isinstance(entity, tuple):
                entity = " ".join(entity)
            entity = entity.strip()
            if entity and entity not in unique_entities:
                unique_entities.append(entity)
        
        return unique_entities[:5]  # Limit to top 5 entities
    
    def _extract_topics(self, query: str, knowledge_type: KnowledgeType) -> List[str]:
        """Extract relevant topics from the query."""
        topics = []
        
        # Extract noun phrases as potential topics
        words = query.split()
        
        # Simple noun phrase extraction (can be enhanced with NLP libraries)
        for i, word in enumerate(words):
            if word.lower() in ["the", "a", "an"] and i + 1 < len(words):
                potential_topic = words[i + 1]
                if len(potential_topic) > 3 and potential_topic.isalpha():
                    topics.append(potential_topic)
        
        # Add knowledge-type specific topics
        if knowledge_type == KnowledgeType.SCIENTIFIC_CONCEPTS:
            science_words = ["quantum", "relativity", "evolution", "DNA", "physics", "chemistry"]
            topics.extend([word for word in science_words if word in query.lower()])
        
        return topics[:3]  # Limit to top 3 topics
    
    def _generate_search_terms(self, query: str, entities: List[str], knowledge_type: KnowledgeType) -> List[str]:
        """Generate appropriate search terms for Wikipedia."""
        search_terms = []
        
        # Add entities as primary search terms
        search_terms.extend(entities)
        
        # Add knowledge-type specific terms
        if knowledge_type == KnowledgeType.BIOGRAPHICAL:
            # For biographical queries, focus on person names
            person_indicators = ["who was", "who is", "biography", "life of"]
            for indicator in person_indicators:
                if indicator in query.lower():
                    # Try to extract the person name after the indicator
                    parts = query.lower().split(indicator)
                    if len(parts) > 1:
                        potential_name = parts[1].strip().split()[0:2]  # First two words
                        if potential_name:
                            search_terms.append(" ".join(potential_name))
        
        elif knowledge_type == KnowledgeType.GEOGRAPHIC:
            # For geographic queries, add location-specific terms
            geo_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
            search_terms.extend(geo_words[:3])
        
        elif knowledge_type == KnowledgeType.HISTORICAL_EVENTS:
            # For historical queries, extract event names and dates
            years = re.findall(r'\b\d{4}\b', query)
            search_terms.extend(years)
            
            # Extract potential event names
            event_keywords = ["war", "battle", "revolution", "treaty", "empire"]
            for keyword in event_keywords:
                if keyword in query.lower():
                    # Look for capitalized words near the keyword
                    pattern = rf'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*{keyword}|\b{keyword}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'
                    matches = re.findall(pattern, query, re.IGNORECASE)
                    search_terms.extend(matches)
        
        # Remove duplicates and clean up
        clean_terms = []
        for term in search_terms:
            term = term.strip()
            if term and len(term) > 1 and term not in clean_terms:
                clean_terms.append(term)
        
        return clean_terms[:5]  # Limit to top 5 search terms
    
    def _explain_detection(self, knowledge_type: KnowledgeType, confidence: float) -> str:
        """Explain why this knowledge type was detected."""
        explanations = {
            KnowledgeType.HISTORICAL_EVENTS: "Query contains historical indicators and references to events, dates, or periods",
            KnowledgeType.BIOGRAPHICAL: "Query asks about a person's life, achievements, or biographical information",
            KnowledgeType.GEOGRAPHIC: "Query requests location-based information about places, regions, or geographic features",
            KnowledgeType.SCIENTIFIC_CONCEPTS: "Query involves scientific terms, theories, or requests explanations of natural phenomena",
            KnowledgeType.STATISTICAL_DATA: "Query asks for numerical data, statistics, or quantitative information",
            KnowledgeType.DEFINITIONS: "Query seeks definitions, explanations, or clarification of terms",
            KnowledgeType.ORGANIZATIONS: "Query asks about institutions, companies, or organized entities"
        }
        
        base_explanation = explanations.get(knowledge_type, "General factual information detected")
        confidence_qualifier = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        
        return f"{base_explanation} (confidence: {confidence_qualifier})"
    
    def _assess_urgency(self, confidence: float, knowledge_type: KnowledgeType) -> str:
        """Assess the urgency of retrieving this knowledge."""
        if confidence > 0.8:
            return "high"
        elif confidence > 0.5:
            return "medium"
        else:
            return "low"
    
    def _generate_detection_reasoning(self, query: str, needs: List[KnowledgeNeed]) -> List[str]:
        """Generate human-readable reasoning for the detection results."""
        reasoning = []
        
        if not needs:
            reasoning.append("No clear encyclopedic knowledge requirements detected")
            reasoning.append("Query appears to be answerable without external factual lookup")
            return reasoning
        
        primary_need = needs[0]
        reasoning.append(f"Primary knowledge need: {primary_need.knowledge_type.value.replace('_', ' ')}")
        reasoning.append(f"Detection confidence: {primary_need.confidence:.1%}")
        
        if primary_need.entities:
            reasoning.append(f"Key entities identified: {', '.join(primary_need.entities[:3])}")
        
        if primary_need.search_terms:
            reasoning.append(f"Suggested search terms: {', '.join(primary_need.search_terms[:3])}")
        
        if len(needs) > 1:
            secondary_types = [n.knowledge_type.value.replace('_', ' ') for n in needs[1:3]]
            reasoning.append(f"Secondary knowledge types: {', '.join(secondary_types)}")
        
        # Add specific recommendations
        if primary_need.knowledge_type == KnowledgeType.BIOGRAPHICAL:
            reasoning.append("Recommendation: Search for biographical information and life details")
        elif primary_need.knowledge_type == KnowledgeType.HISTORICAL_EVENTS:
            reasoning.append("Recommendation: Look up historical context, dates, and related events")
        elif primary_need.knowledge_type == KnowledgeType.SCIENTIFIC_CONCEPTS:
            reasoning.append("Recommendation: Retrieve scientific explanations and theoretical background")
        
        return reasoning
    
    def suggest_search_strategy(self, detection_result: DetectionResult) -> Dict[str, Any]:
        """Suggest a search strategy based on detection results."""
        if not detection_result.needs_wikipedia or not detection_result.primary_need:
            return {"strategy": "none", "reason": "No Wikipedia search needed"}
        
        primary_need = detection_result.primary_need
        
        strategy = {
            "strategy": "multi_search",
            "primary_searches": primary_need.search_terms[:3],
            "knowledge_type": primary_need.knowledge_type.value,
            "urgency": primary_need.urgency,
            "expected_content": self._get_expected_content(primary_need.knowledge_type),
            "fallback_searches": []
        }
        
        # Add fallback searches from secondary needs
        for secondary_need in detection_result.secondary_needs:
            if secondary_need.search_terms:
                strategy["fallback_searches"].extend(secondary_need.search_terms[:2])
        
        # Add strategy-specific recommendations
        if primary_need.knowledge_type == KnowledgeType.BIOGRAPHICAL:
            strategy["sections_to_focus"] = ["Early life", "Career", "Personal life", "Legacy"]
        elif primary_need.knowledge_type == KnowledgeType.HISTORICAL_EVENTS:
            strategy["sections_to_focus"] = ["Background", "Course of events", "Aftermath", "Significance"]
        elif primary_need.knowledge_type == KnowledgeType.SCIENTIFIC_CONCEPTS:
            strategy["sections_to_focus"] = ["Definition", "Theory", "Applications", "Examples"]
        else:
            strategy["sections_to_focus"] = ["Introduction", "Overview", "Details"]
        
        return strategy
    
    def _get_expected_content(self, knowledge_type: KnowledgeType) -> List[str]:
        """Get expected content types for a knowledge type."""
        content_map = {
            KnowledgeType.BIOGRAPHICAL: ["birth/death dates", "achievements", "career highlights", "personal background"],
            KnowledgeType.HISTORICAL_EVENTS: ["dates", "causes", "key figures", "consequences", "timeline"],
            KnowledgeType.GEOGRAPHIC: ["location", "population", "area", "climate", "notable features"],
            KnowledgeType.SCIENTIFIC_CONCEPTS: ["definition", "principles", "applications", "examples", "related theories"],
            KnowledgeType.STATISTICAL_DATA: ["numbers", "percentages", "trends", "comparisons", "sources"],
            KnowledgeType.DEFINITIONS: ["meaning", "etymology", "usage", "examples", "related terms"],
            KnowledgeType.ORGANIZATIONS: ["purpose", "structure", "history", "leadership", "activities"]
        }
        
        return content_map.get(knowledge_type, ["general information", "overview", "key facts"])
    
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze the complexity of knowledge requirements in a query."""
        detection_result = self.detect_knowledge_needs(query)
        
        complexity_factors = {
            "entity_count": len(detection_result.primary_need.entities) if detection_result.primary_need else 0,
            "knowledge_types": len([detection_result.primary_need] + detection_result.secondary_needs),
            "query_length": len(query.split()),
            "specificity": "high" if detection_result.overall_confidence > 0.7 else "medium" if detection_result.overall_confidence > 0.4 else "low"
        }
        
        # Determine overall complexity
        if complexity_factors["knowledge_types"] > 2 or complexity_factors["entity_count"] > 3:
            overall_complexity = "high"
        elif complexity_factors["knowledge_types"] > 1 or complexity_factors["entity_count"] > 1:
            overall_complexity = "medium"
        else:
            overall_complexity = "low"
        
        return {
            "overall_complexity": overall_complexity,
            "factors": complexity_factors,
            "recommendation": self._get_complexity_recommendation(overall_complexity)
        }
    
    def _get_complexity_recommendation(self, complexity: str) -> str:
        """Get recommendation based on query complexity."""
        recommendations = {
            "high": "Use multiple Wikipedia searches and cross-reference information from different articles",
            "medium": "Search for 2-3 related articles and synthesize key information",
            "low": "Single focused Wikipedia search should provide sufficient information"
        }
        
        return recommendations.get(complexity, "Use standard search approach")