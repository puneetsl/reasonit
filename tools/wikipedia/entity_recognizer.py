"""
Entity Recognition for Wikipedia Search Optimization.

This module provides entity recognition capabilities to improve Wikipedia search
accuracy by identifying people, places, organizations, dates, and concepts in queries.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class EntityType(Enum):
    """Types of entities that can be recognized."""
    PERSON = "person"
    PLACE = "place"
    ORGANIZATION = "organization"
    DATE = "date"
    EVENT = "event"
    CONCEPT = "concept"
    WORK = "work"  # Books, movies, songs, etc.
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """Represents a recognized entity."""
    text: str
    entity_type: EntityType
    confidence: float
    start_pos: int
    end_pos: int
    context: str = ""
    variants: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityRecognitionResult:
    """Result of entity recognition analysis."""
    entities: List[Entity] = field(default_factory=list)
    entity_count_by_type: Dict[EntityType, int] = field(default_factory=dict)
    confidence_score: float = 0.0
    processed_query: str = ""


class EntityRecognizer:
    """
    Recognizes and extracts entities from text for improved Wikipedia searching.
    
    Uses pattern matching, contextual analysis, and heuristics to identify
    entities that are likely to have Wikipedia articles.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_patterns()
        self._init_context_indicators()
        self._init_known_entities()
    
    def _init_patterns(self):
        """Initialize regex patterns for entity recognition."""
        
        self.entity_patterns = {
            EntityType.PERSON: [
                # Full names with titles
                r'\b(?:Dr\.?|Prof\.?|President|King|Queen|Emperor|Mr\.?|Ms\.?|Mrs\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
                # Regular full names (First Last, First Middle Last)
                r'\b([A-Z][a-z]+\s+(?:[A-Z][a-z]+\s+)?[A-Z][a-z]+)\b',
                # Names with suffixes
                r'\b([A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:Jr\.?|Sr\.?|III?|IV?))\b',
                # Historical names with epithets
                r'\b([A-Z][a-z]+\s+the\s+[A-Z][a-z]+)\b',
                # Single names (for famous people)
                r'\b(Aristotle|Plato|Shakespeare|Einstein|Napoleon|Cleopatra|Buddha|Confucius)\b'
            ],
            
            EntityType.PLACE: [
                # Countries
                r'\b(United States|United Kingdom|United Arab Emirates|South Africa|New Zealand|Saudi Arabia|Costa Rica)\b',
                # Cities with state/country
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                # Geographic features
                r'\b(Mount|Lake|River|Bay|Gulf|Cape|Sea|Ocean)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                # Regions and landmarks
                r'\b([A-Z][a-z]+\s+(?:Mountains|Hills|Valley|Desert|Forest|Peninsula|Island|Strait))\b',
                # Common place patterns
                r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                r'\bof\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
            ],
            
            EntityType.ORGANIZATION: [
                # Universities and educational institutions
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:University|College|Institute|Academy|School))\b',
                # Companies with Inc, Corp, etc.
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc\.?|Corp\.?|Company|Corporation|Ltd\.?))\b',
                # Government organizations
                r'\b(Department\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                r'\b([A-Z][a-z]+\s+(?:Agency|Bureau|Commission|Administration|Service))\b',
                # International organizations
                r'\b(United Nations|World Bank|European Union|World Health Organization|International Monetary Fund)\b',
                # Acronyms (likely organizations)
                r'\b([A-Z]{2,6})\b',
                # Foundations and nonprofits
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Foundation|Trust|Society|Association))\b'
            ],
            
            EntityType.DATE: [
                # Full dates
                r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b',
                r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
                # Years
                r'\b(\d{4})\b',
                # Date ranges
                r'\b(\d{4}\s*-\s*\d{4})\b',
                # Relative dates
                r'\b((?:early|mid|late)\s+\d{4}s?)\b',
                # Centuries
                r'\b(\d{1,2}(?:st|nd|rd|th)\s+century)\b'
            ],
            
            EntityType.EVENT: [
                # Wars and conflicts
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:War|Battle|Conflict|Revolution|Uprising))\b',
                # Historical events
                r'\b((?:Great|Cold|World)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                # Treaties and agreements
                r'\b(Treaty\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                # Disasters and crises
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Crisis|Disaster|Earthquake|Hurricane|Tsunami))\b'
            ],
            
            EntityType.CONCEPT: [
                # Scientific theories and laws
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:theory|law|principle|effect|equation|model))\b',
                # Philosophical concepts
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:philosophy|doctrine|ideology|movement))\b',
                # Academic disciplines
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:studies|science|engineering|mathematics))\b'
            ],
            
            EntityType.WORK: [
                # Books and literature
                r'\"([^\"]+)\"',  # Quoted titles
                r'\b(The\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # "The" titles
                # Movies and shows
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:movie|film|series|show))\b'
            ]
        }
    
    def _init_context_indicators(self):
        """Initialize context indicators that help identify entity types."""
        
        self.context_indicators = {
            EntityType.PERSON: {
                "before": ["by", "wrote", "invented", "discovered", "said", "born", "died", "created", "founded"],
                "after": ["was", "is", "said", "wrote", "invented", "discovered", "born", "died", "lived"]
            },
            EntityType.PLACE: {
                "before": ["in", "at", "from", "to", "near", "located", "situated"],
                "after": ["is", "was", "located", "situated", "lies", "stands"]
            },
            EntityType.ORGANIZATION: {
                "before": ["at", "with", "by", "from", "founded"],
                "after": ["announced", "reported", "founded", "established", "created"]
            },
            EntityType.EVENT: {
                "before": ["during", "after", "before", "caused", "started", "ended"],
                "after": ["occurred", "happened", "began", "started", "ended", "lasted"]
            },
            EntityType.CONCEPT: {
                "before": ["about", "of", "explains", "describes", "studies"],
                "after": ["is", "describes", "explains", "states", "proposes"]
            }
        }
    
    def _init_known_entities(self):
        """Initialize lists of commonly known entities."""
        
        self.known_entities = {
            EntityType.PERSON: {
                "scientists": ["Einstein", "Newton", "Darwin", "Curie", "Hawking", "Tesla", "Edison"],
                "historical": ["Napoleon", "Caesar", "Cleopatra", "Gandhi", "Churchill", "Roosevelt"],
                "artists": ["Picasso", "Mozart", "Shakespeare", "Beethoven", "Leonardo da Vinci"]
            },
            EntityType.PLACE: {
                "countries": ["France", "Germany", "Japan", "Brazil", "Australia", "Canada", "India"],
                "cities": ["Paris", "London", "Tokyo", "New York", "Los Angeles", "Berlin", "Rome"],
                "landmarks": ["Eiffel Tower", "Great Wall", "Statue of Liberty", "Big Ben"]
            },
            EntityType.ORGANIZATION: {
                "tech": ["Google", "Microsoft", "Apple", "Facebook", "Amazon", "IBM"],
                "international": ["UNESCO", "WHO", "NATO", "UN", "EU", "IMF"]
            }
        }
    
    def recognize_entities(self, text: str) -> EntityRecognitionResult:
        """
        Recognize entities in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            EntityRecognitionResult with detected entities
        """
        self.logger.debug(f"Recognizing entities in: {text[:100]}...")
        
        entities = []
        
        # Apply pattern-based recognition for each entity type
        for entity_type, patterns in self.entity_patterns.items():
            type_entities = self._extract_entities_by_type(text, entity_type, patterns)
            entities.extend(type_entities)
        
        # Apply context-based refinement
        refined_entities = self._refine_entities_with_context(text, entities)
        
        # Remove duplicates and conflicts
        final_entities = self._resolve_entity_conflicts(refined_entities)
        
        # Calculate entity counts by type
        entity_counts = {}
        for entity_type in EntityType:
            count = sum(1 for e in final_entities if e.entity_type == entity_type)
            if count > 0:
                entity_counts[entity_type] = count
        
        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(final_entities)
        
        # Create processed query (can be used for improved search)
        processed_query = self._create_processed_query(text, final_entities)
        
        return EntityRecognitionResult(
            entities=final_entities,
            entity_count_by_type=entity_counts,
            confidence_score=confidence_score,
            processed_query=processed_query
        )
    
    def _extract_entities_by_type(self, text: str, entity_type: EntityType, patterns: List[str]) -> List[Entity]:
        """Extract entities of a specific type using patterns."""
        entities = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                entity_text = match.group(1) if match.groups() else match.group(0)
                entity_text = entity_text.strip()
                
                if len(entity_text) > 1 and self._is_valid_entity(entity_text, entity_type):
                    # Calculate confidence based on pattern specificity and context
                    confidence = self._calculate_entity_confidence(entity_text, entity_type, text, match.start(), match.end())
                    
                    # Extract context
                    context = self._extract_entity_context(text, match.start(), match.end())
                    
                    entity = Entity(
                        text=entity_text,
                        entity_type=entity_type,
                        confidence=confidence,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        context=context,
                        variants=self._generate_entity_variants(entity_text),
                        metadata={}
                    )
                    
                    entities.append(entity)
        
        return entities
    
    def _is_valid_entity(self, text: str, entity_type: EntityType) -> bool:
        """Check if extracted text is a valid entity."""
        # Skip common words that might be incorrectly matched
        common_words = {
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "this", "that", "these", "those", "he", "she", "it", "they", "we", "you"
        }
        
        if text.lower() in common_words:
            return False
        
        # Entity-specific validation
        if entity_type == EntityType.PERSON:
            # Must have at least one uppercase letter
            if not any(c.isupper() for c in text):
                return False
            # Should not be all uppercase (likely acronym)
            if text.isupper() and len(text) > 3:
                return False
        
        elif entity_type == EntityType.PLACE:
            # Should start with uppercase
            if not text[0].isupper():
                return False
        
        elif entity_type == EntityType.DATE:
            # Basic date validation
            if entity_type == EntityType.DATE and re.match(r'^\d{4}$', text):
                year = int(text)
                if year < 1 or year > 2100:  # Reasonable year range
                    return False
        
        return True
    
    def _calculate_entity_confidence(self, entity_text: str, entity_type: EntityType, full_text: str, start_pos: int, end_pos: int) -> float:
        """Calculate confidence score for an entity."""
        confidence = 0.5  # Base confidence
        
        # Boost for known entities
        if self._is_known_entity(entity_text, entity_type):
            confidence += 0.3
        
        # Boost for context indicators
        context_boost = self._get_context_confidence_boost(full_text, start_pos, end_pos, entity_type)
        confidence += context_boost
        
        # Boost for proper capitalization
        if entity_text[0].isupper():
            confidence += 0.1
        
        # Penalty for very short entities (unless they're known)
        if len(entity_text) < 3 and not self._is_known_entity(entity_text, entity_type):
            confidence -= 0.2
        
        # Boost for entities with multiple words (often more specific)
        if len(entity_text.split()) > 1:
            confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _is_known_entity(self, text: str, entity_type: EntityType) -> bool:
        """Check if entity is in known entities list."""
        if entity_type not in self.known_entities:
            return False
        
        for category, entities in self.known_entities[entity_type].items():
            if text in entities or text.lower() in [e.lower() for e in entities]:
                return True
        
        return False
    
    def _get_context_confidence_boost(self, text: str, start_pos: int, end_pos: int, entity_type: EntityType) -> float:
        """Get confidence boost based on surrounding context."""
        if entity_type not in self.context_indicators:
            return 0.0
        
        indicators = self.context_indicators[entity_type]
        boost = 0.0
        
        # Check words before the entity
        words_before = text[:start_pos].lower().split()[-3:]  # Last 3 words before
        for word in words_before:
            if word in indicators.get("before", []):
                boost += 0.15
                break
        
        # Check words after the entity
        words_after = text[end_pos:].lower().split()[:3]  # First 3 words after
        for word in words_after:
            if word in indicators.get("after", []):
                boost += 0.15
                break
        
        return boost
    
    def _extract_entity_context(self, text: str, start_pos: int, end_pos: int, context_window: int = 50) -> str:
        """Extract context around an entity."""
        context_start = max(0, start_pos - context_window)
        context_end = min(len(text), end_pos + context_window)
        
        return text[context_start:context_end].strip()
    
    def _generate_entity_variants(self, entity_text: str) -> List[str]:
        """Generate alternative forms of an entity for search."""
        variants = []
        
        # Remove common prefixes/suffixes
        cleaned = entity_text
        for prefix in ["Dr.", "Prof.", "President", "King", "Queen", "Mr.", "Ms.", "Mrs."]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                variants.append(cleaned)
        
        # Add acronym if multiple words
        words = entity_text.split()
        if len(words) > 1:
            acronym = ''.join(word[0].upper() for word in words if word[0].isupper())
            if len(acronym) > 1:
                variants.append(acronym)
        
        # Add without "The" prefix
        if entity_text.startswith("The "):
            variants.append(entity_text[4:])
        
        return variants
    
    def _refine_entities_with_context(self, text: str, entities: List[Entity]) -> List[Entity]:
        """Refine entity classifications using context analysis."""
        refined_entities = []
        
        for entity in entities:
            # Reassess entity type based on broader context
            refined_type = self._reassess_entity_type(entity, text)
            
            if refined_type != entity.entity_type:
                # Update entity with refined type
                entity.entity_type = refined_type
                entity.confidence *= 0.9  # Slight penalty for type change
            
            # Additional validation
            if entity.confidence > 0.2:  # Keep entities with reasonable confidence
                refined_entities.append(entity)
        
        return refined_entities
    
    def _reassess_entity_type(self, entity: Entity, full_text: str) -> EntityType:
        """Reassess entity type based on broader context."""
        # Check for question patterns that might indicate entity type
        
        # Biographical questions
        if any(pattern in full_text.lower() for pattern in ["who was", "who is", "biography", "life of"]):
            if entity.entity_type in [EntityType.UNKNOWN, EntityType.CONCEPT]:
                return EntityType.PERSON
        
        # Geographic questions
        if any(pattern in full_text.lower() for pattern in ["where is", "capital of", "located in"]):
            if entity.entity_type in [EntityType.UNKNOWN, EntityType.CONCEPT]:
                return EntityType.PLACE
        
        # Historical questions
        if any(pattern in full_text.lower() for pattern in ["when did", "what year", "history of"]):
            if entity.entity_type == EntityType.UNKNOWN:
                # Could be event or person
                if any(word in entity.text.lower() for word in ["war", "battle", "revolution", "treaty"]):
                    return EntityType.EVENT
                else:
                    return EntityType.PERSON
        
        return entity.entity_type
    
    def _resolve_entity_conflicts(self, entities: List[Entity]) -> List[Entity]:
        """Resolve overlapping entities and conflicts."""
        # Sort by position
        sorted_entities = sorted(entities, key=lambda e: e.start_pos)
        
        final_entities = []
        
        for entity in sorted_entities:
            # Check for overlaps with already accepted entities
            overlaps = False
            
            for accepted in final_entities:
                if (entity.start_pos < accepted.end_pos and entity.end_pos > accepted.start_pos):
                    # Overlap detected - keep the one with higher confidence
                    if entity.confidence > accepted.confidence:
                        final_entities.remove(accepted)
                    else:
                        overlaps = True
                    break
            
            if not overlaps:
                final_entities.append(entity)
        
        # Remove very low confidence entities if we have many entities
        if len(final_entities) > 10:
            final_entities = [e for e in final_entities if e.confidence > 0.4]
        
        # Sort by confidence
        final_entities.sort(key=lambda e: e.confidence, reverse=True)
        
        return final_entities[:10]  # Limit to top 10 entities
    
    def _calculate_overall_confidence(self, entities: List[Entity]) -> float:
        """Calculate overall confidence in entity recognition."""
        if not entities:
            return 0.0
        
        # Weighted average based on entity confidence
        total_confidence = sum(e.confidence for e in entities)
        avg_confidence = total_confidence / len(entities)
        
        # Boost for having multiple high-confidence entities
        high_conf_entities = sum(1 for e in entities if e.confidence > 0.7)
        if high_conf_entities > 1:
            avg_confidence += 0.1
        
        # Penalty for having too many low-confidence entities
        low_conf_entities = sum(1 for e in entities if e.confidence < 0.3)
        if low_conf_entities > len(entities) / 2:
            avg_confidence -= 0.1
        
        return min(max(avg_confidence, 0.0), 1.0)
    
    def _create_processed_query(self, original_query: str, entities: List[Entity]) -> str:
        """Create a processed version of the query optimized for search."""
        # Start with original query
        processed = original_query
        
        # Highlight high-confidence entities (could be used for weighted search)
        high_conf_entities = [e for e in entities if e.confidence > 0.7]
        
        if high_conf_entities:
            # Could add emphasis markers or reorder terms
            # For now, just return original query
            pass
        
        return processed
    
    def get_entities_by_type(self, result: EntityRecognitionResult, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type from recognition result."""
        return [e for e in result.entities if e.entity_type == entity_type]
    
    def get_most_confident_entity(self, result: EntityRecognitionResult) -> Optional[Entity]:
        """Get the entity with highest confidence."""
        if not result.entities:
            return None
        
        return max(result.entities, key=lambda e: e.confidence)
    
    def suggest_search_terms(self, result: EntityRecognitionResult) -> List[str]:
        """Suggest search terms based on recognized entities."""
        search_terms = []
        
        # Add high-confidence entities as search terms
        for entity in result.entities:
            if entity.confidence > 0.6:
                search_terms.append(entity.text)
                
                # Add variants for high-confidence entities
                if entity.confidence > 0.8:
                    search_terms.extend(entity.variants[:2])  # Top 2 variants
        
        # Remove duplicates while preserving order
        unique_terms = []
        for term in search_terms:
            if term not in unique_terms:
                unique_terms.append(term)
        
        return unique_terms[:8]  # Limit to 8 terms
    
    def analyze_query_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the structure of a query to understand search intent."""
        result = self.recognize_entities(text)
        
        analysis = {
            "entity_types_present": list(result.entity_count_by_type.keys()),
            "dominant_entity_type": None,
            "complexity": "low",
            "search_intent": "general"
        }
        
        # Determine dominant entity type
        if result.entity_count_by_type:
            dominant_type = max(result.entity_count_by_type, key=result.entity_count_by_type.get)
            analysis["dominant_entity_type"] = dominant_type
            
            # Infer search intent based on dominant type
            if dominant_type == EntityType.PERSON:
                analysis["search_intent"] = "biographical"
            elif dominant_type == EntityType.PLACE:
                analysis["search_intent"] = "geographic"
            elif dominant_type == EntityType.EVENT:
                analysis["search_intent"] = "historical"
            elif dominant_type == EntityType.CONCEPT:
                analysis["search_intent"] = "explanatory"
        
        # Determine complexity
        total_entities = len(result.entities)
        unique_types = len(result.entity_count_by_type)
        
        if total_entities > 5 or unique_types > 3:
            analysis["complexity"] = "high"
        elif total_entities > 2 or unique_types > 1:
            analysis["complexity"] = "medium"
        
        return analysis