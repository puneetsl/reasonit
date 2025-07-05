"""
Wikipedia Content Synthesis and Processing.

This module processes and synthesizes Wikipedia content to create structured,
relevant information for reasoning tasks with smart filtering and summarization.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
import json
from datetime import datetime

from .wikipedia_search import WikipediaPage, SearchResult
from .knowledge_detector import KnowledgeType, KnowledgeNeed


@dataclass
class SynthesizedContent:
    """Structured and synthesized content from Wikipedia sources."""
    query: str
    summary: str
    key_facts: List[str] = field(default_factory=list)
    structured_data: Dict[str, Any] = field(default_factory=dict)
    timeline: List[Dict[str, str]] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    content_type: KnowledgeType = KnowledgeType.NONE
    citations: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ContentSection:
    """A processed section of Wikipedia content."""
    title: str
    content: str
    relevance_score: float
    key_points: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    source_page: str = ""


class WikipediaContentSynthesizer:
    """
    Synthesizes and processes Wikipedia content for reasoning tasks.
    
    Processes raw Wikipedia content to extract relevant information,
    create structured summaries, and filter content based on query needs.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._init_content_patterns()
        self._init_relevance_indicators()
        
    def _init_content_patterns(self):
        """Initialize patterns for content processing."""
        
        # Patterns for extracting structured information
        self.extraction_patterns = {
            "dates": [
                r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{4}\b'
            ],
            "numbers": [
                r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand|%|percent)\b',
                r'\b\d+(?:\.\d+)?\s*(?:km|miles|meters|feet|kg|pounds|years|months|days)\b'
            ],
            "locations": [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z][a-z]+\b',  # City, State/Country
                r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
            ],
            "people": [
                r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                r'\b(?:President|King|Queen|Emperor|Prime Minister|Dr\.?|Prof\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            ]
        }
        
        # Patterns for identifying important content
        self.importance_patterns = {
            "high_importance": [
                r'^[A-Z][^.]*\s+(?:is|was|are|were)\s+',  # Definitional statements
                r'(?:established|founded|created|invented|discovered)\s+(?:in|on|by)',
                r'(?:first|last|only|largest|smallest|most|least)',
                r'(?:significant|important|major|primary|main|key)'
            ],
            "factual_statements": [
                r'\b(?:according to|reported|stated|estimated|measured)\b',
                r'\b\d+(?:\.\d+)?(?:\s*million|\s*billion|\s*thousand|\s*%)\b',
                r'\b(?:located|situated|positioned)\s+(?:in|at|on)\b'
            ]
        }
    
    def _init_relevance_indicators(self):
        """Initialize indicators for content relevance scoring."""
        
        self.relevance_indicators = {
            "biographical": {
                "high": ["born", "died", "career", "achievements", "education", "early life"],
                "medium": ["family", "personal", "later life", "legacy", "influence"],
                "sections": ["Early life", "Career", "Personal life", "Death", "Legacy"]
            },
            "historical": {
                "high": ["date", "year", "century", "war", "battle", "treaty", "events"],
                "medium": ["background", "aftermath", "consequences", "significance"],
                "sections": ["Background", "Events", "Course", "Aftermath", "Significance"]
            },
            "scientific": {
                "high": ["theory", "principle", "law", "discovery", "experiment", "formula"],
                "medium": ["application", "example", "history", "development"],
                "sections": ["Theory", "Principles", "Applications", "Examples", "History"]
            },
            "geographic": {
                "high": ["location", "population", "area", "climate", "geography"],
                "medium": ["history", "economy", "culture", "demographics"],
                "sections": ["Geography", "Climate", "Demographics", "Economy", "History"]
            }
        }
    
    def synthesize_content(self, 
                          search_result: SearchResult, 
                          knowledge_need: KnowledgeNeed,
                          max_length: int = 2000) -> SynthesizedContent:
        """
        Synthesize Wikipedia content based on knowledge needs.
        
        Args:
            search_result: Wikipedia search results
            knowledge_need: Detected knowledge requirements
            max_length: Maximum length of synthesized content
            
        Returns:
            Structured and synthesized content
        """
        self.logger.info(f"Synthesizing content for {knowledge_need.knowledge_type.value} query: {search_result.query}")
        
        # Process each page and extract relevant content
        processed_sections = []
        
        for page in search_result.pages[:5]:  # Use top 5 pages
            sections = self._process_page_content(page, knowledge_need)
            processed_sections.extend(sections)
        
        # Rank sections by relevance
        ranked_sections = self._rank_content_sections(processed_sections, knowledge_need)
        
        # Create structured summary
        summary = self._create_summary(ranked_sections, max_length // 2)
        
        # Extract key facts
        key_facts = self._extract_key_facts(ranked_sections, knowledge_need)
        
        # Extract structured data
        structured_data = self._extract_structured_data(ranked_sections, knowledge_need)
        
        # Create timeline if relevant
        timeline = self._create_timeline(ranked_sections, knowledge_need)
        
        # Extract related topics
        related_topics = self._extract_related_topics(search_result.pages, knowledge_need)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(ranked_sections, search_result)
        
        # Create citations
        citations = self._create_citations(search_result.pages)
        
        return SynthesizedContent(
            query=search_result.query,
            summary=summary,
            key_facts=key_facts,
            structured_data=structured_data,
            timeline=timeline,
            related_topics=related_topics,
            sources=[page.url for page in search_result.pages],
            confidence_score=confidence_score,
            content_type=knowledge_need.knowledge_type,
            citations=citations
        )
    
    def _process_page_content(self, page: WikipediaPage, knowledge_need: KnowledgeNeed) -> List[ContentSection]:
        """Process a single Wikipedia page into relevant sections."""
        sections = []
        
        # Process main summary
        if page.summary:
            summary_section = ContentSection(
                title="Summary",
                content=page.summary,
                relevance_score=self._calculate_section_relevance(page.summary, knowledge_need),
                key_points=self._extract_key_points(page.summary),
                entities=self._extract_entities_from_text(page.summary),
                source_page=page.title
            )
            sections.append(summary_section)
        
        # Process individual sections
        for section_title, section_content in page.sections.items():
            if section_content and len(section_content) > 50:  # Skip very short sections
                processed_section = ContentSection(
                    title=section_title,
                    content=section_content,
                    relevance_score=self._calculate_section_relevance(section_content, knowledge_need, section_title),
                    key_points=self._extract_key_points(section_content),
                    entities=self._extract_entities_from_text(section_content),
                    source_page=page.title
                )
                sections.append(processed_section)
        
        # Process infobox as structured data
        if page.infobox:
            infobox_content = self._format_infobox_as_text(page.infobox)
            infobox_section = ContentSection(
                title="Key Information",
                content=infobox_content,
                relevance_score=0.9,  # Infoboxes are usually highly relevant
                key_points=list(page.infobox.values())[:5],
                entities=self._extract_entities_from_text(infobox_content),
                source_page=page.title
            )
            sections.append(infobox_section)
        
        return sections
    
    def _calculate_section_relevance(self, 
                                   content: str, 
                                   knowledge_need: KnowledgeNeed,
                                   section_title: str = "") -> float:
        """Calculate relevance score for a content section."""
        score = 0.0
        content_lower = content.lower()
        title_lower = section_title.lower()
        
        # Base score from knowledge type indicators
        knowledge_type = knowledge_need.knowledge_type.value
        if knowledge_type in self.relevance_indicators:
            indicators = self.relevance_indicators[knowledge_type]
            
            # High importance keywords
            high_keywords = indicators.get("high", [])
            high_matches = sum(1 for keyword in high_keywords if keyword in content_lower)
            score += min(high_matches / len(high_keywords), 1.0) * 0.4
            
            # Medium importance keywords
            medium_keywords = indicators.get("medium", [])
            medium_matches = sum(1 for keyword in medium_keywords if keyword in content_lower)
            score += min(medium_matches / len(medium_keywords), 1.0) * 0.2
            
            # Preferred sections
            preferred_sections = indicators.get("sections", [])
            if any(pref.lower() in title_lower for pref in preferred_sections):
                score += 0.3
        
        # Entity matching bonus
        entity_matches = 0
        for entity in knowledge_need.entities:
            if entity.lower() in content_lower:
                entity_matches += 1
        
        if knowledge_need.entities:
            entity_score = min(entity_matches / len(knowledge_need.entities), 1.0) * 0.2
            score += entity_score
        
        # Content quality indicators
        if any(pattern in content_lower for pattern in ["according to", "research", "study", "data"]):
            score += 0.1
        
        # Length penalty for very short content
        if len(content) < 100:
            score *= 0.7
        
        return min(score, 1.0)
    
    def _rank_content_sections(self, sections: List[ContentSection], knowledge_need: KnowledgeNeed) -> List[ContentSection]:
        """Rank content sections by relevance."""
        # Sort by relevance score
        ranked = sorted(sections, key=lambda s: s.relevance_score, reverse=True)
        
        # Apply additional ranking factors
        final_ranking = []
        
        for section in ranked:
            # Boost score for sections with entities from the query
            entity_boost = 0
            for entity in knowledge_need.entities:
                if entity.lower() in section.content.lower():
                    entity_boost += 0.1
            
            section.relevance_score += entity_boost
            final_ranking.append(section)
        
        # Re-sort with updated scores
        return sorted(final_ranking, key=lambda s: s.relevance_score, reverse=True)
    
    def _create_summary(self, sections: List[ContentSection], max_length: int) -> str:
        """Create a synthesized summary from top sections."""
        summary_parts = []
        current_length = 0
        
        # Use top sections
        for section in sections[:5]:  # Top 5 sections
            if section.relevance_score > 0.3:  # Only include relevant sections
                # Take key sentences from the section
                sentences = self._extract_key_sentences(section.content, 2)
                
                for sentence in sentences:
                    if current_length + len(sentence) < max_length:
                        summary_parts.append(sentence)
                        current_length += len(sentence)
                    else:
                        break
                
                if current_length >= max_length * 0.8:  # Stop when near limit
                    break
        
        return " ".join(summary_parts)
    
    def _extract_key_sentences(self, text: str, max_sentences: int) -> List[str]:
        """Extract key sentences from text."""
        sentences = re.split(r'[.!?]+', text)
        
        # Score sentences based on importance indicators
        scored_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Skip very short sentences
                score = 0
                
                # Higher score for sentences with important patterns
                for pattern in self.importance_patterns["high_importance"]:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        score += 2
                
                for pattern in self.importance_patterns["factual_statements"]:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        score += 1
                
                # Prefer sentences with numbers, dates, or specific facts
                if re.search(r'\b\d+\b', sentence):
                    score += 1
                
                scored_sentences.append((sentence, score))
        
        # Sort by score and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sent[0] for sent in scored_sentences[:max_sentences]]
    
    def _extract_key_facts(self, sections: List[ContentSection], knowledge_need: KnowledgeNeed) -> List[str]:
        """Extract key facts from processed sections."""
        facts = []
        
        # Extract facts based on knowledge type
        if knowledge_need.knowledge_type == KnowledgeType.BIOGRAPHICAL:
            facts.extend(self._extract_biographical_facts(sections))
        elif knowledge_need.knowledge_type == KnowledgeType.HISTORICAL_EVENTS:
            facts.extend(self._extract_historical_facts(sections))
        elif knowledge_need.knowledge_type == KnowledgeType.GEOGRAPHIC:
            facts.extend(self._extract_geographic_facts(sections))
        elif knowledge_need.knowledge_type == KnowledgeType.SCIENTIFIC_CONCEPTS:
            facts.extend(self._extract_scientific_facts(sections))
        else:
            facts.extend(self._extract_general_facts(sections))
        
        return facts[:10]  # Limit to top 10 facts
    
    def _extract_biographical_facts(self, sections: List[ContentSection]) -> List[str]:
        """Extract biographical facts."""
        facts = []
        
        for section in sections:
            content = section.content
            
            # Extract birth/death information
            birth_matches = re.findall(r'born\s+(?:on\s+)?([^,.]+)', content, re.IGNORECASE)
            for match in birth_matches:
                facts.append(f"Born: {match.strip()}")
            
            death_matches = re.findall(r'died\s+(?:on\s+)?([^,.]+)', content, re.IGNORECASE)
            for match in death_matches:
                facts.append(f"Died: {match.strip()}")
            
            # Extract career information
            career_matches = re.findall(r'(?:was|is)\s+(?:a|an)\s+([^,.]+(?:scientist|artist|writer|politician|inventor|philosopher)[^,.]*)', content, re.IGNORECASE)
            for match in career_matches:
                facts.append(f"Profession: {match.strip()}")
        
        return facts
    
    def _extract_historical_facts(self, sections: List[ContentSection]) -> List[str]:
        """Extract historical facts."""
        facts = []
        
        for section in sections:
            content = section.content
            
            # Extract date-related facts
            date_matches = re.findall(r'(?:in|on|during)\s+(\d{4}[^,.]*)', content)
            for match in date_matches:
                facts.append(f"Date: {match.strip()}")
            
            # Extract event descriptions
            event_matches = re.findall(r'(?:began|started|ended|occurred|happened)\s+([^,.]+)', content, re.IGNORECASE)
            for match in event_matches:
                facts.append(f"Event: {match.strip()}")
        
        return facts
    
    def _extract_geographic_facts(self, sections: List[ContentSection]) -> List[str]:
        """Extract geographic facts."""
        facts = []
        
        for section in sections:
            content = section.content
            
            # Extract location facts
            location_matches = re.findall(r'located\s+(?:in|at|on)\s+([^,.]+)', content, re.IGNORECASE)
            for match in location_matches:
                facts.append(f"Location: {match.strip()}")
            
            # Extract population/area facts
            pop_matches = re.findall(r'population\s+(?:of\s+)?([^,.]+)', content, re.IGNORECASE)
            for match in pop_matches:
                facts.append(f"Population: {match.strip()}")
            
            area_matches = re.findall(r'area\s+(?:of\s+)?([^,.]+)', content, re.IGNORECASE)
            for match in area_matches:
                facts.append(f"Area: {match.strip()}")
        
        return facts
    
    def _extract_scientific_facts(self, sections: List[ContentSection]) -> List[str]:
        """Extract scientific facts."""
        facts = []
        
        for section in sections:
            content = section.content
            
            # Extract definitions
            def_matches = re.findall(r'(?:is|are)\s+(?:a|an|the)\s+([^,.]+(?:theory|principle|law|concept|phenomenon)[^,.]*)', content, re.IGNORECASE)
            for match in def_matches:
                facts.append(f"Definition: {match.strip()}")
            
            # Extract discoveries
            disc_matches = re.findall(r'(?:discovered|invented|developed)\s+(?:by|in)\s+([^,.]+)', content, re.IGNORECASE)
            for match in disc_matches:
                facts.append(f"Discovery: {match.strip()}")
        
        return facts
    
    def _extract_general_facts(self, sections: List[ContentSection]) -> List[str]:
        """Extract general facts."""
        facts = []
        
        for section in sections:
            # Use key points from high-relevance sections
            if section.relevance_score > 0.5:
                facts.extend(section.key_points[:3])
        
        return facts
    
    def _extract_structured_data(self, sections: List[ContentSection], knowledge_need: KnowledgeNeed) -> Dict[str, Any]:
        """Extract structured data from sections."""
        structured = {}
        
        # Extract dates
        dates = []
        for section in sections:
            for pattern in self.extraction_patterns["dates"]:
                matches = re.findall(pattern, section.content)
                dates.extend(matches)
        if dates:
            structured["dates"] = list(set(dates))[:5]
        
        # Extract numbers/statistics
        numbers = []
        for section in sections:
            for pattern in self.extraction_patterns["numbers"]:
                matches = re.findall(pattern, section.content)
                numbers.extend(matches)
        if numbers:
            structured["statistics"] = list(set(numbers))[:5]
        
        # Extract locations
        locations = []
        for section in sections:
            for pattern in self.extraction_patterns["locations"]:
                matches = re.findall(pattern, section.content)
                locations.extend(matches)
        if locations:
            structured["locations"] = list(set(locations))[:5]
        
        # Extract people
        people = []
        for section in sections:
            for pattern in self.extraction_patterns["people"]:
                matches = re.findall(pattern, section.content)
                people.extend(matches)
        if people:
            structured["people"] = list(set(people))[:5]
        
        return structured
    
    def _create_timeline(self, sections: List[ContentSection], knowledge_need: KnowledgeNeed) -> List[Dict[str, str]]:
        """Create a timeline from temporal information."""
        timeline_events = []
        
        if knowledge_need.knowledge_type not in [KnowledgeType.HISTORICAL_EVENTS, KnowledgeType.BIOGRAPHICAL]:
            return timeline_events
        
        # Extract date-event pairs
        date_pattern = r'(?:in|on|during)\s+(\d{4})[^.]*?([^.]+)'
        
        for section in sections:
            matches = re.findall(date_pattern, section.content)
            for year, event in matches:
                if len(event.strip()) > 10:  # Skip very short events
                    timeline_events.append({
                        "year": year,
                        "event": event.strip()[:200],  # Limit event description
                        "source": section.source_page
                    })
        
        # Sort by year and deduplicate
        unique_events = {}
        for event in timeline_events:
            key = event["year"]
            if key not in unique_events or len(event["event"]) > len(unique_events[key]["event"]):
                unique_events[key] = event
        
        sorted_timeline = sorted(unique_events.values(), key=lambda x: x["year"])
        
        return sorted_timeline[:10]  # Limit to 10 events
    
    def _extract_related_topics(self, pages: List[WikipediaPage], knowledge_need: KnowledgeNeed) -> List[str]:
        """Extract related topics from page categories and links."""
        related = set()
        
        # Extract from categories
        for page in pages[:3]:  # Top 3 pages
            for category in page.categories[:5]:
                # Filter out generic categories
                if not any(generic in category.lower() for generic in ["articles", "pages", "wikipedia", "living people"]):
                    related.add(category)
        
        # Extract from infobox keys (often represent related concepts)
        for page in pages[:2]:
            for key in page.infobox.keys():
                if len(key) > 3 and key not in ["Name", "Image", "Caption"]:
                    related.add(key)
        
        return list(related)[:8]  # Limit to 8 related topics
    
    def _calculate_confidence_score(self, sections: List[ContentSection], search_result: SearchResult) -> float:
        """Calculate confidence score for synthesized content."""
        if not sections:
            return 0.0
        
        # Base score from section relevance
        avg_relevance = sum(s.relevance_score for s in sections[:5]) / min(len(sections), 5)
        
        # Boost for multiple sources
        source_count = len(set(s.source_page for s in sections))
        source_bonus = min(source_count * 0.1, 0.3)
        
        # Boost for content length (indicates comprehensive information)
        total_content_length = sum(len(s.content) for s in sections[:5])
        length_bonus = min(total_content_length / 5000, 0.2)  # Up to 0.2 bonus for 5000+ chars
        
        # Boost for structured data presence
        structured_bonus = 0.1 if any(s.title == "Key Information" for s in sections) else 0.0
        
        final_score = avg_relevance + source_bonus + length_bonus + structured_bonus
        
        return min(final_score, 1.0)
    
    def _create_citations(self, pages: List[WikipediaPage]) -> List[Dict[str, str]]:
        """Create citation information for sources."""
        citations = []
        
        for page in pages:
            citation = {
                "title": page.title,
                "url": page.url,
                "source": "Wikipedia",
                "language": page.language,
                "accessed": datetime.now().strftime("%Y-%m-%d")
            }
            
            if page.last_modified:
                citation["last_modified"] = page.last_modified.strftime("%Y-%m-%d")
            
            citations.append(citation)
        
        return citations
    
    def _format_infobox_as_text(self, infobox: Dict[str, str]) -> str:
        """Format infobox data as readable text."""
        formatted_items = []
        
        for key, value in infobox.items():
            # Clean up the value
            clean_value = re.sub(r'\s+', ' ', value).strip()
            if clean_value and len(clean_value) < 200:  # Skip very long values
                formatted_items.append(f"{key}: {clean_value}")
        
        return ". ".join(formatted_items)
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text content."""
        # Split into sentences and score them
        sentences = re.split(r'[.!?]+', text)
        key_points = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Skip very short sentences
                # Score based on presence of important indicators
                score = 0
                if any(pattern in sentence.lower() for pattern in ["is", "was", "are", "were"]):
                    score += 1
                if re.search(r'\b\d+\b', sentence):  # Contains numbers
                    score += 1
                if any(word in sentence.lower() for word in ["first", "largest", "most", "only", "main"]):
                    score += 2
                
                if score >= 1:
                    key_points.append(sentence)
        
        return key_points[:5]  # Return top 5 key points
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract entities from text using simple patterns."""
        entities = []
        
        # Extract potential person names (capitalized words)
        person_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        person_matches = re.findall(person_pattern, text)
        entities.extend(person_matches)
        
        # Extract potential place names
        place_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?=\s+(?:is|was|located|situated))\b'
        place_matches = re.findall(place_pattern, text)
        entities.extend(place_matches)
        
        # Remove duplicates and return
        return list(set(entities))[:5]