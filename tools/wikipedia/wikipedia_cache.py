"""
Wikipedia Content Caching System.

This module provides intelligent caching for Wikipedia content with TTL management,
update tracking, and efficient storage for offline capability.
"""

import json
import logging
import pickle
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import sqlite3
import threading

from .wikipedia_search import WikipediaPage, SearchResult


@dataclass
class CacheEntry:
    """Represents a cached Wikipedia entry."""
    key: str
    content: Any  # Can be WikipediaPage or SearchResult
    cached_at: datetime
    ttl: timedelta
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    content_type: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > (self.cached_at + self.ttl)
    
    @property
    def age_hours(self) -> float:
        """Get age of cache entry in hours."""
        return (datetime.now() - self.cached_at).total_seconds() / 3600


@dataclass
class CacheStats:
    """Cache usage statistics."""
    total_entries: int = 0
    total_size_mb: float = 0.0
    hit_count: int = 0
    miss_count: int = 0
    expired_count: int = 0
    evicted_count: int = 0
    avg_access_count: float = 0.0
    cache_hit_ratio: float = 0.0
    oldest_entry_age_hours: float = 0.0
    newest_entry_age_hours: float = 0.0


class WikipediaCache:
    """
    Intelligent caching system for Wikipedia content.
    
    Provides persistent storage, TTL management, LRU eviction,
    and intelligent cache warming for frequently accessed content.
    """
    
    def __init__(self, 
                 cache_dir: str = "wikipedia_cache",
                 max_entries: int = 1000,
                 default_ttl_hours: int = 24,
                 max_size_mb: int = 500,
                 enable_persistence: bool = True):
        self.cache_dir = Path(cache_dir)
        self.max_entries = max_entries
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.max_size_mb = max_size_mb
        self.enable_persistence = enable_persistence
        
        self.logger = logging.getLogger(__name__)
        
        # In-memory cache
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.RLock()
        
        # Statistics
        self._stats = CacheStats()
        
        # Initialize storage
        if enable_persistence:
            self._init_persistent_storage()
            self._load_cache_from_disk()
    
    def _init_persistent_storage(self):
        """Initialize persistent storage directory and database."""
        self.cache_dir.mkdir(exist_ok=True)
        
        # SQLite database for metadata
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()
        
        # Directory for content files
        self.content_dir = self.cache_dir / "content"
        self.content_dir.mkdir(exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    cached_at TEXT,
                    ttl_hours REAL,
                    access_count INTEGER,
                    last_accessed TEXT,
                    content_type TEXT,
                    file_path TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_stats (
                    stat_name TEXT PRIMARY KEY,
                    stat_value TEXT,
                    updated_at TEXT
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cached_at ON cache_entries(cached_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_access_count ON cache_entries(access_count)")
    
    def _create_cache_key(self, identifier: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Create a unique cache key."""
        key_data = identifier
        if params:
            params_str = json.dumps(params, sort_keys=True)
            key_data += f"_{params_str}"
        
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[Any]:
        """Get an item from the cache."""
        with self._cache_lock:
            if key not in self._cache:
                self._stats.miss_count += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired:
                self.logger.debug(f"Cache entry expired: {key}")
                del self._cache[key]
                self._stats.expired_count += 1
                self._stats.miss_count += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            self._stats.hit_count += 1
            
            self.logger.debug(f"Cache hit: {key} (accessed {entry.access_count} times)")
            return entry.content
    
    def put(self, 
            key: str, 
            content: Any, 
            ttl: Optional[timedelta] = None,
            content_type: str = "unknown") -> None:
        """Put an item in the cache."""
        ttl = ttl or self.default_ttl
        
        with self._cache_lock:
            # Check if we need to evict entries
            if len(self._cache) >= self.max_entries:
                self._evict_entries()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                content=content,
                cached_at=datetime.now(),
                ttl=ttl,
                content_type=content_type,
                metadata={}
            )
            
            self._cache[key] = entry
            
            # Persist to disk if enabled
            if self.enable_persistence:
                self._persist_entry(entry)
            
            self.logger.debug(f"Cached entry: {key} (TTL: {ttl})")
    
    def _evict_entries(self, target_count: Optional[int] = None):
        """Evict entries using LRU policy."""
        target_count = target_count or int(self.max_entries * 0.8)  # Evict to 80% capacity
        
        if len(self._cache) <= target_count:
            return
        
        # Sort by last accessed time (LRU)
        entries_by_access = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_accessed or x[1].cached_at
        )
        
        # Remove oldest entries
        entries_to_remove = len(self._cache) - target_count
        
        for i in range(entries_to_remove):
            key, entry = entries_by_access[i]
            del self._cache[key]
            self._stats.evicted_count += 1
            
            # Remove from persistent storage
            if self.enable_persistence:
                self._remove_persistent_entry(key)
        
        self.logger.info(f"Evicted {entries_to_remove} cache entries")
    
    def cache_wikipedia_page(self, page: WikipediaPage, custom_ttl: Optional[timedelta] = None) -> str:
        """Cache a Wikipedia page and return the cache key."""
        key = self._create_cache_key(f"page_{page.title}")
        self.put(key, page, custom_ttl, "WikipediaPage")
        return key
    
    def get_wikipedia_page(self, title: str) -> Optional[WikipediaPage]:
        """Get a cached Wikipedia page by title."""
        key = self._create_cache_key(f"page_{title}")
        return self.get(key)
    
    def cache_search_result(self, result: SearchResult, custom_ttl: Optional[timedelta] = None) -> str:
        """Cache a search result and return the cache key."""
        key = self._create_cache_key(f"search_{result.query}")
        self.put(key, result, custom_ttl, "SearchResult")
        return key
    
    def get_search_result(self, query: str) -> Optional[SearchResult]:
        """Get a cached search result by query."""
        key = self._create_cache_key(f"search_{query}")
        return self.get(key)
    
    def invalidate(self, key: str) -> bool:
        """Remove a specific entry from cache."""
        with self._cache_lock:
            if key in self._cache:
                del self._cache[key]
                
                if self.enable_persistence:
                    self._remove_persistent_entry(key)
                
                self.logger.debug(f"Invalidated cache entry: {key}")
                return True
            
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Remove entries matching a pattern."""
        with self._cache_lock:
            keys_to_remove = [key for key in self._cache.keys() if pattern in key]
            
            for key in keys_to_remove:
                del self._cache[key]
                
                if self.enable_persistence:
                    self._remove_persistent_entry(key)
            
            self.logger.info(f"Invalidated {len(keys_to_remove)} entries matching pattern: {pattern}")
            return len(keys_to_remove)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._cache_lock:
            entry_count = len(self._cache)
            self._cache.clear()
            
            if self.enable_persistence:
                self._clear_persistent_storage()
            
            self.logger.info(f"Cleared {entry_count} cache entries")
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        with self._cache_lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self._cache[key]
                
                if self.enable_persistence:
                    self._remove_persistent_entry(key)
            
            self._stats.expired_count += len(expired_keys)
            
            if expired_keys:
                self.logger.info(f"Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._cache_lock:
            stats = CacheStats(
                total_entries=len(self._cache),
                hit_count=self._stats.hit_count,
                miss_count=self._stats.miss_count,
                expired_count=self._stats.expired_count,
                evicted_count=self._stats.evicted_count
            )
            
            # Calculate derived statistics
            total_requests = stats.hit_count + stats.miss_count
            if total_requests > 0:
                stats.cache_hit_ratio = stats.hit_count / total_requests
            
            if self._cache:
                access_counts = [entry.access_count for entry in self._cache.values()]
                stats.avg_access_count = sum(access_counts) / len(access_counts)
                
                ages = [entry.age_hours for entry in self._cache.values()]
                stats.oldest_entry_age_hours = max(ages)
                stats.newest_entry_age_hours = min(ages)
            
            # Estimate cache size
            stats.total_size_mb = self._estimate_cache_size()
            
            return stats
    
    def _estimate_cache_size(self) -> float:
        """Estimate total cache size in MB."""
        if not self._cache:
            return 0.0
        
        # Sample a few entries to estimate average size
        sample_entries = list(self._cache.values())[:min(10, len(self._cache))]
        
        total_sample_size = 0
        for entry in sample_entries:
            try:
                # Estimate size using pickle
                serialized = pickle.dumps(entry.content)
                total_sample_size += len(serialized)
            except Exception:
                # Fallback estimation
                total_sample_size += 1024  # 1KB default
        
        if sample_entries:
            avg_entry_size = total_sample_size / len(sample_entries)
            total_size_bytes = avg_entry_size * len(self._cache)
            return total_size_bytes / (1024 * 1024)  # Convert to MB
        
        return 0.0
    
    def _persist_entry(self, entry: CacheEntry) -> None:
        """Persist a cache entry to disk."""
        try:
            # Save content to file
            content_file = self.content_dir / f"{entry.key}.pkl"
            with open(content_file, 'wb') as f:
                pickle.dump(entry.content, f)
            
            # Save metadata to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, cached_at, ttl_hours, access_count, last_accessed, content_type, file_path, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.key,
                    entry.cached_at.isoformat(),
                    entry.ttl.total_seconds() / 3600,
                    entry.access_count,
                    entry.last_accessed.isoformat() if entry.last_accessed else None,
                    entry.content_type,
                    str(content_file),
                    json.dumps(entry.metadata)
                ))
        
        except Exception as e:
            self.logger.warning(f"Failed to persist cache entry {entry.key}: {e}")
    
    def _load_cache_from_disk(self) -> None:
        """Load cache entries from persistent storage."""
        if not self.db_path.exists():
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM cache_entries")
                rows = cursor.fetchall()
            
            loaded_count = 0
            for row in rows:
                key, cached_at_str, ttl_hours, access_count, last_accessed_str, content_type, file_path, metadata_str = row
                
                # Check if content file exists
                content_file = Path(file_path)
                if not content_file.exists():
                    continue
                
                try:
                    # Load content
                    with open(content_file, 'rb') as f:
                        content = pickle.load(f)
                    
                    # Create cache entry
                    entry = CacheEntry(
                        key=key,
                        content=content,
                        cached_at=datetime.fromisoformat(cached_at_str),
                        ttl=timedelta(hours=ttl_hours),
                        access_count=access_count,
                        last_accessed=datetime.fromisoformat(last_accessed_str) if last_accessed_str else None,
                        content_type=content_type,
                        metadata=json.loads(metadata_str) if metadata_str else {}
                    )
                    
                    # Only load if not expired
                    if not entry.is_expired:
                        self._cache[key] = entry
                        loaded_count += 1
                    else:
                        # Remove expired content file
                        content_file.unlink(missing_ok=True)
                
                except Exception as e:
                    self.logger.warning(f"Failed to load cache entry {key}: {e}")
                    # Clean up corrupted file
                    content_file.unlink(missing_ok=True)
            
            if loaded_count > 0:
                self.logger.info(f"Loaded {loaded_count} cache entries from disk")
        
        except Exception as e:
            self.logger.error(f"Failed to load cache from disk: {e}")
    
    def _remove_persistent_entry(self, key: str) -> None:
        """Remove a persistent cache entry."""
        try:
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            
            # Remove content file
            content_file = self.content_dir / f"{key}.pkl"
            content_file.unlink(missing_ok=True)
        
        except Exception as e:
            self.logger.warning(f"Failed to remove persistent entry {key}: {e}")
    
    def _clear_persistent_storage(self) -> None:
        """Clear all persistent storage."""
        try:
            # Clear database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")
            
            # Remove all content files
            for content_file in self.content_dir.glob("*.pkl"):
                content_file.unlink(missing_ok=True)
        
        except Exception as e:
            self.logger.error(f"Failed to clear persistent storage: {e}")
    
    def warm_cache(self, popular_queries: List[str]) -> None:
        """Warm the cache with popular queries (to be called with actual search)."""
        self.logger.info(f"Cache warming requested for {len(popular_queries)} queries")
        # This would typically involve pre-loading popular content
        # Implementation depends on integration with WikipediaSearchTool
    
    def export_cache_report(self) -> Dict[str, Any]:
        """Export a detailed cache report."""
        stats = self.get_stats()
        
        with self._cache_lock:
            # Get top accessed entries
            top_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].access_count,
                reverse=True
            )[:10]
            
            # Get entries by type
            entries_by_type = {}
            for entry in self._cache.values():
                content_type = entry.content_type
                if content_type not in entries_by_type:
                    entries_by_type[content_type] = 0
                entries_by_type[content_type] += 1
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "statistics": asdict(stats),
                "top_accessed_entries": [
                    {
                        "key": key,
                        "access_count": entry.access_count,
                        "age_hours": entry.age_hours,
                        "content_type": entry.content_type
                    }
                    for key, entry in top_entries
                ],
                "entries_by_type": entries_by_type,
                "cache_settings": {
                    "max_entries": self.max_entries,
                    "default_ttl_hours": self.default_ttl.total_seconds() / 3600,
                    "max_size_mb": self.max_size_mb,
                    "persistence_enabled": self.enable_persistence
                }
            }
            
            return report
    
    def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache performance and return optimization results."""
        with self._cache_lock:
            initial_count = len(self._cache)
            
            # Clean up expired entries
            expired_removed = self.cleanup_expired()
            
            # Check if we're over size limit
            current_size = self._estimate_cache_size()
            size_optimized = False
            
            if current_size > self.max_size_mb:
                # Evict less frequently accessed entries
                self._evict_entries(int(self.max_entries * 0.7))
                size_optimized = True
            
            final_count = len(self._cache)
            
            optimization_result = {
                "initial_entries": initial_count,
                "final_entries": final_count,
                "expired_removed": expired_removed,
                "size_optimized": size_optimized,
                "estimated_size_mb": self._estimate_cache_size()
            }
            
            self.logger.info(f"Cache optimization completed: {optimization_result}")
            
            return optimization_result


# Convenience functions

def create_default_cache() -> WikipediaCache:
    """Create a Wikipedia cache with default settings."""
    return WikipediaCache()


def create_memory_only_cache(max_entries: int = 500) -> WikipediaCache:
    """Create an in-memory only cache."""
    return WikipediaCache(
        max_entries=max_entries,
        enable_persistence=False
    )