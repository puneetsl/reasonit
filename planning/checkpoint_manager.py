"""
Checkpoint manager for saving and restoring planning state.

This module provides functionality to save plan execution state at various
checkpoints and restore from them in case of failures or interruptions.
"""

import asyncio
import json
import logging
import pickle
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import asdict
import sqlite3
import gzip

from .task_planner import Plan, Task, TaskStatus, PlanningMetrics

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoints for plan execution state.
    
    Provides functionality to save and restore plan state, handle recovery
    from failures, and maintain execution history.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        auto_checkpoint_interval: int = 300,  # 5 minutes
        max_checkpoints_per_plan: int = 10,
        enable_compression: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.auto_checkpoint_interval = auto_checkpoint_interval
        self.max_checkpoints_per_plan = max_checkpoints_per_plan
        self.enable_compression = enable_compression
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database for metadata
        self.db_path = self.checkpoint_dir / "checkpoints.db"
        self._init_database()
        
        # Auto-checkpoint task
        self.auto_checkpoint_task: Optional[asyncio.Task] = None
        self.active_plans: Dict[str, Plan] = {}
        
        logger.info(f"Initialized CheckpointManager at {checkpoint_dir}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for checkpoint metadata."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    checkpoint_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    compressed BOOLEAN DEFAULT FALSE,
                    metadata TEXT
                )
            """)
            
            # Create indexes separately
            conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_plan_id ON checkpoints(plan_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON checkpoints(created_at)")
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plan_metadata (
                    plan_id TEXT PRIMARY KEY,
                    name TEXT,
                    status TEXT,
                    created_at TIMESTAMP,
                    last_checkpoint TIMESTAMP,
                    total_tasks INTEGER,
                    completed_tasks INTEGER,
                    metadata TEXT
                )
            """)
        
        logger.debug("Checkpoint database initialized")
    
    async def save_checkpoint(
        self,
        plan: Plan,
        checkpoint_type: str = "manual",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a checkpoint for the given plan.
        
        Args:
            plan: Plan to checkpoint
            checkpoint_type: Type of checkpoint (manual, auto, milestone)
            metadata: Additional metadata to store
            
        Returns:
            Checkpoint ID
        """
        
        checkpoint_id = f"{plan.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{checkpoint_type}"
        timestamp = datetime.now()
        
        # Prepare plan data for serialization
        plan_data = self._serialize_plan(plan)
        
        # Save to file
        file_path = self.checkpoint_dir / f"{checkpoint_id}.checkpoint"
        
        try:
            if self.enable_compression:
                with gzip.open(f"{file_path}.gz", 'wb') as f:
                    pickle.dump(plan_data, f)
                file_path = f"{file_path}.gz"
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(plan_data, f)
            
            file_size = os.path.getsize(file_path)
            
            # Save metadata to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO checkpoints 
                    (id, plan_id, checkpoint_type, created_at, file_path, file_size, compressed, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    checkpoint_id,
                    plan.id,
                    checkpoint_type,
                    timestamp,
                    str(file_path),
                    file_size,
                    self.enable_compression,
                    json.dumps(metadata or {})
                ))
                
                # Update plan metadata
                conn.execute("""
                    INSERT OR REPLACE INTO plan_metadata
                    (plan_id, name, status, created_at, last_checkpoint, total_tasks, completed_tasks, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    plan.id,
                    plan.name,
                    plan.status.value,
                    plan.created_at,
                    timestamp,
                    len(plan.tasks),
                    sum(1 for t in plan.tasks.values() if t.status == TaskStatus.COMPLETED),
                    json.dumps(plan.metadata)
                ))
            
            # Clean up old checkpoints
            await self._cleanup_old_checkpoints(plan.id)
            
            logger.info(f"Saved checkpoint {checkpoint_id} for plan {plan.id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            # Clean up partial file
            if file_path.exists():
                file_path.unlink()
            raise
    
    async def restore_checkpoint(self, checkpoint_id: str) -> Optional[Plan]:
        """
        Restore a plan from checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to restore
            
        Returns:
            Restored plan or None if not found
        """
        
        try:
            # Get checkpoint metadata
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT file_path, compressed, metadata 
                    FROM checkpoints 
                    WHERE id = ?
                """, (checkpoint_id,))
                
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Checkpoint {checkpoint_id} not found")
                    return None
                
                file_path, compressed, metadata_json = row
            
            # Load plan data
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"Checkpoint file {file_path} not found")
                return None
            
            if compressed:
                with gzip.open(file_path, 'rb') as f:
                    plan_data = pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    plan_data = pickle.load(f)
            
            # Deserialize plan
            plan = self._deserialize_plan(plan_data)
            
            logger.info(f"Restored plan {plan.id} from checkpoint {checkpoint_id}")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            return None
    
    async def restore_latest_checkpoint(self, plan_id: str) -> Optional[Plan]:
        """
        Restore the latest checkpoint for a plan.
        
        Args:
            plan_id: ID of the plan
            
        Returns:
            Restored plan or None if not found
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id FROM checkpoints 
                WHERE plan_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (plan_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            checkpoint_id = row[0]
        
        return await self.restore_checkpoint(checkpoint_id)
    
    def _serialize_plan(self, plan: Plan) -> Dict[str, Any]:
        """Serialize plan to dictionary for storage."""
        
        # Convert plan to dictionary with special handling for complex types
        plan_dict = {
            "id": plan.id,
            "name": plan.name,
            "description": plan.description,
            "status": plan.status.value,
            "root_task_ids": plan.root_task_ids,
            "current_executing": list(plan.current_executing),
            "total_cost": plan.total_cost,
            "total_time": plan.total_time,
            "success_rate": plan.success_rate,
            "created_at": plan.created_at.isoformat(),
            "updated_at": plan.updated_at.isoformat(),
            "metadata": plan.metadata,
            "tasks": {}
        }
        
        # Serialize tasks
        for task_id, task in plan.tasks.items():
            plan_dict["tasks"][task_id] = self._serialize_task(task)
        
        return plan_dict
    
    def _serialize_task(self, task: Task) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        
        task_dict = {
            "id": task.id,
            "name": task.name,
            "description": task.description,
            "task_type": task.task_type.value,
            "priority": task.priority.value,
            "query": task.query,
            "expected_output": task.expected_output,
            "status": task.status.value,
            "assigned_strategy": task.assigned_strategy.value if task.assigned_strategy else None,
            "context_variant": task.context_variant.value,
            "children": task.children,
            "parent": task.parent,
            "error": task.error,
            "start_time": task.start_time.isoformat() if task.start_time else None,
            "end_time": task.end_time.isoformat() if task.end_time else None,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
            "metadata": task.metadata,
            "tags": list(task.tags)
        }
        
        # Serialize dependencies
        task_dict["dependencies"] = []
        for dep in task.dependencies:
            task_dict["dependencies"].append({
                "task_id": dep.task_id,
                "dependency_type": dep.dependency_type,
                "condition": dep.condition,
                "timeout": dep.timeout
            })
        
        # Serialize constraints
        if task.constraints:
            task_dict["constraints"] = {
                "max_time": task.constraints.max_time,
                "max_cost": task.constraints.max_cost,
                "max_memory": task.constraints.max_memory,
                "required_capabilities": task.constraints.required_capabilities,
                "excluded_strategies": [s.value for s in task.constraints.excluded_strategies],
                "minimum_confidence": task.constraints.minimum_confidence
            }
        
        # Serialize request and result (basic info only)
        if task.request:
            task_dict["request"] = {
                "query": task.request.query,
                "strategy": task.request.strategy.value,
                "context_variant": task.request.context_variant.value,
                "confidence_threshold": task.request.confidence_threshold,
                "session_id": task.request.session_id
            }
        
        if task.result:
            task_dict["result"] = {
                "final_answer": task.result.final_answer,
                "total_cost": task.result.total_cost,
                "total_time": task.result.total_time,
                "confidence_score": task.result.confidence_score,
                "outcome": task.result.outcome.value,
                "timestamp": task.result.timestamp.isoformat()
            }
        
        return task_dict
    
    def _deserialize_plan(self, plan_data: Dict[str, Any]) -> Plan:
        """Deserialize plan from dictionary."""
        
        from .task_planner import Plan, TaskStatus
        
        # Create plan
        plan = Plan(
            id=plan_data["id"],
            name=plan_data["name"],
            description=plan_data["description"],
            root_task_ids=plan_data["root_task_ids"],
            total_cost=plan_data["total_cost"],
            total_time=plan_data["total_time"],
            success_rate=plan_data["success_rate"],
            metadata=plan_data["metadata"]
        )
        
        # Set status and times
        plan.status = TaskStatus(plan_data["status"])
        plan.current_executing = set(plan_data["current_executing"])
        plan.created_at = datetime.fromisoformat(plan_data["created_at"])
        plan.updated_at = datetime.fromisoformat(plan_data["updated_at"])
        
        # Deserialize tasks
        for task_id, task_data in plan_data["tasks"].items():
            plan.tasks[task_id] = self._deserialize_task(task_data)
        
        return plan
    
    def _deserialize_task(self, task_data: Dict[str, Any]) -> Task:
        """Deserialize task from dictionary."""
        
        from .task_planner import Task, TaskType, TaskPriority, TaskStatus, TaskDependency, TaskConstraint
        from models import ReasoningStrategy, ContextVariant
        
        # Create task
        task = Task(
            id=task_data["id"],
            name=task_data["name"],
            description=task_data["description"],
            task_type=TaskType(task_data["task_type"]),
            priority=TaskPriority(task_data["priority"]),
            query=task_data["query"],
            expected_output=task_data["expected_output"],
            status=TaskStatus(task_data["status"]),
            context_variant=ContextVariant(task_data["context_variant"]),
            children=task_data["children"],
            parent=task_data["parent"],
            error=task_data["error"],
            retry_count=task_data["retry_count"],
            max_retries=task_data["max_retries"],
            metadata=task_data["metadata"],
            tags=set(task_data["tags"])
        )
        
        # Set strategy
        if task_data["assigned_strategy"]:
            task.assigned_strategy = ReasoningStrategy(task_data["assigned_strategy"])
        
        # Set times
        if task_data["start_time"]:
            task.start_time = datetime.fromisoformat(task_data["start_time"])
        if task_data["end_time"]:
            task.end_time = datetime.fromisoformat(task_data["end_time"])
        
        # Deserialize dependencies
        for dep_data in task_data["dependencies"]:
            dep = TaskDependency(
                task_id=dep_data["task_id"],
                dependency_type=dep_data["dependency_type"],
                condition=dep_data["condition"],
                timeout=dep_data["timeout"]
            )
            task.dependencies.append(dep)
        
        # Deserialize constraints
        if "constraints" in task_data and task_data["constraints"]:
            constraints_data = task_data["constraints"]
            task.constraints = TaskConstraint(
                max_time=constraints_data["max_time"],
                max_cost=constraints_data["max_cost"],
                max_memory=constraints_data["max_memory"],
                required_capabilities=constraints_data["required_capabilities"],
                excluded_strategies=[ReasoningStrategy(s) for s in constraints_data["excluded_strategies"]],
                minimum_confidence=constraints_data["minimum_confidence"]
            )
        
        # Note: Request and result are not fully restored to avoid complexity
        # They can be reconstructed if needed during execution
        
        return task
    
    async def _cleanup_old_checkpoints(self, plan_id: str) -> None:
        """Clean up old checkpoints for a plan."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get checkpoints to delete (keep only max_checkpoints_per_plan)
                cursor = conn.execute("""
                    SELECT id, file_path FROM checkpoints 
                    WHERE plan_id = ? 
                    ORDER BY created_at DESC 
                    OFFSET ?
                """, (plan_id, self.max_checkpoints_per_plan))
                
                old_checkpoints = cursor.fetchall()
                
                for checkpoint_id, file_path in old_checkpoints:
                    # Delete file
                    file_path = Path(file_path)
                    if file_path.exists():
                        file_path.unlink()
                    
                    # Delete from database
                    conn.execute("DELETE FROM checkpoints WHERE id = ?", (checkpoint_id,))
                
                if old_checkpoints:
                    logger.debug(f"Cleaned up {len(old_checkpoints)} old checkpoints for plan {plan_id}")
                    
        except Exception as e:
            logger.warning(f"Failed to clean up old checkpoints: {e}")
    
    async def start_auto_checkpointing(self, planner) -> None:
        """Start automatic checkpointing for active plans."""
        
        if self.auto_checkpoint_task:
            return
        
        async def auto_checkpoint_loop():
            while True:
                try:
                    await asyncio.sleep(self.auto_checkpoint_interval)
                    
                    # Checkpoint all active plans
                    for plan_id, plan in planner.active_plans.items():
                        await self.save_checkpoint(plan, "auto")
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Auto-checkpoint error: {e}")
        
        self.auto_checkpoint_task = asyncio.create_task(auto_checkpoint_loop())
        logger.info("Started auto-checkpointing")
    
    async def stop_auto_checkpointing(self) -> None:
        """Stop automatic checkpointing."""
        
        if self.auto_checkpoint_task:
            self.auto_checkpoint_task.cancel()
            try:
                await self.auto_checkpoint_task
            except asyncio.CancelledError:
                pass
            self.auto_checkpoint_task = None
            logger.info("Stopped auto-checkpointing")
    
    def list_checkpoints(
        self,
        plan_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        
        with sqlite3.connect(self.db_path) as conn:
            if plan_id:
                cursor = conn.execute("""
                    SELECT id, plan_id, checkpoint_type, created_at, file_size, metadata
                    FROM checkpoints 
                    WHERE plan_id = ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (plan_id, limit))
            else:
                cursor = conn.execute("""
                    SELECT id, plan_id, checkpoint_type, created_at, file_size, metadata
                    FROM checkpoints 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
            
            checkpoints = []
            for row in cursor.fetchall():
                checkpoint_id, plan_id, checkpoint_type, created_at, file_size, metadata_json = row
                checkpoints.append({
                    "id": checkpoint_id,
                    "plan_id": plan_id,
                    "type": checkpoint_type,
                    "created_at": created_at,
                    "file_size": file_size,
                    "metadata": json.loads(metadata_json) if metadata_json else {}
                })
            
            return checkpoints
    
    def list_plans(self) -> List[Dict[str, Any]]:
        """List all plans with metadata."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT plan_id, name, status, created_at, last_checkpoint, 
                       total_tasks, completed_tasks, metadata
                FROM plan_metadata 
                ORDER BY created_at DESC
            """)
            
            plans = []
            for row in cursor.fetchall():
                (plan_id, name, status, created_at, last_checkpoint,
                 total_tasks, completed_tasks, metadata_json) = row
                
                plans.append({
                    "id": plan_id,
                    "name": name,
                    "status": status,
                    "created_at": created_at,
                    "last_checkpoint": last_checkpoint,
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "completion_rate": completed_tasks / max(total_tasks, 1),
                    "metadata": json.loads(metadata_json) if metadata_json else {}
                })
            
            return plans
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get file path
                cursor = conn.execute("SELECT file_path FROM checkpoints WHERE id = ?", (checkpoint_id,))
                row = cursor.fetchone()
                
                if not row:
                    return False
                
                file_path = Path(row[0])
                
                # Delete file
                if file_path.exists():
                    file_path.unlink()
                
                # Delete from database
                conn.execute("DELETE FROM checkpoints WHERE id = ?", (checkpoint_id,))
            
            logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    async def delete_plan_checkpoints(self, plan_id: str) -> int:
        """Delete all checkpoints for a plan."""
        
        try:
            deleted_count = 0
            
            with sqlite3.connect(self.db_path) as conn:
                # Get all checkpoints for the plan
                cursor = conn.execute("SELECT id, file_path FROM checkpoints WHERE plan_id = ?", (plan_id,))
                checkpoints = cursor.fetchall()
                
                for checkpoint_id, file_path in checkpoints:
                    file_path = Path(file_path)
                    if file_path.exists():
                        file_path.unlink()
                    deleted_count += 1
                
                # Delete from database
                conn.execute("DELETE FROM checkpoints WHERE plan_id = ?", (plan_id,))
                conn.execute("DELETE FROM plan_metadata WHERE plan_id = ?", (plan_id,))
            
            logger.info(f"Deleted {deleted_count} checkpoints for plan {plan_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoints for plan {plan_id}: {e}")
            return 0
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get checkpoint statistics."""
        
        with sqlite3.connect(self.db_path) as conn:
            # Total checkpoints
            cursor = conn.execute("SELECT COUNT(*) FROM checkpoints")
            total_checkpoints = cursor.fetchone()[0]
            
            # Total file size
            cursor = conn.execute("SELECT SUM(file_size) FROM checkpoints")
            total_size = cursor.fetchone()[0] or 0
            
            # Checkpoints by type
            cursor = conn.execute("""
                SELECT checkpoint_type, COUNT(*) 
                FROM checkpoints 
                GROUP BY checkpoint_type
            """)
            by_type = dict(cursor.fetchall())
            
            # Plans with checkpoints
            cursor = conn.execute("SELECT COUNT(DISTINCT plan_id) FROM checkpoints")
            plans_with_checkpoints = cursor.fetchone()[0]
            
            # Recent activity (last 24 hours)
            cursor = conn.execute("""
                SELECT COUNT(*) FROM checkpoints 
                WHERE created_at > datetime('now', '-1 day')
            """)
            recent_checkpoints = cursor.fetchone()[0]
        
        return {
            "total_checkpoints": total_checkpoints,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "checkpoints_by_type": by_type,
            "plans_with_checkpoints": plans_with_checkpoints,
            "recent_checkpoints_24h": recent_checkpoints,
            "compression_enabled": self.enable_compression,
            "auto_checkpoint_interval": self.auto_checkpoint_interval,
            "max_checkpoints_per_plan": self.max_checkpoints_per_plan
        }
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.stop_auto_checkpointing()
        logger.info("CheckpointManager closed")