"""
Advanced session management for the ReasonIt CLI.

This module provides sophisticated session management capabilities including
session persistence, context switching, and collaborative reasoning sessions.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
import sqlite3
import uuid

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

from models import ReasoningRequest, ReasoningResult, ReasoningStrategy, ContextVariant

console = Console()


@dataclass
class SessionMessage:
    """A message in a reasoning session."""
    
    id: str
    session_id: str
    timestamp: datetime
    message_type: str  # "user", "assistant", "system"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningSession:
    """A complete reasoning session."""
    
    id: str
    name: str
    created_at: datetime
    last_active: datetime
    
    # Session configuration
    default_strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    context_variant: ContextVariant = ContextVariant.STANDARD
    use_tools: bool = True
    max_cost_per_query: float = 0.10
    confidence_threshold: float = 0.8
    
    # Session state
    messages: List[SessionMessage] = field(default_factory=list)
    total_queries: int = 0
    total_cost: float = 0.0
    total_time: float = 0.0
    success_count: int = 0
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """Manages reasoning sessions with persistence and advanced features."""
    
    def __init__(self, sessions_dir: str = "./sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        
        self.db_path = self.sessions_dir / "sessions.db"
        self.current_session: Optional[ReasoningSession] = None
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for session persistence."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_active TIMESTAMP NOT NULL,
                    default_strategy TEXT,
                    context_variant TEXT,
                    use_tools BOOLEAN,
                    max_cost_per_query REAL,
                    confidence_threshold REAL,
                    total_queries INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    total_time REAL DEFAULT 0.0,
                    success_count INTEGER DEFAULT 0,
                    tags TEXT,
                    description TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON session_messages(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON session_messages(timestamp)")
    
    def create_session(
        self,
        name: Optional[str] = None,
        description: str = "",
        tags: Optional[Set[str]] = None
    ) -> ReasoningSession:
        """Create a new reasoning session."""
        
        session_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        if not name:
            name = f"Session {timestamp.strftime('%Y-%m-%d %H:%M')}"
        
        session = ReasoningSession(
            id=session_id,
            name=name,
            created_at=timestamp,
            last_active=timestamp,
            description=description,
            tags=tags or set()
        )
        
        self._save_session(session)
        
        console.print(f"[green]✅ Created new session: {name} ({session_id[:8]})[/green]")
        return session
    
    def _save_session(self, session: ReasoningSession):
        """Save session to database."""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sessions 
                (id, name, created_at, last_active, default_strategy, context_variant,
                 use_tools, max_cost_per_query, confidence_threshold, total_queries,
                 total_cost, total_time, success_count, tags, description, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.id,
                session.name,
                session.created_at,
                session.last_active,
                session.default_strategy.value,
                session.context_variant.value,
                session.use_tools,
                session.max_cost_per_query,
                session.confidence_threshold,
                session.total_queries,
                session.total_cost,
                session.total_time,
                session.success_count,
                json.dumps(list(session.tags)),
                session.description,
                json.dumps(session.metadata)
            ))
    
    def load_session(self, session_id: str) -> Optional[ReasoningSession]:
        """Load session from database."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, name, created_at, last_active, default_strategy, context_variant,
                       use_tools, max_cost_per_query, confidence_threshold, total_queries,
                       total_cost, total_time, success_count, tags, description, metadata
                FROM sessions WHERE id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Parse row data
            (id, name, created_at, last_active, default_strategy, context_variant,
             use_tools, max_cost_per_query, confidence_threshold, total_queries,
             total_cost, total_time, success_count, tags_json, description, metadata_json) = row
            
            session = ReasoningSession(
                id=id,
                name=name,
                created_at=datetime.fromisoformat(created_at),
                last_active=datetime.fromisoformat(last_active),
                default_strategy=ReasoningStrategy(default_strategy),
                context_variant=ContextVariant(context_variant),
                use_tools=use_tools,
                max_cost_per_query=max_cost_per_query,
                confidence_threshold=confidence_threshold,
                total_queries=total_queries,
                total_cost=total_cost,
                total_time=total_time,
                success_count=success_count,
                tags=set(json.loads(tags_json)) if tags_json else set(),
                description=description,
                metadata=json.loads(metadata_json) if metadata_json else {}
            )
            
            # Load messages
            session.messages = self._load_session_messages(session_id)
            
            return session
    
    def _load_session_messages(self, session_id: str) -> List[SessionMessage]:
        """Load messages for a session."""
        
        messages = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, session_id, timestamp, message_type, content, metadata
                FROM session_messages 
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,))
            
            for row in cursor.fetchall():
                id, session_id, timestamp, message_type, content, metadata_json = row
                
                message = SessionMessage(
                    id=id,
                    session_id=session_id,
                    timestamp=datetime.fromisoformat(timestamp),
                    message_type=message_type,
                    content=content,
                    metadata=json.loads(metadata_json) if metadata_json else {}
                )
                
                messages.append(message)
        
        return messages
    
    def list_sessions(
        self,
        limit: int = 20,
        search: Optional[str] = None,
        tags: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """List sessions with optional filtering."""
        
        query = """
            SELECT id, name, created_at, last_active, total_queries, total_cost,
                   success_count, tags, description
            FROM sessions
        """
        
        conditions = []
        params = []
        
        if search:
            conditions.append("(name LIKE ? OR description LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])
        
        if tags:
            # Simple tag filtering - could be improved for complex queries
            tag_conditions = [f"tags LIKE ?" for _ in tags]
            conditions.append(f"({' OR '.join(tag_conditions)})")
            params.extend([f"%{tag}%" for tag in tags])
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY last_active DESC LIMIT ?"
        params.append(limit)
        
        sessions = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            
            for row in cursor.fetchall():
                (id, name, created_at, last_active, total_queries, total_cost,
                 success_count, tags_json, description) = row
                
                sessions.append({
                    "id": id,
                    "name": name,
                    "created_at": created_at,
                    "last_active": last_active,
                    "total_queries": total_queries,
                    "total_cost": total_cost,
                    "success_count": success_count,
                    "success_rate": success_count / max(total_queries, 1),
                    "tags": json.loads(tags_json) if tags_json else [],
                    "description": description
                })
        
        return sessions
    
    def switch_session(self, session_id: str) -> bool:
        """Switch to a different session."""
        
        session = self.load_session(session_id)
        if not session:
            console.print(f"[red]Session {session_id} not found[/red]")
            return False
        
        # Save current session if active
        if self.current_session:
            self._save_session(self.current_session)
        
        self.current_session = session
        self.current_session.last_active = datetime.now()
        self._save_session(self.current_session)
        
        console.print(f"[green]✅ Switched to session: {session.name}[/green]")
        return True
    
    def add_message(
        self,
        content: str,
        message_type: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to the current session."""
        
        if not self.current_session:
            console.print("[red]No active session[/red]")
            return
        
        message = SessionMessage(
            id=str(uuid.uuid4()),
            session_id=self.current_session.id,
            timestamp=datetime.now(),
            message_type=message_type,
            content=content,
            metadata=metadata or {}
        )
        
        self.current_session.messages.append(message)
        self.current_session.last_active = datetime.now()
        
        # Save message to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO session_messages 
                (id, session_id, timestamp, message_type, content, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                message.id,
                message.session_id,
                message.timestamp,
                message.message_type,
                message.content,
                json.dumps(message.metadata)
            ))
        
        # Update session stats
        self._save_session(self.current_session)
    
    def update_session_stats(
        self,
        cost: float = 0.0,
        time: float = 0.0,
        success: bool = True
    ):
        """Update session statistics."""
        
        if not self.current_session:
            return
        
        self.current_session.total_queries += 1
        self.current_session.total_cost += cost
        self.current_session.total_time += time
        
        if success:
            self.current_session.success_count += 1
        
        self.current_session.last_active = datetime.now()
        self._save_session(self.current_session)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        
        if self.current_session and self.current_session.id == session_id:
            self.current_session = None
        
        with sqlite3.connect(self.db_path) as conn:
            # Delete messages first (CASCADE should handle this, but being explicit)
            conn.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
            
            # Delete session
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            
            if cursor.rowcount > 0:
                console.print(f"[green]✅ Deleted session {session_id[:8]}[/green]")
                return True
            else:
                console.print(f"[red]Session {session_id} not found[/red]")
                return False
    
    def display_sessions(self, sessions: Optional[List[Dict[str, Any]]] = None):
        """Display sessions in a rich table."""
        
        if sessions is None:
            sessions = self.list_sessions()
        
        if not sessions:
            console.print("[yellow]No sessions found[/yellow]")
            return
        
        sessions_table = Table(title="Reasoning Sessions")
        sessions_table.add_column("ID", style="cyan", max_width=10)
        sessions_table.add_column("Name", style="white", max_width=25)
        sessions_table.add_column("Last Active", style="green")
        sessions_table.add_column("Queries", style="yellow")
        sessions_table.add_column("Success Rate", style="blue")
        sessions_table.add_column("Cost", style="magenta")
        sessions_table.add_column("Tags", style="dim")
        
        for session in sessions:
            # Format last active time
            last_active = datetime.fromisoformat(session["last_active"])
            time_diff = datetime.now() - last_active
            
            if time_diff < timedelta(hours=1):
                last_active_str = f"{int(time_diff.total_seconds() // 60)}m ago"
            elif time_diff < timedelta(days=1):
                last_active_str = f"{int(time_diff.total_seconds() // 3600)}h ago"
            else:
                last_active_str = f"{int(time_diff.days)}d ago"
            
            # Format success rate
            success_rate = f"{session['success_rate']:.1%}"
            
            # Format cost
            cost_str = f"${session['total_cost']:.3f}"
            
            # Format tags
            tags_str = ", ".join(session["tags"][:3])
            if len(session["tags"]) > 3:
                tags_str += "..."
            
            sessions_table.add_row(
                session["id"][:8],
                session["name"],
                last_active_str,
                str(session["total_queries"]),
                success_rate,
                cost_str,
                tags_str or "-"
            )
        
        console.print(sessions_table)
    
    def display_session_details(self, session_id: Optional[str] = None):
        """Display detailed information about a session."""
        
        if session_id:
            session = self.load_session(session_id)
        else:
            session = self.current_session
        
        if not session:
            console.print("[red]No session specified or loaded[/red]")
            return
        
        # Session info panel
        info_text = f"""[bold]Session ID:[/bold] {session.id}
[bold]Name:[/bold] {session.name}
[bold]Description:[/bold] {session.description or 'No description'}
[bold]Created:[/bold] {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}
[bold]Last Active:[/bold] {session.last_active.strftime('%Y-%m-%d %H:%M:%S')}
[bold]Tags:[/bold] {', '.join(session.tags) or 'None'}"""
        
        info_panel = Panel(info_text, title="Session Information", border_style="blue")
        
        # Stats panel
        stats_text = f"""[bold]Total Queries:[/bold] {session.total_queries}
[bold]Successful:[/bold] {session.success_count}
[bold]Success Rate:[/bold] {session.success_count / max(session.total_queries, 1):.1%}
[bold]Total Cost:[/bold] ${session.total_cost:.4f}
[bold]Total Time:[/bold] {session.total_time:.2f}s
[bold]Avg Cost/Query:[/bold] ${session.total_cost / max(session.total_queries, 1):.4f}
[bold]Messages:[/bold] {len(session.messages)}"""
        
        stats_panel = Panel(stats_text, title="Statistics", border_style="green")
        
        # Config panel
        config_text = f"""[bold]Default Strategy:[/bold] {session.default_strategy.value}
[bold]Context Variant:[/bold] {session.context_variant.value}
[bold]Use Tools:[/bold] {'Yes' if session.use_tools else 'No'}
[bold]Max Cost/Query:[/bold] ${session.max_cost_per_query:.3f}
[bold]Confidence Threshold:[/bold] {session.confidence_threshold:.1%}"""
        
        config_panel = Panel(config_text, title="Configuration", border_style="yellow")
        
        # Display all panels
        console.print(info_panel)
        console.print(stats_panel)
        console.print(config_panel)
        
        # Show recent messages if any
        if session.messages:
            recent_messages = session.messages[-5:]  # Last 5 messages
            
            console.print("\n[bold]Recent Messages:[/bold]")
            for msg in recent_messages:
                timestamp = msg.timestamp.strftime('%H:%M:%S')
                style = {
                    'user': 'cyan',
                    'assistant': 'green',
                    'system': 'yellow'
                }.get(msg.message_type, 'white')
                
                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                console.print(f"[{style}]{timestamp} {msg.message_type.upper()}:[/{style}] {content}")
    
    def export_session(self, session_id: str, format: str = "json") -> str:
        """Export session data."""
        
        session = self.load_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session.name.replace(' ', '_')}_{timestamp}.{format}"
        
        export_data = {
            "session": {
                "id": session.id,
                "name": session.name,
                "description": session.description,
                "created_at": session.created_at.isoformat(),
                "last_active": session.last_active.isoformat(),
                "total_queries": session.total_queries,
                "total_cost": session.total_cost,
                "total_time": session.total_time,
                "success_count": session.success_count,
                "tags": list(session.tags),
                "metadata": session.metadata
            },
            "messages": [
                {
                    "id": msg.id,
                    "timestamp": msg.timestamp.isoformat(),
                    "type": msg.message_type,
                    "content": msg.content,
                    "metadata": msg.metadata
                }
                for msg in session.messages
            ],
            "export_timestamp": datetime.now().isoformat()
        }
        
        filepath = self.sessions_dir / filename
        
        with open(filepath, 'w') as f:
            if format == "json":
                json.dump(export_data, f, indent=2)
            else:
                f.write(str(export_data))
        
        return str(filepath)
    
    def get_session_context(self, max_messages: int = 10) -> str:
        """Get recent session context for reasoning."""
        
        if not self.current_session or not self.current_session.messages:
            return ""
        
        recent_messages = self.current_session.messages[-max_messages:]
        
        context_parts = []
        for msg in recent_messages:
            if msg.message_type in ["user", "assistant"]:
                context_parts.append(f"{msg.message_type.upper()}: {msg.content}")
        
        return "\n".join(context_parts)
    
    async def interactive_session_management(self):
        """Interactive session management."""
        
        console.print(Panel.fit(
            "[bold green]Session Management[/bold green]\n"
            "Manage your reasoning sessions, switch between contexts, and view history.",
            title="Session Manager",
            border_style="green"
        ))
        
        while True:
            try:
                # Show current session
                if self.current_session:
                    console.print(f"\n[bold]Current Session:[/bold] {self.current_session.name} ({self.current_session.id[:8]})")
                else:
                    console.print("\n[yellow]No active session[/yellow]")
                
                # Session management menu
                console.print("\n[bold]Session Commands:[/bold]")
                console.print("• [cyan]list[/cyan] - List all sessions")
                console.print("• [cyan]new[/cyan] - Create new session")
                console.print("• [cyan]switch <id>[/cyan] - Switch to session")
                console.print("• [cyan]details[/cyan] - Show current session details")
                console.print("• [cyan]delete <id>[/cyan] - Delete session")
                console.print("• [cyan]export <id>[/cyan] - Export session")
                console.print("• [cyan]search <term>[/cyan] - Search sessions")
                console.print("• [cyan]exit[/cyan] - Exit session manager")
                
                command = Prompt.ask("\n[bold cyan]❯[/bold cyan] Session command").strip().lower()
                
                if command == "exit":
                    break
                elif command == "list":
                    self.display_sessions()
                elif command == "new":
                    await self._interactive_create_session()
                elif command.startswith("switch "):
                    session_id = command.split(" ", 1)[1]
                    self.switch_session(session_id)
                elif command == "details":
                    self.display_session_details()
                elif command.startswith("delete "):
                    session_id = command.split(" ", 1)[1]
                    if Confirm.ask(f"[red]Delete session {session_id}?[/red]"):
                        self.delete_session(session_id)
                elif command.startswith("export "):
                    session_id = command.split(" ", 1)[1]
                    filepath = self.export_session(session_id)
                    console.print(f"[green]✅ Exported to {filepath}[/green]")
                elif command.startswith("search "):
                    search_term = command.split(" ", 1)[1]
                    sessions = self.list_sessions(search=search_term)
                    self.display_sessions(sessions)
                else:
                    console.print("[red]Unknown command. Type 'exit' to quit.[/red]")
                    
            except KeyboardInterrupt:
                if Confirm.ask("\n[yellow]Exit session manager?[/yellow]"):
                    break
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
    
    async def _interactive_create_session(self):
        """Interactive session creation."""
        
        name = Prompt.ask("[cyan]Session name[/cyan]", default=f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        description = Prompt.ask("[cyan]Description[/cyan] (optional)", default="")
        
        # Tags
        tags_input = Prompt.ask("[cyan]Tags[/cyan] (comma-separated, optional)", default="")
        tags = set(tag.strip() for tag in tags_input.split(",") if tag.strip()) if tags_input else set()
        
        session = self.create_session(name, description, tags)
        
        if Confirm.ask("[cyan]Switch to this session?[/cyan]", default=True):
            self.current_session = session
            console.print(f"[green]✅ Switched to new session: {name}[/green]")