"""
Custom LangGraph checkpointer using SQLAlchemy for persistence.

This module implements a custom BaseCheckpointSaver that integrates with the
existing Geist database infrastructure for agent state persistence.
"""
import json
import logging
from typing import Optional, Dict, Any, Iterator, Tuple
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from app.models.database.database import Session
from app.models.database.agent import Agent
from sqlalchemy.orm import Session as SessionType

logger = logging.getLogger(__name__)


class SQLAlchemyCheckpointSaver(BaseCheckpointSaver):
    """
    Custom checkpoint saver that persists agent state to PostgreSQL using SQLAlchemy.

    This checkpointer integrates with the existing Agent model in the database,
    storing checkpoint data as JSON in dedicated fields.
    """

    def __init__(self):
        """Initialize the checkpoint saver."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

    def _get_session(self) -> SessionType:
        """Get a database session."""
        return Session()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        """
        Save a checkpoint to the database.

        Args:
            config: Configuration for the checkpoint (contains thread_id)
            checkpoint: The checkpoint data to save
            metadata: Metadata about the checkpoint

        Returns:
            Updated configuration with checkpoint_id
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                raise ValueError("thread_id is required in config")

            session = self._get_session()

            # Find or create agent record
            agent = session.query(Agent).filter_by(agent_identifier=thread_id).first()

            if not agent:
                self.logger.info(f"Creating new agent record for thread_id: {thread_id}")
                agent = Agent(
                    agent_identifier=thread_id,
                    world_context=json.dumps([]),
                    task_context=json.dumps([]),
                    execution_context=json.dumps([])
                )
                session.add(agent)

            # Serialize checkpoint data
            checkpoint_data = {
                "v": checkpoint.get("v", 1),
                "ts": checkpoint.get("ts", datetime.now().isoformat()),
                "id": checkpoint.get("id"),
                "channel_values": checkpoint.get("channel_values", {}),
                "channel_versions": checkpoint.get("channel_versions", {}),
                "versions_seen": checkpoint.get("versions_seen", {}),
            }

            # Store checkpoint and metadata in agent fields
            agent.checkpoint_data = json.dumps(checkpoint_data)
            agent.checkpoint_metadata = json.dumps({
                "source": metadata.get("source", "unknown"),
                "step": metadata.get("step", 0),
                "writes": metadata.get("writes", {}),
            })

            # Update context fields from checkpoint state if available
            channel_values = checkpoint.get("channel_values", {})
            if "world_context" in channel_values:
                agent.world_context = json.dumps(channel_values["world_context"])
            if "task_context" in channel_values:
                agent.task_context = json.dumps(channel_values["task_context"])
            if "execution_context" in channel_values:
                agent.execution_context = json.dumps(channel_values["execution_context"])

            session.commit()

            # Return config with checkpoint_id
            checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
            checkpoint_id = checkpoint.get("id")

            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """
        Store intermediate writes (state updates) for a checkpoint.

        Args:
            config: Configuration for the checkpoint
            writes: List of (channel, value) tuples to write
            task_id: Task identifier
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                return

            session = self._get_session()
            agent = session.query(Agent).filter_by(agent_identifier=thread_id).first()

            if agent:
                # Store writes as pending updates
                if not hasattr(agent, 'pending_writes') or not agent.pending_writes:
                    agent.pending_writes = json.dumps([])

                pending = json.loads(agent.pending_writes)
                pending.append({
                    "task_id": task_id,
                    "writes": [(channel, value) for channel, value in writes],
                    "timestamp": datetime.now().isoformat()
                })
                agent.pending_writes = json.dumps(pending)
                session.commit()

        except Exception as e:
            self.logger.error(f"Failed to save writes: {e}")
            session.rollback()
        finally:
            session.close()

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """
        Retrieve a checkpoint from the database.

        Args:
            config: Configuration specifying which checkpoint to retrieve

        Returns:
            CheckpointTuple containing checkpoint data and metadata, or None if not found
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                return None

            session = self._get_session()
            agent = session.query(Agent).filter_by(agent_identifier=thread_id).first()

            if not agent or not hasattr(agent, 'checkpoint_data') or not agent.checkpoint_data:
                return None

            # Deserialize checkpoint data
            checkpoint_data = json.loads(agent.checkpoint_data)
            checkpoint = Checkpoint(
                v=checkpoint_data.get("v", 1),
                ts=checkpoint_data.get("ts"),
                id=checkpoint_data.get("id"),
                channel_values=checkpoint_data.get("channel_values", {}),
                channel_versions=checkpoint_data.get("channel_versions", {}),
                versions_seen=checkpoint_data.get("versions_seen", {}),
            )

            # Deserialize metadata
            metadata_dict = {}
            if hasattr(agent, 'checkpoint_metadata') and agent.checkpoint_metadata:
                metadata_dict = json.loads(agent.checkpoint_metadata)

            metadata = CheckpointMetadata(
                source=metadata_dict.get("source", "unknown"),
                step=metadata_dict.get("step", 0),
                writes=metadata_dict.get("writes", {}),
            )

            # Get pending writes
            pending_writes = []
            if hasattr(agent, 'pending_writes') and agent.pending_writes:
                pending_data = json.loads(agent.pending_writes)
                for item in pending_data:
                    pending_writes.extend([(w[0], w[1]) for w in item.get("writes", [])])

            parent_config = None  # Could be enhanced to support checkpoint hierarchy

            return CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=pending_writes,
            )

        except Exception as e:
            self.logger.error(f"Failed to retrieve checkpoint: {e}")
            return None
        finally:
            session.close()

    def list(
        self,
        config: Optional[RunnableConfig] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """
        List checkpoints from the database.

        Args:
            config: Optional configuration to filter by
            filter: Additional filters
            before: Only return checkpoints before this config
            limit: Maximum number of checkpoints to return

        Yields:
            CheckpointTuple objects matching the criteria
        """
        try:
            session = self._get_session()
            query = session.query(Agent)

            # Apply filters
            if config and config.get("configurable", {}).get("thread_id"):
                thread_id = config["configurable"]["thread_id"]
                query = query.filter_by(agent_identifier=thread_id)

            if limit:
                query = query.limit(limit)

            # Order by most recent first
            query = query.order_by(Agent.agent_id.desc())

            for agent in query:
                if not hasattr(agent, 'checkpoint_data') or not agent.checkpoint_data:
                    continue

                try:
                    checkpoint_data = json.loads(agent.checkpoint_data)
                    checkpoint = Checkpoint(
                        v=checkpoint_data.get("v", 1),
                        ts=checkpoint_data.get("ts"),
                        id=checkpoint_data.get("id"),
                        channel_values=checkpoint_data.get("channel_values", {}),
                        channel_versions=checkpoint_data.get("channel_versions", {}),
                        versions_seen=checkpoint_data.get("versions_seen", {}),
                    )

                    metadata_dict = {}
                    if hasattr(agent, 'checkpoint_metadata') and agent.checkpoint_metadata:
                        metadata_dict = json.loads(agent.checkpoint_metadata)

                    metadata = CheckpointMetadata(
                        source=metadata_dict.get("source", "unknown"),
                        step=metadata_dict.get("step", 0),
                        writes=metadata_dict.get("writes", {}),
                    )

                    config_tuple = {
                        "configurable": {
                            "thread_id": agent.agent_identifier,
                            "checkpoint_id": checkpoint.get("id"),
                        }
                    }

                    yield CheckpointTuple(
                        config=config_tuple,
                        checkpoint=checkpoint,
                        metadata=metadata,
                        parent_config=None,
                        pending_writes=[],
                    )

                except Exception as e:
                    self.logger.error(f"Failed to deserialize checkpoint for agent {agent.agent_id}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Failed to list checkpoints: {e}")
        finally:
            session.close()
