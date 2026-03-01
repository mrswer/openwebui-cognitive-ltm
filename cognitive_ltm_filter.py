"""
title: Cognitive LTM Filter (Auto-Adaptive Memory)
author: mr.swer
description: A hybrid Long-Term Memory filter for Open WebUI. Extracts, tags, and safely consolidates memories asynchronously.
version: 0.0.3
"""

import json
import logging
import re
import uuid
import aiohttp
import asyncio
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Open WebUI often runs in environments where chromadb is available
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Filter:
    class Valves(BaseModel):
        ollama_base_url: str = Field(
            default="http://host.docker.internal:11434",
            description="Base URL for the Ollama instance used for background extraction."
        )
        extractor_model: str = Field(
            default="llama3:8b",
            description="Smaller, fast model used strictly for background tagging."
        )
        chroma_db_path: str = Field(
            default="./data/cognitive_ltm_db",
            description="Local path to store the persistent ChromaDB vectors."
        )
        retrieval_distance_threshold: float = Field(
            default=0.25,
            description="Maximum distance to retrieve a memory during chat (Inlet)."
        )
        consolidation_distance_threshold: float = Field(
            default=0.18,
            description="Maximum distance to append a new memory to an existing one (Outlet)."
        )
        max_memories_injected: int = Field(
            default=3,
            description="Maximum number of memory blocks to inject into the system prompt."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.collection_name = "cognitive_journal_memory"
        self._db_client = None
        self._collection = None
        self._init_db()

    def _init_db(self) -> None:
        """Initializes the persistent ChromaDB client."""
        try:
            # Note: In multi-worker environments (WEBUI_WORKERS > 1), SQLite might lock.
            # For pure production scale, consider chromadb.HttpClient() instead.
            self._db_client = chromadb.PersistentClient(path=self.valves.chroma_db_path)
            self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
            self._collection = self._db_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.emb_fn
            )
            logger.info(f"Cognitive LTM Database initialized at {self.valves.chroma_db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")

    def _extract_text_from_content(self, content: Any) -> str:
        """Safely extracts text from potentially multimodal messages (e.g., images + text)."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Handles Open WebUI multimodal list structure
            texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
            return " ".join(texts)
        return ""

    async def _call_llm_extractor(self, text: str) -> Optional[Dict[str, Any]]:
        """Asynchronously calls the background LLM to extract JSON metadata."""
        prompt = f"""
        You are a cognitive memory extractor for a Life Journal.
        Analyze the user's message. Extract significant life facts, emotional states, or preferences.
        If it's just conversational noise, set "is_important" to false.
        
        Strictly respond with a JSON object. Format:
        {{
            "is_important": true,
            "tag": "EMOTION" | "FACT" | "PREFERENCE" | "RELATIONSHIP",
            "content": "A concise summary of the memory."
        }}

        User message: "{text}"
        """

        payload = {
            "model": self.valves.extractor_model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        try:
            # Non-blocking HTTP request using aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.valves.ollama_base_url}/api/generate", json=payload, timeout=15) as response:
                    response.raise_for_status()
                    data = await response.json()
                    result_text = data.get("response", "")
            
            # Regex fallback for dirty JSON
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group(0))
                # Validate strict keys to prevent KeyErrors later
                if parsed_data.get("is_important") and "tag" in parsed_data and "content" in parsed_data:
                    return parsed_data
            return None

        except Exception as e:
            logger.error(f"Async LLM Extractor failed: {e}")
            return None

    def _consolidate_memory(self, memory_data: Dict[str, Any]) -> None:
        """Evolves existing memories or creates new ones based on vector similarity."""
        if not self._collection:
            return

        new_content = memory_data["content"]
        tag = memory_data["tag"]

        try:
            results = self._collection.query(
                query_texts=[new_content],
                n_results=1,
                where={"tag": tag}
            )

            # Evolutionary Consolidation: Append instead of overwrite
            if results["distances"] and results["distances"][0] and results["distances"][0][0] <= self.valves.consolidation_distance_threshold:
                doc_id = results["ids"][0][0]
                old_content = results["documents"][0][0]
                
                # Check to avoid duplicating exact same strings
                if new_content.lower() not in old_content.lower():
                    evolved_content = f"{old_content} | Updated: {new_content}"
                    logger.info(f"Evolving memory {doc_id} with new context.")
                    
                    self._collection.update(
                        ids=[doc_id],
                        documents=[evolved_content],
                        metadatas=[{"tag": tag}]
                    )
            else:
                # Add entirely new memory
                new_id = str(uuid.uuid4())
                logger.info(f"Adding new memory: [{tag}] {new_content}")
                
                self._collection.add(
                    ids=[new_id],
                    documents=[new_content],
                    metadatas=[{"tag": tag}]
                )

        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")

    async def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Retrieval Phase (Async): Injects relevant context from the DB."""
        if not self._collection:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        last_user_msg_raw = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
        if not last_user_msg_raw:
            return body

        # Mult
