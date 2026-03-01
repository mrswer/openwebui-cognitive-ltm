"""
title: Cognitive LTM Filter (Auto-Adaptive Memory)
author: mr.swer
description: A hybrid Long-Term Memory filter for Open WebUI. Extracts, tags, and safely consolidates memories asynchronously.
version: 0.0.5
"""

import json
import logging
import re
import uuid
import asyncio
import aiohttp
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

# Open WebUI commonly provides access to chromadb in its environment
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
            texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
            return " ".join(texts)
        return ""

    async def _call_llm_extractor(self, text: str) -> Optional[Dict[str, Any]]:
        """Asynchronously calls the background LLM to extract JSON metadata."""
        safe_text = json.dumps(text)
        prompt = f"""
        You are a cognitive memory extractor for a Life Journal.
        Analyze the user's message enclosed in <message> tags. 
        Extract significant life facts, emotional states, or preferences.
        If it's just conversational noise or not worth remembering long-term, set "is_important" to false.
        
        Strictly respond with a JSON object. Format:
        {{
            "is_important": true,
            "tag": "EMOTION" | "FACT" | "PREFERENCE" | "RELATIONSHIP",
            "content": "A concise summary of the memory."
        }}

        <message>
        {safe_text}
        </message>
        """

        payload = {
            "model": self.valves.extractor_model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        # Increased timeout to 30s to handle heavy load on local GPUs
        timeout = aiohttp.ClientTimeout(total=30)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{self.valves.ollama_base_url}/api/generate", json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    result_text = data.get("response", "")
            
            # Non-greedy regex fallback for dirty JSON
            json_match = re.search(r'\{.*?\}', result_text, re.DOTALL)
            if json_match:
                try:
                    parsed_data = json.loads(json_match.group(0))
                    
                    # We only proceed if the LLM deemed it important
                    if parsed_data.get("is_important") is True:
                        if "tag" in parsed_data and "content" in parsed_data:
                            return parsed_data
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in LLM output: {e}")
                    return None
            return None

        except aiohttp.ClientError as e:
            logger.error(f"Async LLM Extractor network error: {e}")
            return None
        except asyncio.TimeoutError:
            logger.error("Async LLM Extractor timeout: The model took too long to respond.")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in _call_llm_extractor: {e}")
            return None

    def _consolidate_memory(self, memory_data: Dict[str, Any]) -> None:
        """
        Synchronous method to evolve existing memories or create new ones.
        Must be called via asyncio.to_thread to avoid blocking the event loop.
        """
        if not self._collection:
            return

        new_content = memory_data.get("content", "")
        tag = memory_data.get("tag", "MEMORY")

        try:
            results = self._collection.query(
                query_texts=[new_content],
                n_results=1,
                where={"tag": tag}
            )

            # Robust check for empty results or IndexError
            has_valid_distance = (
                results and 
                "distances" in results and 
                results["distances"] and 
                len(results["distances"][0]) > 0
            )

            # Evolutionary Consolidation: Append instead of overwrite
            if has_valid_distance and results["distances"][0][0] <= self.valves.consolidation_distance_threshold:
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

        last_user_text = self._extract_text_from_content(last_user_msg_raw)
        if not last_user_text.strip():
            return body

        try:
            # Running synchronous DB query in a separate thread
            results = await asyncio.to_thread(
                self._collection.query,
                query_texts=[last_user_text],
                n_results=self.valves.max_memories_injected
            )

            injected_memories = []
            if results and results.get("distances") and results.get("documents"):
                for i, distance in enumerate(results["distances"][0]):
                    if distance <= self.valves.retrieval_distance_threshold:
                        tag = results["metadatas"][0][i].get("tag", "MEMORY")
                        doc = results["documents"][0][i]
                        injected_memories.append(f"[{tag}] {doc}")

            if injected_memories:
                memory_context = "\n".join(injected_memories)
                system_injection = f"\n\n--- RELEVANT PAST MEMORIES ---\n{memory_context}\n------------------------------\n"
                
                if messages[0]["role"] == "system":
                    current_system = self._extract_text_from_content(messages[0]["content"])
                    messages[0]["content"] = current_system + system_injection
                else:
                    messages.insert(0, {"role": "system", "content": system_injection})
                
                logger.info(f"Injected {len(injected_memories)} memories.")

        except Exception as e:
            logger.error(f"Error during inlet retrieval: {e}")

        return body

    async def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """Extraction Phase (Async): Analyzes user input in the background."""
        messages = body.get("messages", [])
        if not messages:
            return body

        last_user_msg_raw = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
        if not last_user_msg_raw:
            return body

        last_user_text = self._extract_text_from_content(last_user_msg_raw)
        if not last_user_text.strip():
            return body

        # 1. Async LLM extraction (non-blocking)
        memory_data = await self._call_llm_extractor(last_user_text)
        
        # 2. Vector DB Consolidation
        if memory_data:
            # Delegate the synchronous ChromaDB write operation to a separate thread
            # This completely frees the async event loop during DB operations
            await asyncio.to_thread(self._consolidate_memory, memory_data)

        return body
