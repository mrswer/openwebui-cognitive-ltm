"""
title: Hybrid LTM (Auto-Adaptive Memory)
author: mr.swer
description: Long-Term Memory filter combining dynamic tagging (Adaptive) and vector-based consolidation (Auto).
version: 0.0.2
"""

import json
import logging
import re
import requests
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import chromadb
from chromadb.utils import embedding_functions

# Setup structured logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Filter:
    class Valves(BaseModel):
        # API and Model Configuration
        ollama_base_url: str = Field(
            default="http://host.docker.internal:11434",
            description="Base URL for the Ollama instance used for background extraction."
        )
        extractor_model: str = Field(
            default="llama3:8b",
            description="Smaller, fast model used strictly for background tagging and extraction."
        )
        # Memory Database Configuration
        chroma_db_path: str = Field(
            default="./data/hybrid_memory_db",
            description="Local path to store the persistent ChromaDB vectors."
        )
        # Thresholds (Distance: lower means more similar. 0.0 is identical)
        retrieval_distance_threshold: float = Field(
            default=0.25,
            description="Maximum distance to retrieve a memory during chat (Inlet)."
        )
        consolidation_distance_threshold: float = Field(
            default=0.18,
            description="Maximum distance to overwrite/consolidate an existing memory (Outlet)."
        )
        max_memories_injected: int = Field(
            default=3,
            description="Maximum number of memory blocks to inject into the system prompt."
        )

    def __init__(self):
        self.valves = self.Valves()
        self.collection_name = "life_journal_memory"
        self._db_client = None
        self._collection = None
        self._init_db()

    def _init_db(self) -> None:
        """Initializes the persistent ChromaDB client and collection."""
        try:
            self._db_client = chromadb.PersistentClient(path=self.valves.chroma_db_path)
            # Using default MiniLM embedding function (standard, fast, local)
            self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
            self._collection = self._db_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.emb_fn
            )
            logger.info(f"Hybrid LTM Database initialized at {self.valves.chroma_db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")

    def _call_llm_extractor(self, text: str) -> Optional[Dict[str, Any]]:
        """Calls the background LLM to extract JSON metadata and content."""
        prompt = f"""
        You are a cognitive memory extractor for a Life Journal.
        Analyze the user's message. If it contains significant life facts, emotional states, preferences, or relationships, extract it.
        Otherwise, ignore it.
        
        Strictly respond with a JSON object inside a code block. Do not add conversational text.
        Format:
        {{
            "is_important": true/false,
            "tag": "EMOTION" | "FACT" | "PREFERENCE" | "RELATIONSHIP",
            "content": "A concise, third-person summary of the memory."
        }}

        User message: "{text}"
        """

        payload = {
            "model": self.valves.extractor_model,
            "prompt": prompt,
            "stream": False,
            "format": "json" # Forces JSON mode if supported by the model
        }

        try:
            response = requests.post(f"{self.valves.ollama_base_url}/api/generate", json=payload, timeout=15)
            response.raise_for_status()
            result_text = response.json().get("response", "")
            
            # Robust JSON extraction using regex in case the model hallucinates markdown wrappers
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group(0))
                if parsed_data.get("is_important"):
                    return parsed_data
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON output: {e}")
            return None

    def _consolidate_memory(self, memory_data: Dict[str, Any]) -> None:
        """Checks for existing similar memories and updates or inserts accordingly."""
        if not self._collection:
            return

        content = memory_data["content"]
        tag = memory_data["tag"]

        try:
            # Query DB for the most similar existing memory
            results = self._collection.query(
                query_texts=[content],
                n_results=1,
                where={"tag": tag} # Metadata filtering ensures we don't mix FACTS with EMOTIONS
            )

            # If a highly similar memory exists, update it (Consolidation)
            if results["distances"][0] and results["distances"][0][0] <= self.valves.consolidation_distance_threshold:
                doc_id = results["ids"][0][0]
                logger.info(f"Consolidating memory {doc_id} (Distance: {results['distances'][0][0]:.3f})")
                
                self._collection.update(
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[{"tag": tag}]
                )
            # Otherwise, add as a new memory
            else:
                import uuid
                new_id = str(uuid.uuid4())
                logger.info(f"Adding new memory: [{tag}] {content}")
                
                self._collection.add(
                    ids=[new_id],
                    documents=[content],
                    metadatas=[{"tag": tag}]
                )

        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Retrieval Phase: Intercepts the request before it reaches the main model.
        Injects relevant context from the vector database.
        """
        if not self._collection:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        # Get the latest user message
        last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
        if not last_user_msg:
            return body

        try:
            # Query the database for context
            results = self._collection.query(
                query_texts=[last_user_msg],
                n_results=self.valves.max_memories_injected
            )

            injected_memories = []
            if results["distances"] and results["documents"]:
                for i, distance in enumerate(results["distances"][0]):
                    if distance <= self.valves.retrieval_distance_threshold:
                        tag = results["metadatas"][0][i].get("tag", "MEMORY")
                        doc = results["documents"][0][i]
                        injected_memories.append(f"[{tag}] {doc}")

            # If relevant memories found, inject them into the system prompt
            if injected_memories:
                memory_context = "\n".join(injected_memories)
                system_injection = f"\n\n--- RELEVANT PAST MEMORIES ---\n{memory_context}\n------------------------------\n"
                
                # Check if system message exists, otherwise create one
                if messages[0]["role"] == "system":
                    messages[0]["content"] += system_injection
                else:
                    messages.insert(0, {"role": "system", "content": system_injection})
                
                logger.info(f"Injected {len(injected_memories)} memories into context.")

        except Exception as e:
            logger.error(f"Error during memory retrieval (inlet): {e}")

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Extraction Phase: Runs after the chat interaction.
        Analyzes the user's input and updates the memory database.
        """
        messages = body.get("messages", [])
        if not messages:
            return body

        # Extract the latest user message to evaluate
        last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
        
        if last_user_msg:
            # 1. Background LLM extraction (Adaptive Logic)
            memory_data = self._call_llm_extractor(last_user_msg)
            
            # 2. Vector DB Consolidation (Auto Logic)
            if memory_data:
                self._consolidate_memory(memory_data)

        return body
