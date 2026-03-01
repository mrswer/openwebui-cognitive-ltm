# Open WebUI Cognitive LTM 🧠

A cognitive, hybrid Long-Term Memory (Auto-Adaptive RAG) filter tailored for Open WebUI. Built specifically for Life Journaling, AI companions, and multi-year conversations, it seamlessly extracts, tags, and safely consolidates memories without context degradation.

## 📖 Overview

Traditional RAG (Retrieval-Augmented Generation) setups suffer from "context bloat" and lossy summarization over long periods. **Cognitive LTM** solves this by combining the best of two worlds:
1. **Adaptive Tagging:** A background LLM extracts only meaningful facts, emotions, and preferences, assigning strict metadata tags (e.g., `[EMOTION]`, `[FACT]`).
2. **Auto-Consolidation:** Uses vector distance thresholds to intelligently merge or append new memories to existing ones, creating an evolving, non-destructive memory timeline.



## ✨ Features

* **Zero-Click Memory Management:** Runs entirely in the background. You just chat natively in Open WebUI.
* **Smart Categorization:** Prevents "memory hallucination" by strictly separating facts from passing emotions using metadata.
* **Evolutionary Consolidation:** Instead of destructively overwriting past memories, it evolves them, preserving the chronology of your Life Journal.
* **Fully Asynchronous:** Built with `aiohttp` and FastAPI asynchronous patterns to ensure zero latency impact on your primary chat generation.
* **Multimodal Safe:** Safely handles image and file inputs without crashing the text-extraction pipeline.

## 🛠️ Prerequisites

* **Open WebUI:** Installed and running.
* **Ollama:** (Or any compatible API) for the background extraction model.
* **ChromaDB:** Handled locally within the Open WebUI environment.

## 🚀 Installation

1. Open your Open WebUI instance.
2. Navigate to **Workspace** -> **Functions** -> **(+) Add**.
3. Copy the entire content of `cognitive_ltm_filter.py` and paste it into the code editor.
4. Name it `Cognitive LTM` and assign it as a **Filter**.
5. Click **Save** and enable it for your desired models.

## ⚙️ Configuration (Valves)

You can fine-tune the cognitive engine directly from the Open WebUI interface via the Valves mechanism:

| Valve | Default | Description |
| :--- | :--- | :--- |
| `Extractor Model` | `llama3:8b` | The model used in the background to analyze and tag messages. A fast, quantized 8B model is highly recommended. |
| `Retrieval Distance` | `0.25` | How similar a message must be to past memories to trigger injection into the current context. |
| `Consolidation Distance` | `0.18` | The threshold to decide if a new memory should update an old one or be saved as a completely new entry. |
| `Max Injected Memories` | `3` | Maximum number of memory blocks injected into the system prompt to prevent context window overflow. |

## 🧠 Why is it perfect for Life Journaling?

Standard summarizers destroy the nuance of your daily life. A journal entry like *"I felt so lonely today moving to a new city"* eventually gets compressed into *"User moved in 2026."* **Cognitive LTM** extracts the raw essence, tags it as an `[EMOTION]`, and stores it intact. Years later, when you ask your bot about that time, it retrieves your exact feelings, allowing the AI to respond with genuine empathy and deep historical context.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](#).

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
