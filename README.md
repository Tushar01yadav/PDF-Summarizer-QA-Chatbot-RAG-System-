# 📄 PDF Summarizer RAG System with PDF Image-to-Text Extraction

## 🔍 Overview
This repository implements a **Retrieval-Augmented Generation (RAG) system** powered by **LLMs (Large Language Models)**.  
Unlike traditional PDF text extraction (which often fails on scanned or image-based PDFs), this system:

1. Converts **PDF pages into images**  
2. Generates **captions (OCR + image captioning)** to accurately extract text  
3. Creates **embeddings** of extracted content  
4. Embeds the **user query** for semantic similarity search  
5. Retrieves **top-k most relevant results**  
6. Uses the **LLM to generate an answer** with citations to retrieved sources  

This ensures robust **question answering** even with scanned/image-heavy PDFs.

---

## ✨ Features
- 📑 **PDF → Image → Caption** for reliable text extraction  
- 🧠 **Semantic Search** with vector embeddings  
- 💬 **LLM-powered Q&A** with context  
- 🔗 **Source citations** for transparency  
- ⚡ **Modular design** (swap embedding models, vector stores, or LLMs)

---

