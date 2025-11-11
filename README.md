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

Screenshots :
<img width="955" height="410" alt="image" src="https://github.com/user-attachments/assets/22680837-b6c4-4c96-9ef9-506a4a66876c" />
<img width="956" height="412" alt="image" src="https://github.com/user-attachments/assets/337d91e2-a52b-43de-9489-28ccc1d90fb3" />
<img width="954" height="412" alt="image" src="https://github.com/user-attachments/assets/b8b3cb6a-16b6-4783-aebf-9cc2c23f4ae5" />
<img width="955" height="416" alt="image" src="https://github.com/user-attachments/assets/f15e3adc-6bdd-465d-b29f-219bdd073139" />
<img width="959" height="362" alt="image" src="https://github.com/user-attachments/assets/ba4ab9c1-ee9d-4c53-b169-b5d1c6d3d1f2" />





---

## ✨ Features
- 📑 **PDF → Image → Caption** for reliable text extraction  
- 🧠 **Semantic Search** with vector embeddings  
- 💬 **LLM-powered Q&A** with context  
- 🔗 **Source citations** for transparency  
- ⚡ **Modular design** (swap embedding models, vector stores, or LLMs)

---

---
## Author 

- ** Tushar Yadav **
- **Email**: tusharyadav61900@gmail.conm
- **LinkedIn**: [Tushar Yadav](https://www.linkedin.com/in/tushar-yadav-5829bb353/)



