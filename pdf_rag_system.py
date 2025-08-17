import json
import os
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import time

# Core libraries
try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Install with: pip install PyMuPDF")
    raise
from PIL import Image
import io
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PDFRAGSystem:
    """
    RAG System for PDF processing with image captioning using Mistral
    """
    
    def __init__(self, mistral_api_key: str, output_dir: str = "rag_output"):
        self.mistral_api_key = mistral_api_key
        self.output_dir = output_dir
        self.mistral_api_url = "https://api.mistral.ai/v1/chat/completions"
        
        # Initialize sentence transformer for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage for processed documents
        self.documents_db = []
        
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract embedded images from PDF"""
        doc = fitz.open(pdf_path)
        extracted_images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]  # xref number
                try:
                    # Extract image data
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert to base64
                    image_base64 = self.image_to_base64(image)
                    
                    extracted_images.append({
                        "page_number": page_num + 1,
                        "image_index": img_index,
                        "image": image,
                        "image_base64": image_base64,
                        "extension": image_ext,
                        "size": image.size
                    })
                    
                except Exception as e:
                    print(f"Could not extract image {img_index} from page {page_num + 1}: {e}")
        
        doc.close()
        return extracted_images
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text content from PDF pages"""
        doc = fitz.open(pdf_path)
        text_data = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            if text.strip():  # Only add pages with text content
                text_data.append({
                    "page_number": page_num + 1,
                    "text_content": text.strip()
                })
        
        doc.close()
        return text_data
    def pdf_to_images(self, pdf_path: str, dpi: int = 150) -> List[Image.Image]:
        """Convert PDF pages to images (for pages without extractable content)"""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Convert to image
            mat = fitz.Matrix(dpi/72, dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            
        doc.close()
        return images
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return img_base64
    
    def process_embedded_image(self, image_data: Dict, document_name: str) -> Dict:
        """Process a single embedded image with Mistral"""
        page_num = image_data["page_number"]
        img_index = image_data["image_index"]
        
        print(f"  Processing embedded image {img_index + 1} from page {page_num}...")
        
        # Generate caption with Mistral
        caption = self.generate_caption_with_mistral(
            image_data["image_base64"], 
            page_num, 
            f"embedded image {img_index + 1}"
        )
        
        # Create embeddings
        embedding = self.create_embeddings(caption)
        
        # Save image
        image_path = f"{self.output_dir}/{document_name}_page_{page_num}_img_{img_index + 1}.{image_data['extension']}"
        image_data["image"].save(image_path)
        
        return {
            "content_type": "embedded_image",
            "page_number": page_num,
            "image_index": img_index,
            "image_path": image_path,
            "image_size": image_data["size"],
            "caption": caption,
            "embedding": embedding.tolist(),
            "processed_at": datetime.now().isoformat()
        }
    
    def process_text_content(self, text_data: Dict, document_name: str) -> Dict:
        """Process text content from PDF"""
        page_num = text_data["page_number"]
        text_content = text_data["text_content"]
        
        print(f"  Processing text from page {page_num}...")
        
        # Create embeddings for text
        embedding = self.create_embeddings(text_content)
        
        # For text content, we can also generate a summary using Mistral if needed
        # For now, we'll use the raw text
        
        return {
            "content_type": "text",
            "page_number": page_num,
            "text_content": text_content,
            "caption": f"Text content from page {page_num}: {text_content[:200]}...",  # Preview
            "embedding": embedding.tolist(),
            "processed_at": datetime.now().isoformat()
        }
    def generate_caption_with_mistral(self, image_base64: str, page_number: int, content_type: str = "page") -> str:
        """Generate caption for image using Mistral API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mistral_api_key}"
        }
        
        if "embedded" in content_type:
            prompt_text = f"Analyze this embedded image from PDF page {page_number} and provide a detailed description. Focus on: 1) What the image shows (objects, people, scenes, diagrams, charts), 2) Any text visible in the image, 3) Important details, colors, or visual elements, 4) Context or purpose of the image. Make it comprehensive for QA retrieval."
        else:
            prompt_text = f"Analyze this PDF page (page {page_number}) and provide a detailed, structured summary. Focus on: 1) Main topics and key information, 2) Important data, numbers, or statistics, 3) Key concepts or terminology, 4) Any diagrams, charts, or visual elements. Make it comprehensive for QA retrieval."
        
        payload = {
            "model": "pixtral-12b-2409",  # Mistral's vision model
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(self.mistral_api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"Error generating caption for {content_type} {page_number}: {str(e)}"
    
    def create_embeddings(self, text: str) -> np.ndarray:
        """Create embeddings for text using sentence transformer"""
        return self.embedder.encode([text])[0]
    
    def process_pdf(self, pdf_path: str, document_name: str = None) -> Dict[str, Any]:
        """Enhanced PDF processing function that handles text, embedded images, and page images"""
        if not document_name:
            document_name = os.path.basename(pdf_path).replace('.pdf', '')
        
        print(f"Processing PDF: {document_name}")
        
        all_content = []
        
        # 1. Extract and process text content
        print("Extracting text content...")
        text_data = self.extract_text_from_pdf(pdf_path)
        for text_item in text_data:
            content_item = self.process_text_content(text_item, document_name)
            all_content.append(content_item)
            time.sleep(0.5)  # Small delay
        
        # 2. Extract and process embedded images
        print("Extracting embedded images...")
        embedded_images = self.extract_images_from_pdf(pdf_path)
        print(f"Found {len(embedded_images)} embedded images")
        
        for img_data in embedded_images:
            content_item = self.process_embedded_image(img_data, document_name)
            all_content.append(content_item)
            time.sleep(1)  # Delay for API rate limiting
        
        # 3. Process full page images (for pages with complex layouts or when text/images aren't enough)
        print("Processing full page images...")
        page_images = self.pdf_to_images(pdf_path)
        
        for i, image in enumerate(page_images):
            page_num = i + 1
            
            # Skip if we already have good text content for this page
            has_text = any(item["page_number"] == page_num and item["content_type"] == "text" 
                          for item in all_content)
            has_images = any(item["page_number"] == page_num and item["content_type"] == "embedded_image" 
                            for item in all_content)
            
            # Process page image if no text or limited content
            if not has_text or not has_images:
                print(f"  Processing full page image {page_num}/{len(page_images)}...")
                
                # Convert to base64
                image_base64 = self.image_to_base64(image)
                
                # Generate caption with Mistral
                caption = self.generate_caption_with_mistral(image_base64, page_num, "full page")
                
                # Create embeddings
                embedding = self.create_embeddings(caption)
                
                # Save image
                image_path = f"{self.output_dir}/{document_name}_full_page_{page_num}.png"
                image.save(image_path)
                
                page_data = {
                    "content_type": "full_page",
                    "page_number": page_num,
                    "image_path": image_path,
                    "caption": caption,
                    "embedding": embedding.tolist(),
                    "processed_at": datetime.now().isoformat()
                }
                
                all_content.append(page_data)
                time.sleep(1)  # Delay for API rate limiting
        
        # Create document record
        document_record = {
            "document_id": str(uuid.uuid4()),
            "document_name": document_name,
            "pdf_path": pdf_path,
            "total_pages": len(page_images),
            "total_content_items": len(all_content),
            "content_breakdown": {
                "text_items": len([c for c in all_content if c["content_type"] == "text"]),
                "embedded_images": len([c for c in all_content if c["content_type"] == "embedded_image"]),
                "full_page_images": len([c for c in all_content if c["content_type"] == "full_page"])
            },
            "content": all_content,
            "created_at": datetime.now().isoformat(),
            "summary": self.generate_document_summary(all_content)
        }
        
        # Add to database
        self.documents_db.append(document_record)
        
        # Save as JSON
        json_path = f"{self.output_dir}/{document_name}_processed.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(document_record, f, indent=2, ensure_ascii=False)
        
        print(f"Processing complete! JSON saved to: {json_path}")
        return document_record
    
    def generate_document_summary(self, content_data: List[Dict]) -> str:
        """Generate overall document summary from all content"""
        text_items = len([c for c in content_data if c["content_type"] == "text"])
        embedded_images = len([c for c in content_data if c["content_type"] == "embedded_image"])
        full_pages = len([c for c in content_data if c["content_type"] == "full_page"])
        
        return f"Document contains {len(content_data)} content items: {text_items} text sections, {embedded_images} embedded images, {full_pages} full page analyses."
    
    def query_documents(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query the RAG system with a question"""
        if not self.documents_db:
            return []
        
        # Create embedding for question
        question_embedding = self.create_embeddings(question)
        
        # Find most relevant pages across all documents
        candidates = []
        
        for doc in self.documents_db:
            for content_item in doc["content"]:
                content_embedding = np.array(content_item["embedding"])
                similarity = cosine_similarity([question_embedding], [content_embedding])[0][0]
                
                candidates.append({
                    "document_name": doc["document_name"],
                    "page_number": content_item["page_number"],
                    "content_type": content_item["content_type"],
                    "caption": content_item["caption"],
                    "similarity": similarity,
                    "document_id": doc["document_id"],
                    "content_item": content_item
                })
        
        # Sort by similarity and return top results
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:top_k]
    
    def answer_question(self, question: str, context_pages: List[Dict] = None) -> str:
        """Generate answer using Mistral with retrieved context"""
        if context_pages is None:
            context_pages = self.query_documents(question)
        
        if not context_pages:
            return "No relevant information found in the processed documents."
        
        # Prepare context
        context = "\n\n".join([
            f"Page {item['page_number']} ({item['content_type']}) from {item['document_name']}:\n{item['caption']}"
            for item in context_pages
        ])
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mistral_api_key}"
        }
        
        payload = {
            "model": "mistral-large-latest",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context from PDF documents. Use only the information provided in the context."
                },
                {
                    "role": "user",
                    "content": f"Context from documents:\n{context}\n\nQuestion: {question}\n\nAnswer based on the context above:"
                }
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        try:
            # Add delay to avoid rate limiting
            time.sleep(2)  # 2 second delay between requests
            
            response = requests.post(self.mistral_api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                return "Rate limit reached. Please wait a moment before asking another question."
            return f"Error generating answer: {str(e)}"
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def save_all_documents(self, filename: str = "rag_database.json"):
        """Save all processed documents to a single JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "documents": self.documents_db,
                "total_documents": len(self.documents_db),
                "created_at": datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        print(f"Database saved to: {filepath}")
    
    def load_documents(self, filename: str = "rag_database.json"):
        """Load previously processed documents"""
        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents_db = data.get("documents", [])
            print(f"Loaded {len(self.documents_db)} documents from {filepath}")
        except FileNotFoundError:
            print(f"No existing database found at {filepath}")


def main():
    """Interactive PDF RAG System"""
    
    # Initialize the system with your API key
    MISTRAL_API_KEY = "SCMvDbHeReWSUVk2wkJkAK5yixxL8i2b"
    rag_system = PDFRAGSystem(mistral_api_key=MISTRAL_API_KEY)
    
    # Use PDF from same folder
    pdf_path = "sample_pdf.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found in the current folder.")
        print("Please make sure 'sample_pdf.pdf' is in the same directory as this script.")
        return
    
    try:
        # Process the PDF
        print(f"\n🔄 Processing {pdf_path}...")
        document_name = "Sample PDF Document"
        
        result = rag_system.process_pdf(pdf_path, document_name)
        
        # Save the database
        rag_system.save_all_documents()
        
        print(f"\n✅ PDF processed successfully!")
        print(f"📄 Document: {result['document_name']}")
        print(f"📃 Total pages: {result['total_pages']}")
        print(f"📋 Content breakdown:")
        breakdown = result['content_breakdown']
        print(f"   - Text sections: {breakdown['text_items']}")
        print(f"   - Embedded images: {breakdown['embedded_images']}")
        print(f"   - Full page images: {breakdown['full_page_images']}")
        print(f"💾 Data saved to: {rag_system.output_dir}")
        
        # Interactive Q&A session
        print("\n" + "="*60)
        print("🤖 INTERACTIVE Q&A SESSION")
        print("="*60)
        print("Ask questions about your PDF! (Type 'quit' or 'exit' to stop)")
        print("-"*60)
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for using the PDF RAG System!")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            print("\n🔍 Searching for relevant information...")
            
            # Get relevant pages
            relevant_pages = rag_system.query_documents(question, top_k=3)
            
            if not relevant_pages:
                print("❌ No relevant information found.")
                continue
            
            # Show relevant content
            print(f"📋 Found {len(relevant_pages)} relevant content items:")
            for i, item in enumerate(relevant_pages, 1):
                content_type = item['content_type']
                print(f"  {i}. Page {item['page_number']} ({content_type}) - similarity: {item['similarity']:.3f}")
            
            # Generate answer
            print("\n🤔 Generating answer...")
            answer = rag_system.answer_question(question, relevant_pages)
            
            print(f"\n💡 Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            
            # Show sources
            sources = [f"Page {p['page_number']} ({p['content_type']})" for p in relevant_pages]
            print(f"📚 Sources: {', '.join(sources)}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()


# Installation requirements:
# pip install PyMuPDF Pillow sentence-transformers scikit-learn requests