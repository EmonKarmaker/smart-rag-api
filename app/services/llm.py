from huggingface_hub import InferenceClient
from app.config import settings


class LLMService:
    """Handles LLM interactions using HuggingFace Hub."""
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.HUGGINGFACE_MODEL
        self.client = InferenceClient(
            model=self.model,
            token=settings.HUGGINGFACE_API_KEY
        )
    
    def generate_answer(self, question: str, context: str) -> dict:
        """Generate an answer based on context and question."""
        prompt = self._build_prompt(question, context)
        response = self._call_huggingface(prompt)
        
        return {
            "answer": response,
            "prompt_used": prompt,
            "model": f"{self.provider}/{self.model}"
        }
    
    def _call_huggingface(self, prompt: str) -> str:
        """Call HuggingFace Inference API."""
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build a RAG prompt."""
        return f"""You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer based ONLY on the context provided above.
- If the context doesn't contain enough information, say so.
- Be concise and direct.
- Mention which source the information comes from.

QUESTION: {question}

ANSWER:"""
    
    def process_image_question(self, question: str, image_text: str, context: str) -> dict:
        """Generate answer for image-based question."""
        full_context = f"TEXT FROM IMAGE:\n{image_text}\n\nCONTEXT:\n{context}"
        return self.generate_answer(question, full_context)
    
    def health_check(self) -> bool:
        """Check if LLM service is available."""
        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            print(f"LLM health check failed: {e}")
            return False


llm_service = LLMService()