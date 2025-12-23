from huggingface_hub import InferenceClient
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from app.config import settings


class LLMService:
    """Handles LLM interactions using LangChain with HuggingFace Hub."""
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.HUGGINGFACE_MODEL
        
        # Direct client for health checks
        self.client = InferenceClient(
            model=self.model,
            token=settings.HUGGINGFACE_API_KEY
        )
        
        # LangChain HuggingFace Endpoint (modern approach)
        self.langchain_llm = HuggingFaceEndpoint(
            repo_id=self.model,
            huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
            temperature=0.7,
            max_new_tokens=512
        )
        
        # LangChain prompt template for RAG
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer based ONLY on the context provided above.
- If the context doesn't contain enough information, say so.
- Be concise and direct.
- Mention which source the information comes from.

QUESTION: {question}

ANSWER:"""
        )
        
        # Modern LangChain: prompt | llm
        self.chain = self.rag_prompt | self.langchain_llm
    
    def generate_answer(self, question: str, context: str) -> dict:
        """Generate answer using LangChain."""
        try:
            # Use LangChain chain
            response = self.chain.invoke({"context": context, "question": question})
        except Exception as e:
            # Fallback to direct API
            print(f"LangChain failed, using fallback: {e}")
            response = self._call_direct(question, context)
        
        return {
            "answer": response,
            "prompt_used": self.rag_prompt.format(context=context, question=question),
            "model": f"{self.provider}/{self.model}"
        }
    
    def _call_direct(self, question: str, context: str) -> str:
        """Direct HuggingFace API call as fallback."""
        prompt = self.rag_prompt.format(context=context, question=question)
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def process_image_question(self, question: str, image_text: str, context: str) -> dict:
        """Generate answer for image+text multimodal question using LangChain."""
        full_context = f"TEXT FROM IMAGE:\n{image_text}\n\nDOCUMENT CONTEXT:\n{context}"
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