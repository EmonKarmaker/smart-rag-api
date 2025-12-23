import ollama
from huggingface_hub import InferenceClient
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from app.config import settings


class LLMService:
    """Handles LLM interactions using LangChain with Ollama (local) or HuggingFace (web)."""
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        
        if self.provider == "ollama":
            self.model = settings.OLLAMA_MODEL
            self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            self.langchain_llm = None  # Ollama uses direct calls
        elif self.provider == "huggingface":
            self.model = settings.HUGGINGFACE_MODEL
            self.client = InferenceClient(
                model=self.model,
                token=settings.HUGGINGFACE_API_KEY
            )
            # LangChain HuggingFace integration
            self.langchain_llm = HuggingFaceHub(
                repo_id=self.model,
                huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
                model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")
        
        # Create LangChain prompt template
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

INSTRUCTIONS:
- Answer the question based ONLY on the context provided above.
- If the context doesn't contain enough information to answer, say "I don't have enough information in the provided documents to answer this question."
- Be concise and direct in your answer.
- If relevant, mention which source the information comes from.

QUESTION: {question}

ANSWER:"""
        )
        
        # Create LangChain chain if using HuggingFace
        if self.langchain_llm:
            self.chain = LLMChain(llm=self.langchain_llm, prompt=self.rag_prompt)
    
    def generate_answer(self, question: str, context: str) -> dict:
        """
        Generate an answer based on context and question using LangChain.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            
        Returns:
            Dict with 'answer' and 'prompt_used'
        """
        prompt = self.rag_prompt.format(context=context, question=question)
        
        if self.provider == "ollama":
            response = self._call_ollama(prompt)
        else:
            # Use LangChain for HuggingFace
            response = self._call_langchain(context, question)
        
        return {
            "answer": response,
            "prompt_used": prompt,
            "model": f"{self.provider}/{self.model}"
        }
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API directly."""
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=False
        )
        return response["response"]
    
    def _call_langchain(self, context: str, question: str) -> str:
        """Call LLM using LangChain."""
        response = self.chain.run(context=context, question=question)
        return response
    
    def _call_huggingface(self, prompt: str) -> str:
        """Call HuggingFace Inference API using chat completion."""
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def process_image_question(self, question: str, image_text: str, context: str) -> dict:
        """
        Generate answer for image-based question.
        
        Args:
            question: User's question
            image_text: OCR text extracted from image
            context: Retrieved context from documents
            
        Returns:
            Dict with 'answer' and 'prompt_used'
        """
        full_context = f"TEXT FROM IMAGE:\n{image_text}\n\nADDITIONAL CONTEXT:\n{context}"
        return self.generate_answer(question, full_context)
    
    def health_check(self) -> bool:
        """Check if LLM service is available."""
        try:
            if self.provider == "ollama":
                models = self.client.list()
                model_names = [m["name"] for m in models.get("models", [])]
                return any(self.model in name for name in model_names)
            else:
                # Test HuggingFace with chat completion
                messages = [{"role": "user", "content": "Hi"}]
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=5
                )
                return True
        except Exception as e:
            print(f"LLM health check failed: {e}")
            return False


# Create singleton instance
llm_service = LLMService()