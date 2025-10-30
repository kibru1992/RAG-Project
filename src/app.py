import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
# from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


def load_documents() -> List[Dict]:
    """
    Load documents for demonstration.

    Returns:
        List of sample documents with content and metadata
    """
    results = []
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to parent directory, then into data
    data_dir = os.path.join(script_dir, "..", "data")
    data_dir = os.path.normpath(data_dir)  # Normalize the path
    
    print(f"Looking for data in: {data_dir}")  # Debug line
    
    
    # Read all text files from data directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    if content:  # Only add non-empty files
                        results.append({
                            "content": content,
                            "metadata": {"source": filename, "type": "text_file"}
                        })
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    
    print(f"Loaded {len(results)} documents from {data_dir}")
    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Create RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Use the following context to answer the user's question. 
If the context doesn't contain relevant information, use your general knowledge but indicate this.

Context:
{context}

Question: {question}

Please provide a comprehensive and accurate answer based on the context above:
""")

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def query(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            String containing the answer
        """
        # Search for relevant context
        search_results = self.vector_db.search(input, n_results=n_results)
        
        # Combine retrieved document chunks into context
        context_chunks = search_results.get('documents', [])
        if not context_chunks:
            context = "No relevant context found in the knowledge base."
        else:
            context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context_chunks)])
        
        # Generate response using the chain
        llm_answer = self.chain.invoke({
            "context": context,
            "question": input
        })
        
        return llm_answer


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        sample_docs = load_documents()
        print(f"Loaded {len(sample_docs)} documents")

        # Add documents to the knowledge base
        if sample_docs:
            assistant.add_documents(sample_docs)
            print("Documents added to knowledge base successfully!")
        else:
            print("No documents found to add to knowledge base.")

        # Interactive query loop
        print("\n" + "="*50)
        print("RAG Assistant is ready! Ask questions or type 'quit' to exit.")
        print("="*50)
        
        done = False
        while not done:
            question = input("\nEnter your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                done = True
                print("Goodbye!")
            elif question:
                print("\nProcessing your question...")
                try:
                    result = assistant.query(question)
                    print(f"\nAnswer: {result}")
                except Exception as e:
                    print(f"Error processing question: {e}")
            else:
                print("Please enter a valid question.")

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have set up your .env file with at least one API key:")
        print("   - OPENAI_API_KEY (OpenAI models)")
        print("   - GROQ_API_KEY (Groq Llama models)")
        print("   - GOOGLE_API_KEY (Google Gemini models)")
        print("2. Install required dependencies: pip install -r requirements.txt")
        print("3. Check your internet connection for API calls")


if __name__ == "__main__":
    main()