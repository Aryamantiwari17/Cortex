from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import json
import os
from getpass import getpass

# Updated LlamaIndex imports
from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    VectorStoreIndex,
    Document
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.indices.struct_store.sql import SQLDatabase
from llama_index.core.tools import ToolMetadata
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.composability import ComposableGraph
from llama_index.llms.langchain import LangChainLLM

from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float
from langchain_groq import ChatGroq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_groq_llm():
    """Setup Groq LLM with API key input"""
    # Check for API key in environment
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    # If not found, prompt user for API key
    if not groq_api_key:
        print("\nGroq API key not found in environment variables.")
        groq_api_key = getpass("Enter your Groq API key: ")
        os.environ["GROQ_API_KEY"] = groq_api_key
    
    # Initialize Groq LLM
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768",
        temperature=0.7,
        max_tokens=4096
    )

class CompanyData(BaseModel):
    name: str
    faq_content: str
    knowledge_base_path: str
    db_connection: str
    response_templates: Dict[str, str]

class FeedbackMetadata(BaseModel):
    category: str
    sentiment: float
    urgency: int
    timestamp: datetime = Field(default_factory=datetime.now)

class FeedbackSystem:
    def __init__(self, company_data: CompanyData):
        self.company_data = company_data
        # Initialize Groq LLM
        groq_llm = setup_groq_llm()
        self.llm = LangChainLLM(llm=groq_llm)
        # Set up global settings with the LLM
        Settings.llm = self.llm
        self.setup_indices()
        self.setup_query_engines()

    def setup_indices(self):
        """Setup different indices for various data sources"""
        try:
            # Setup FAQ Index
            faq_doc = Document(text=self.company_data.faq_content, metadata={"type": "faq"})
            self.faq_index = VectorStoreIndex.from_documents([faq_doc])

            # Setup Knowledge Base Index
            if os.path.exists(self.company_data.knowledge_base_path):
                documents = SimpleDirectoryReader(
                    self.company_data.knowledge_base_path
                ).load_data()
                self.kb_index = VectorStoreIndex.from_documents(documents)
            else:
                logger.warning(f"Knowledge base path {self.company_data.knowledge_base_path} not found. Creating empty index.")
                self.kb_index = VectorStoreIndex.from_documents(
                    [Document(text="", metadata={"type": "kb"})]
                )

            # Setup SQL Database connection
            self.sql_database = SQLDatabase(
                engine=create_engine(self.company_data.db_connection)
            )
        except Exception as e:
            logger.error(f"Error setting up indices: {str(e)}")
            raise

    def setup_query_engines(self):
        """Setup specialized query engines for different types of queries"""
        try:
            # Create tools for different data sources
            tools = [
                QueryEngineTool(
                    query_engine=self.faq_index.as_query_engine(),
                    metadata=ToolMetadata(
                        name="faq_tool",
                        description="Useful for answering frequently asked questions"
                    )
                ),
                QueryEngineTool(
                    query_engine=self.kb_index.as_query_engine(),
                    metadata=ToolMetadata(
                        name="knowledge_base_tool",
                        description="Useful for detailed product and service information"
                    )
                ),
                QueryEngineTool(
                    query_engine=self.sql_database.as_query_engine(),
                    metadata=ToolMetadata(
                        name="database_tool",
                        description="Useful for querying specific product or customer data"
                    )
                )
            ]

            # Create sub-question query engine
            self.query_engine = SubQuestionQueryEngine.from_defaults(
                query_engine_tools=tools
            )

            # Create composable graph for complex queries
            self.graph = ComposableGraph.from_indices(
                indices=[self.faq_index, self.kb_index],
                index_summaries=["FAQ information", "Knowledge base information"]
            )
        except Exception as e:
            logger.error(f"Error setting up query engines: {str(e)}")
            raise

    async def analyze_sentiment(self, feedback: str) -> FeedbackMetadata:
        """Analyze feedback sentiment and metadata"""
        try:
            prompt = f"""
            Analyze the following feedback for {self.company_data.name}:
            {feedback}

            Provide the following in JSON format:
            - category (complaint/praise/suggestion/mixed)
            - sentiment (float between -1 and 1)
            - urgency (integer 1-5)
            """

            response = await self.llm.apredict(prompt)
            data = json.loads(response)
            return FeedbackMetadata(**data)
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return FeedbackMetadata(
                category="error",
                sentiment=0.0,
                urgency=1
            )

    async def extract_topics(self, feedback: str) -> List[str]:
        """Extract main topics from feedback"""
        try:
            query = f"""
            Extract the main topics discussed in this feedback:
            {feedback}

            Return as a list of topic strings.
            """
            
            response = self.query_engine.query(query)
            return json.loads(response.response)
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return ["general"]

    async def get_relevant_information(self, topics: List[str]) -> Dict[str, str]:
        """Get relevant information for each topic"""
        info = {}
        try:
            for topic in topics:
                # Query each index for relevant information
                faq_response = self.faq_index.as_query_engine().query(topic)
                kb_response = self.kb_index.as_query_engine().query(topic)
                db_response = self.sql_database.as_query_engine().query(topic)

                info[topic] = {
                    "faq": faq_response.response,
                    "knowledge_base": kb_response.response,
                    "database": db_response.response
                }
            return info
        except Exception as e:
            logger.error(f"Error getting relevant information: {str(e)}")
            return {"error": str(e)}

    async def generate_response(
        self, 
        feedback: str,
        metadata: FeedbackMetadata,
        relevant_info: Dict[str, str]
    ) -> str:
        """Generate appropriate response based on analysis"""
        try:
            # Get response template based on category
            template = self.company_data.response_templates.get(
                metadata.category,
                self.company_data.response_templates["default"]
            )

            # Create comprehensive query
            query = f"""
            Generate a response using this template:
            {template}

            Consider the following:
            - Feedback: {feedback}
            - Sentiment: {metadata.sentiment}
            - Urgency: {metadata.urgency}
            - Relevant Information: {json.dumps(relevant_info)}

            The response should:
            1. Address all mentioned topics
            2. Maintain {self.company_data.name}'s tone
            3. Provide specific solutions
            4. Include relevant information from our knowledge base
            """

            response = self.graph.as_query_engine().query(query)
            return response.response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Thank you for your feedback. We are experiencing technical difficulties processing your request."

    async def process_feedback(self, feedback: str) -> Dict:
        """Process customer feedback using LlamaIndex capabilities"""
        try:
            # Analyze sentiment and metadata
            metadata = await self.analyze_sentiment(feedback)
            
            # Extract main topics
            topics = await self.extract_topics(feedback)
            
            # Get relevant information for topics
            relevant_info = await self.get_relevant_information(topics)
            
            # Generate response
            response = await self.generate_response(
                feedback,
                metadata,
                relevant_info
            )

            return {
                "status": "success",
                "metadata": metadata.dict(),
                "topics": topics,
                "relevant_information": relevant_info,
                "response": response
            }
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

async def main():
    print("\n=== TechCorp Feedback Analysis System (LlamaIndex + Groq) ===")
    
    try:
        # Example company data
        company_data = CompanyData(
            name="TechCorp",
            faq_content="""
            Q: What are your business hours?
            A: Our customer service is available Monday-Friday 9AM-6PM EST.
            
            Q: What is your return policy?
            A: We offer a 30-day money-back guarantee on all products.
            
            Q: Do you offer international shipping?
            A: Yes, we ship to over 50 countries worldwide.
            """,
            knowledge_base_path="./knowledge_base",
            db_connection="sqlite:///company.db",
            response_templates={
                "complaint": "We apologize for your experience with {issue}...",
                "praise": "Thank you for your kind words about {topic}...",
                "suggestion": "Thank you for your suggestion about {topic}...",
                "mixed": "Thank you for your feedback. We're glad you enjoyed {positive_aspect}...",
                "default": "Thank you for your feedback..."
            }
        )

        # Initialize system
        system = FeedbackSystem(company_data)
        print("\nSystem initialized successfully!")
        print("Type 'quit' to exit or enter customer feedback to analyze")

        while True:
            print("\n" + "="*50)
            print("Enter customer feedback (or 'quit' to exit):")
            feedback = input("> ")
            
            if feedback.lower() == 'quit':
                print("Thank you for using our system. Goodbye!")
                break
            
            if not feedback.strip():
                print("Please enter some feedback text.")
                continue
            
            print("\nProcessing feedback...")
            result = await system.process_feedback(feedback)
            
            if result["status"] == "success":
                print("\n=== Analysis Results ===")
                metadata = result["metadata"]
                print(f"Category: {metadata['category']}")
                print(f"Sentiment: {metadata['sentiment']:.2f}")
                print(f"Urgency: {metadata['urgency']}")
                
                print("\n=== Identified Topics ===")
                for topic in result["topics"]:
                    print(f"- {topic}")
                
                print("\n=== Generated Response ===")
                print(result["response"])
            else:
                print("\nError:", result["message"])

    except Exception as e:
        print(f"\nError initializing system: {str(e)}")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())