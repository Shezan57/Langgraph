import os
from typing import TypedDict, List, Literal
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.documents import Document # Add this import

os.environ["OPENAI_API_KEY"] 

# List of URLs to scrape
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

# Load the documents from the URLs
docs = [WebBaseLoader(url).load()for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs_split = text_splitter.split_documents(docs_list) # Renamed to avoid conflict with 'docs' from WebBaseLoader

# store documents in chroma vectorstore
vectorstore = Chroma.from_documents(
    documents=docs_split, # Use the split documents
    embedding=OpenAIEmbeddings(),
    collection_name='rag-chroma'
)
# Create a retriever from the vectorstore
retriever = vectorstore.as_retriever()

# set up prompt and model
prompt = ChatPromptTemplate.from_template(
    """
    Use the following context to answer the question concisely:
    Question: {question}
    Context: {context}
    Answer:
    """
)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
rag_chain = prompt | model | StrOutputParser()

# Implementing Self-Reflective Workflow Steps Define nodes for
# retrieval, generation, grading, and query transformation.

class GraphState(TypedDict):
    question : str
    generation: str
    documents : List[Document] # Changed 'document' to 'documents' and type to List[Document]

# Retrival grader setup
class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Documents are relevant to the question, 'yes' or 'no'")

retrieval_prompt = ChatPromptTemplate.from_template(
"""You are a grader assessing if a document is relevant to a user's question.
Document: {document}
Question: {question}
Is the document relevant? Answer 'yes' or 'no'."""
)
retrieval_grader = retrieval_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(GradeDocuments)

# Hallucination  grader setup
class GradeHallucination(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Answer is grounded in the documents, 'yes' or 'no'")

hallucination_prompt = ChatPromptTemplate.from_template(
"""You are a grader assessing if an answer is grounded in retrieved documents.
Documents: {documents}
Answer: {generation}
Is the answer grounded in the documents? Answer 'yes' or 'no'."""
)
hallucination_grader = hallucination_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(GradeHallucination)

# Answer grader setup
class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Answer address the question, 'yes' or 'no'")
answer_prompt = ChatPromptTemplate.from_template(
    """You are a grader assessing if an answer addresses the user's question.
    Question: {question}
    Answer: {generation}
    Does the answer address the question? Answer 'yes' or 'no'."""
)
answer_grader = answer_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(GradeAnswer)

# Define Langgraph Functions
def retrieve(state: GraphState) -> GraphState: # Added type hint for clarity
    question = state["question"]
    # Use get_relevant_documents for retrievers
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def generate(state: GraphState) -> GraphState: # Added type hint for clarity
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation, "question": question, "documents": documents}

def grade_documents(state: GraphState) -> GraphState: # Added type hint for clarity
    """
    Grades documents based on relevance to the question.
    """
    question = state["question"]
    documents = state["documents"]
    relevant_docs = []
    for doc in documents:
        # Pass doc.page_content to the grader as it expects a string document
        response = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        if response.binary_score == "yes": # Access binary_score as an attribute
            relevant_docs.append(doc)
    return {"documents": relevant_docs, "question": question}

def decide_to_generate(state: GraphState) -> str: # Added type hint for clarity
    """
    Decides whether to generate an answer based on the number of relevant documents.
    """
    if not state.get("documents") or not state["documents"]: # Added .get for safety
        return "transform_query" # No relevant docs found; rephrase query
    return "generate" # Relevant docs found; proceed to generate

def grade_generation_v_documents_and_question(state: GraphState) -> str: # Added type hint for clarity
    """Check if the generation is grounded in the retrieved documents and addresses the question."""
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    # Step 1: Check if the generation is grounded in documents
    hallucination_check = hallucination_grader.invoke(
        {"documents": documents, "generation": generation})
    if hallucination_check.binary_score == "no": # Access binary_score as an attribute
        return "not supported" # Generation is not grounded in the documents
    # Step 2: Check if generation addresses the question
    answer_check = answer_grader.invoke(
        {"question": question, "generation": generation})
    return "useful" if answer_check.binary_score == "yes" else "not useful" # Access binary_score

def transform_query(state: GraphState) -> GraphState: # Added type hint for clarity
    """
    Rephrase the question to improve retrieval.
    """
    transform_prompt = ChatPromptTemplate.from_template(
    """You are a question re-writer that converts an input question to a better version
    optimized for retrieving relevant documents.
    Original question: {question}
    Please provide a rephrased question."""
    )
    question_rewriter = transform_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()
    question = state["question"]
    # Rephrase the question
    transformed_question = question_rewriter.invoke({"question": question})
    # Preserve existing documents if any, or initialize to empty list
    current_documents = state.get("documents", [])
    return {"question": transformed_question, "documents": current_documents}


# Workflow setup
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate"
    }
)
workflow.add_edge("transform_query", "retrieve") # Loop back to retrieve after transforming
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate", # Potentially retry generation or could go to transform_query
        "useful": END,
        "not useful": "transform_query"
    }
)

#Compile the workflow
app = workflow.compile()
# Example input
inputs = {"question": "Explain how the different types of agent memory work?"}
for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")