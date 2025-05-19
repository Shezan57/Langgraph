from typing import TypedDict, Any, NotRequired
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph, state
import asyncio
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sphinx.addnodes import document

os.environ["OPENAI_API_KEY"]

urls = [
    "https://github.com/facebookresearch/faiss",
    "https://github.com/facebookresearch/faiss/wiki",
    "https://github.com/facebookresearch/faiss/wiki/Faiss-indexes"
]

docs = [WebBaseLoader(url).load()for url in urls]
doc_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 1000,
    chunk_overlap = 20,
)
doc_splits = text_splitter.split_documents(doc_list)
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(),
    collection_name="rag-chroma"
)
retriever = vectorstore.as_retriever()

# RAG chain
prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context
    to answer the question. If you don't know the answer, just say that you don't know. Use three
    sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
rag_chain = prompt | model | StrOutputParser()

# Define the graph
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
    question: question
    generation: LLM generation
    documents: list of documents
    """
    question: str
    generation: NotRequired[str]
    documents: list[Any]
    web_search: NotRequired[str]
# Retrieve node
def retrieve(state):
    """
    Retrieve documents
    Args:
    state (dict): The current graph state
    Returns:
    state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("Retrieving documents...")
    question = state["question"]
    # retrieval
    documents = retriever.invoke(question)
    return {
        "question": question,
        "documents": documents,
        "generation": state.get("generation"),
        "web_search": state.get("web_search"),
    }

# Generate node
def generate(state):
    """
    Generate answer
    Args:
    state (dict): The current graph state
    Returns:
    state (dict): New key added to state, generation, that contains LLM generation
    """
    print("Generating answer...")
    question = state["question"]
    documents = state["documents"]
    context = "\n".join([doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in documents])
    generation = rag_chain.invoke({"context": context, "question": question})
    # If generation is not a string, extract the string value
    if isinstance(generation, dict) and "text" in generation:
        generation = generation["text"]
    return {
        "question": question,
        "documents": documents,
        "generation": generation,
        "web_search": state.get("web_search"),
    }
# Define the workflow
def create_workflow():
    workflow = StateGraph(GraphState)
    # add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    # add edges
    workflow.add_edge(START,"retrieve")
    workflow.add_edge("retrieve","generate")
    workflow.add_edge("generate",END)

    return workflow.compile(checkpointer=MemorySaver())
#Run the workflow
async def run_workflow():
    app = create_workflow()
    config = {"configurable":{"thread_id":"1"},
              "recursion_limit":50
              }
    inputs = {"question":f"what are flat indexes?"}
    try:
        for event in app.stream(inputs, config=config, stream_mode="values"):
            if "error" in event:
                print(f"Error:{event['error']}")
                break
            print(event)
    except Exception as e:
        print(f"workflow exectuion failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_workflow())

