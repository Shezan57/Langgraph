import os
from typing import Literal, TypedDict, List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import START, END, StateGraph
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from pprint import pprint

from sympy import content

os.environ["TAVILY_API_KEY"]
os.environ["OPENAI_API_KEY"]

# Step 1: Load and prepare documents
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

# Load and split the documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=750, chunk_overlap=50)
doc_splits = text_splitter.split_documents(docs_list)
# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(doc_splits, embeddings, collection_name='crag-chroma')
retriever = vector_store.as_retriever()

# Step 2: Define Graders and Relevance Model
class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Documents are relevant to the question, 'yes' or 'no")
retrieval_prompt = ChatPromptTemplate.from_template(
    """You are a grader assessing if a document is relevant to a user's question.
    Document: {document}
    Question: {question}
    Is the document relevant? Answer 'yes' or 'no'."""
)
retrieval_grader = retrieval_prompt | ChatOpenAI(model='gpt-4o-mini', temperature=0).with_structured_output(GradeDocuments)

# Step 3: Query Re-writer
class ImproveQuestion(BaseModel):
    improved_question: str = Field(description="Formulate and Improved question")
re_write_prompt = ChatPromptTemplate.from_template(
    "Here is the initial question: \n\n{question}\nFormulate an improved question that is more specific and clear."
)
question_rewriter = re_write_prompt | ChatOpenAI(model='gpt-4o-mini', temperature=0).with_structured_output(ImproveQuestion)

# Define prompt template
prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question:
Question: {question}
Context: {context}
Answer:
""")
rag_chain = prompt | ChatOpenAI(model='gpt-4o-mini', temperature=0) | StrOutputParser()

# Define CRAG State
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]

# Step 4: Define Workflow Nodes
def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search_needed = "No"
    for doc in documents:
        grade = retrieval_grader.invoke({"document": doc.page_content, "question": question})
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search_needed = "Yes"
    return {"documents": filtered_docs, "question": question, "web_search": web_search_needed}
def transform_query(state):
    question = state["question"]
    rewriten_question = question_rewriter.invoke({"question": question})
    return {"documents": state["documents"], "question": rewriten_question.improved_question}

def web_search(state):
    """
    Web search based on the re-phrased question.
    Args:
    state (dict): The current graph state
    Returns:
    state (dict): Updates a documents key with appended web results
    """
    question = state["question"]
    documents = state["documents"]
    pprint(question+"\n")
    # Perform web search using TavilySearchResults and extract only the 'content' field for Document
    search_result = TavilySearchResults(k=3).invoke({"query": question})
    web_documents = [
        Document(page_content=result["content"]) for result in search_result if "content" in result
    ]
    # Append the web search results to the existing documents
    documents.extend(web_documents)
    return {"documents": documents, "question": question}
def generate(state):
    generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"generation":generation}
def decide_to_generate(state):
    """"
    Determines whether to generate an answer, or re-generate a question.
    Args:
    state (dict): The current graph state
    Returns:
    str: Binary decision for next node to call"""
    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]
    if web_search == "yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
    return "transform_query"

# Step 6: Build and Compile the Graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_searcher", web_search)
workflow.add_node("generate", generate)

# Define edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate,
{"transform_query": "transform_query", "generate": "generate"})
workflow.add_edge("transform_query", "web_searcher")
workflow.add_edge("web_searcher", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()

# Example input
inputs = {"question": "Explain how the different types of agent memory work?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print the full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")
pprint(value["generation"])