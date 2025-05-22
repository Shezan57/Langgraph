import os
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from typing import List, Literal, TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document


os.environ["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"]

# Step 1 Documents loading and store in a vector database
# Define documents for indexing
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]
docs = [WebBaseLoader(url).load() for url in urls]
doc_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=20)
doc_splits = text_splitter.split_documents(doc_list)

#Store documents in a vector database (chroma)
vectorstore = Chroma.from_documents(
    doc_splits,
    embedding=OpenAIEmbeddings(),
    collection_name="adaptive_RAG",
)
retriever = vectorstore.as_retriever()

# Step 2 Query routing
# Defines routing model
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"]
route_prompt = ChatPromptTemplate(
    [("system", "You are an expert at routing a user questio to vectorestore or web search."),
    ("human","{question}")])
question_router = route_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RouteQuery)

# Step 3 Retrieval Grarder and self-correction
class GraderDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Documents are relevant to the question, 'yes' or 'no'.")
grader_prompt = ChatPromptTemplate(
    [
        ("system","Evaluate if the document is relevant to the question. Answer 'yes' or 'no'."),
        ("human","Document:{document}\nQuestion:{question}")
    ])
retrival_grader = grader_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(GraderDocuments)

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[Document]

# Step 4 Web Search and RAG Generation
web_search_tool = TavilySearchResults(k=3)
def web_search(state):
    search_results = web_search_tool.invoke({"query": state["question"]})
    web_documents = [Document(page_content=result["content"]) for result in search_results if "content" in result]
    return {"documents": web_documents, "question": state["question"]}

# Decision-Making Logic and Workflow Graph
def generates(state):
    question = state["question"]
    documents = state["documents"]
    prompt_template = [
        ("system", "Use the following context to answer the question concisely and accurately:"),
        ("human", "Context: {context}\nQuestion: {question}")
    ]
    # Define ChatPromptTemplate using the above template
    rag_prompt = ChatPromptTemplate(prompt_template)
    # Create a rag generation chain with llm and output parsing
    rag_chain = rag_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"generation": generation}
# Route question based on a source
def route_question(state):
    source = question_router.invoke({"question":state["question"]}).datasource
    return "web_search" if source == "web_search" else "retrieve"
def retrieve(state):
    question = state["question"]
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}
def grade_documents(state: GraphState) -> GraphState:
    """
    Grades documents based on relevance to the question.
    """
    question = state["question"]
    documents = state["documents"]
    docs = []
    for doc in documents:
        # Pass doc.page_content to the grader as it expects a string document
        relevant = retrival_grader.invoke({"question":question, "document": doc.page_content})
        if relevant.binary_score == "yes":
            docs.append(doc)

    return {"question":question, "documents":docs}
#Compile and run the Graph
workflow = StateGraph(GraphState)
workflow.add_node("web_search", web_search)
workflow.add_node("generates", generates)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("route_question", route_question)
workflow.add_node("retrieve",retrieve)

workflow.add_conditional_edges(START, route_question, {"web_search":"web_search", "retrieve":"retrieve"})
workflow.add_edge("web_search", "generates")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("grade_documents","generates")
workflow.add_edge("generates",END)

app = workflow.compile()
# Run with example inputs
inputs = {"question":"What are the types of agent memory?"}
for output in app.stream(inputs):
    print(output)