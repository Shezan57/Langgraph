from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import Tool
import json
import os

api_key = os.environ["GROQ_API_KEY"]

# Define product catalog
product_catalog = {
    "iPhone 14": "The iPhone 14 features a 6.1-inch display, A15 Bionic chip, and improved camera system.",
    "Samsung Galaxy S21": "The Samsung Galaxy S21 has a 6.2-inch display, Exynos 2100 processor, and triple camera setup.",
    "Google Pixel 6": "The Google Pixel 6 comes with a 6.4-inch display, Google Tensor chip, and advanced AI features.",
    "OnePlus 9": "The OnePlus 9 features a 6.55-inch display, Snapdragon 888 processor, and Hasselblad camera system.",
    "Sony WH-1000XM4": "The Sony WH-1000XM4 is a wireless noise-canceling headphone with up to 30 hours of battery life.",
    "Bose QuietComfort 35 II": "The Bose QuietComfort 35 II offers excellent noise cancellation and up to 20 hours of battery life.",
    "Apple AirPods Pro": "The Apple AirPods Pro features active noise cancellation and a customizable fit.",
    "Jabra Elite 85h": "The Jabra Elite 85h has SmartSound technology and up to 36 hours of battery life.",
    "Dell XPS 13": "The Dell XPS 13 is a premium ultrabook with a 13.4-inch InfinityEdge display and Intel Core i7 processor.",
    "MacBook Air M1": "The MacBook Air M1 features Apple's M1 chip, Retina display, and up to 18 hours of battery life.",
    "HP Spectre x360": "The HP Spectre x360 is a 2-in-1 laptop with a 13.3-inch display and Intel Core i7 processor.",
    "Lenovo ThinkPad X1 Carbon": "The Lenovo ThinkPad X1 Carbon is a business laptop with a 14-inch display and Intel Core i7 processor.",
    "Asus ROG Zephyrus G14": "The Asus ROG Zephyrus G14 is a gaming laptop with a 14-inch display and AMD Ryzen 9 processor.",
    "Microsoft Surface Pro 7": "The Microsoft Surface Pro 7 is a 2-in-1 device with a 12.3-inch display and Intel Core i7 processor."
}
stock_data = {
    "iPhone 14": "In stock",
    "Samsung Galaxy": "In stock",
    "Google Pixel": "In stock",
    "OnePlus": "Out of stock",
    "Sony WH-1000XM4": "In stock",
    "Bose QuietComfort 35 II": "Out of stock",
    "Apple AirPods Pro": "In stock",
    "Jabra Elite 85h": "Out of stock",
    "Dell XPS 13": "In stock",
    "MacBook Air M1": "Out of stock",
    "HP Spectre x360": "In stock",
    "Lenovo ThinkPad X1 Carbon": "Out of stock",
    "Asus ROG Zephyrus G14": "In stock",
    "Microsoft Surface Pro 7": "In stock"
}

# Properly define the tool function with args dictionary
def get_product_info(args):
    """Fetch product information."""
    # Handle both string and dictionary inputs
    if isinstance(args, str):
        try:
            # Parse JSON string
            args = json.loads(args)
        except json.JSONDecodeError:
            # If not valid JSON, use entire string as product name
            return product_catalog.get(args, "Sorry, product not found.")

    # Now handle dictionary input
    if isinstance(args, dict):
        product_name = args.get("product_name")
        return product_catalog.get(product_name, "Sorry, product not found.")

    # Fallback
    return "Sorry, I couldn't understand the product name."

# Function to check stock availability
def check_stock(args):
    """Check stock availability."""
    if isinstance(args, str):
        try:
            # Parse JSON string
            args = json.loads(args)
        except json.JSONDecodeError:
            return stock_data.get(args, "Sorry, product not found.")

    if isinstance(args, dict):
        product_name = args.get("product_name")
        return stock_data.get(product_name, "Sorry, product not found.")

    return "Sorry, I couldn't understand the product name."


# Create proper Tool with schema
product_info_tool = Tool.from_function(
    func=get_product_info,
    name="product_info",
    description="Get information about consumer electronic products",
    args_schema={
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "The name of the product to look up"
            }
        },
        "required": ["product_name"]
    }
)

# Create stock check tool
stock_check_tool = Tool.from_function(
    func=check_stock,
    name="check_stock",
    description="Check stock availability of products",
    args_schema={
        "type": "object",
        "properties": {
            "product_name": {
                "type": "string",
                "description": "The name of the product to check stock for"
            }
        },
        "required": ["product_name"]
    }
)

# Initialize the memory saver
checkpoint = MemorySaver()

# Initialize the Groq model with a known compatible model
llm = ChatGroq(
    api_key=api_key,
    model="deepseek-r1-distill-llama-70b"  # Known available model on Groq
)

# Create the ReAct agent with memory saver
graph = create_react_agent(
    model=llm,
    tools=[product_info_tool, stock_check_tool],
    checkpointer=checkpoint
)

# Set up thread configuration
config = {"configurable": {"thread_id": "thread_1"}}

# User input initial inquiry
inputs = {"messages": [("user", "Hi, I'm Shezan. Tell me about the Jabra Elite 85h.")]}
try:
    messages = graph.invoke(inputs, config=config)
    for message in messages["messages"]:
        print(message.content)
except Exception as e:
    print(f"Error: {str(e)}")

# User input: repeated inquiry (memory recall)
input2 = {"messages": [("user", "Is the Jabra Elite 85h in stock?")]}
try:
    messages2 = graph.invoke(input2, config=config)
    for message2 in messages2["messages"]:
        print(message2.content)
except Exception as e:
    print(f"Error: {str(e)}")