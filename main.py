import os
from rag import RAGManager

# Initialize the RAGManager with your OpenAI API key
rag_manager = RAGManager(embedding_model_api_key=os.getenv("EMBED_API_KEY"))

# Function to offer the user a choice of materials
def offer_material_choice():
    return """
    Hi! What would you like to learn today? Please choose one of the following:
    1. Peristiwa Rengasdengklok
    2. Peristiwa 10 Nopember
    3. Konferensi Meja Bundar
    """

# Function to map user's choice to a material
def get_material_choice(user_input):
    choices = {
        "1": "Peristiwa Rengasdengklok",
        "2": "Peristiwa 10 Nopember",
        "3": "Konferensi Meja Bundar"
    }
    return choices.get(user_input.strip(), None)  # Returns None if input is invalid

# Main interaction flow
def interact_with_user():
    # Offer material choice to the user
    print(offer_material_choice())
    
    # Simulate user input (replace this with actual input capture)
    user_input = input("Your choice: ").strip()
    
    # Get the selected material
    material_choice = get_material_choice(user_input)
    
    if material_choice:
        print(f"You chose to learn about {material_choice}.")
        
        # Simulate a query (replace this with actual question from user)
        query = f"Tolong jelaskan tentang {material_choice}."
        
        # Query the selected material's FAISS vector store
        retrieved_docs = rag_manager.query_faiss(material_choice, query)
        
        # Display the retrieved content
        print("\nRelevant content retrieved:")
        for doc in retrieved_docs:
            print(doc.page_content)
    else:
        print("Invalid choice. Please try again.")

# Run the interaction flow
if __name__ == "__main__":
    interact_with_user()
