from rag import RAGManager
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os

rag_manager = RAGManager(embedding_model_api_key=os.getenv("EMBED_API_KEY"))

# Function to offer the user a choice of materials
def offer_material_choice():
    return """
    Halo! Selamat datang di kelas sejarah! Apa yang mau kamu pelajari hari ini? Di kelas ini kami menyediakan beberapa topik, seperti:
    - Peristiwa Rengasdengklok
    - Peristiwa 10 Nopember
    - Konferensi Meja Bundar
    """

def get_material_choice(user_input):
    # normalisasi input ke lowercase
    user_input = user_input.lower()

    # pilihan yang memungkinkan
    choices = {
        "rengasdengklok": "Peristiwa Rengasdengklok",
        "10 nopember": "Peristiwa 10 Nopember",
        "konferensi meja bundar": "Konferensi Meja Bundar"
    }

    # mencari kecocokan dari pilihan
    for keyword, material in choices.items():
        if keyword in user_input:
            return material
    
    # Return None jika tidak match
    return None

# Main Interaction
def interact_with_user():
    print(offer_material_choice())
    
    user_input = input("Topik apa yang mau kamu pelajari? ").strip()
    
    material_choice = get_material_choice(user_input)
    
    if material_choice:
        print(f"Kamu memilih untuk mempelajari tentang {material_choice}.")
        
        while True:
            query = input(f"Apa yang mau kamu tanyakan mengenai {material_choice}? (ketik 'exit' untuk keluar) ").strip()
            
            # Cek apakah user ingin keluar
            if query.lower() in ["exit", "quit"]:
                print("Terima kasih! Sampai jumpa lagi.")
                break
            
            # Loading and querying FAISS
            retrieved_docs = rag_manager.query_faiss(material_choice, query)
            
            # Menyiapkan kontext untuk llm
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            llm = ChatOpenAI(
                api_key=os.getenv("LLM_API_KEY"),
                model="gpt-4o-mini",
                temperature=.7
            )
            
            messages = [
                SystemMessage(content="Kamu seorang guru sejarah. Jawablah pertanyaan yang diberikan sesuai konteks yang diberikan. Jawablah seringkas mungkin, namun jawaban terbatas pada maksimal 3 kalimat. Jika tidak ada data yang berkaitan dengan konteks di bawah, jawab dengan 'Saya tidak tahu'."),
                HumanMessage(content=f"Pertanyaan: {query}\n\nKonteks:\n{context}")
            ]
            
            response = llm.invoke(messages) 
            
            # Output
            print(f"AI's Response: {response.content}")
    else:
        print("Pilihan tidak valid. Silakan coba lagi.")


# Run the interaction flow
if __name__ == "__main__":
    interact_with_user()
