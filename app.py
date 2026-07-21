from utils import RAGPipeline

print("=" * 60)
print("      Retrieval-Augmented Generation (RAG) Demo")
print("=" * 60)

rag = RAGPipeline()

print("\nLoading documents...")
rag.load_documents("data/knowledge.txt")

print("Generating embeddings...")
rag.create_embeddings()

print("\nRAG System Ready!")

while True:

    query = input("\nAsk a question (type exit to quit): ")

    if query.lower() == "exit":
        break

    chunks = rag.retrieve(query)

    print("\nRetrieved Context\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")

    print("\nGenerated Response\n")

    answer = (
        "Based on the retrieved information:\n\n"
        + " ".join(chunks)
    )

    print(answer)

print("\nGoodbye!")
