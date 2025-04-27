import os
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from utils.clean_text import clean_text

load_dotenv()


class SearchCustomer:

    def __init__(self):
        try:
            # assign the Search variables for Azure Cogintive Search - use .env file and in the web app configure the application settings
            AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
            AZURE_SEARCH_ADMIN_KEY = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
            AZURE_SEARCH_INDEX_CUSTOMER = os.environ.get("AZURE_SEARCH_INDEX_CUSTOMER")
            OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL")

            if (
                not AZURE_SEARCH_ENDPOINT
                or not AZURE_SEARCH_ADMIN_KEY
                or not AZURE_SEARCH_INDEX_CUSTOMER
                or not OPENAI_EMBED_MODEL
            ):
                raise EnvironmentError(
                    "Missing one or more environment variables required for initialization."
                )

            credential_search = AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY)

            self.sc = SearchClient(
                endpoint=AZURE_SEARCH_ENDPOINT,
                index_name=AZURE_SEARCH_INDEX_CUSTOMER,
                credential=credential_search,
            )
            self.model = OPENAI_EMBED_MODEL
            self.openai_client = OpenAI()

            print(
                f"[SearchCustomer]:  Init SearchCustomer for index - {AZURE_SEARCH_INDEX_CUSTOMER}"
            )
        except Exception as e:
            raise RuntimeError(f"Error initializing SearchCustomer: {e}")

    def get_embedding(self, text, model):
        try:
            text = text.replace("\n", " ")
            return (
                self.openai_client.embeddings.create(input=[text], model=model)
                .data[0]
                .embedding
            )
        except OpenAIError as ai_err:
            ai_response_msg = ai_err.body["message"]
            print(ai_response_msg)
            pass  # (optional)

    def search_hybrid(self, query: str) -> str:
        try:
            vector_query = VectorizedQuery(
                vector=self.get_embedding(query, self.model),
                k_nearest_neighbors=5,
                fields="text_vector",
            )
            results = []

            r = self.sc.search(
                search_text=query,  # set this to engage a Hybrid Search
                vector_queries=[vector_query],
                #select=["category", "sourcefile", "content"],
                select=["title", "chunk"],
                top=3,
            )
            for doc in r:
                results.append(
                    doc["title"] + ":  " +  doc["chunk"]
                )
            return "\n".join(results)
        except Exception as e:
            raise RuntimeError(f"Error performing hybrid search: {e}")
