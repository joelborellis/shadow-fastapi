from typing import Annotated
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from tools.searchshadow import SearchShadow
from tools.searchcustomer import SearchCustomer


class ShadowInsightsPlugin:
    """Plugin class that accepts a PromptTemplateConfig for advanced configuration."""

    def __init__(
        self, search_shadow_client: SearchShadow, search_customer_client: SearchCustomer
    ):
        """
        :param search_shadow_client: A SearchShadow client used for shadow index searches.
        :param search_customer_client: A SearchCustomer client used for customer index searches.
        """
        self.search_shadow_client = search_shadow_client
        self.search_customer_client = search_customer_client

    @kernel_function(
        name="get_sales_docs",
        description="Given a user query determine if it is a question related to a sales pursuit and search the sales index.",
    )
    def get_sales_docs(
        self, query: Annotated[str, "The query from the user."]
    ) -> Annotated[str, "Returns documents from the sales index."]:
        try:
            # Ensure query is valid
            if not isinstance(query, str) or not query.strip():
                raise ValueError("The query must be a non-empty string.")

            # Perform the search
            docs = self.search_shadow_client.search_hybrid(query)
            if not docs:
                return "No relevant documents found in the sales index."
            return docs
        except ValueError as ve:
            return f"Input error: {ve}"
        except Exception as e:
            return f"An error occurred while retrieving documents from the sales index: {e}"

    @kernel_function(
        name="get_customer_docs",
        description="Given a user query determine if a target pursuit account company name is provided. Use the query and the target pursuit account company name to search the pursuit index.",
    )
    def get_customer_docs(
        self, query: Annotated[str, "The query and the target pursuit account company name provided by the user."]
    ) -> Annotated[str, "Returns documents from the pursuit index."]:
        try:
            # Ensure query is valid
            if not isinstance(query, str) or not query.strip():
                raise ValueError("The query must be a non-empty string.")

            # Perform the search
            docs = self.search_customer_client.search_hybrid(query)
            if not docs:
                return "No relevant documents found in the pursuit index."
            return docs
        except ValueError as ve:
            return f"Input error: {ve}"
        except Exception as e:
            return f"An error occurred while retrieving documents from the pursuit index: {e}"