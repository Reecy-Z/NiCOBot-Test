from langchain.tools import BaseTool


class Query2ReactionSimilarity(BaseTool):
    name = "Query2ReactionSimilarity"
    description = "This tool is used to return a reaction based on molecule SMILES, Input a molecule's SMILES and return reaction information from the dataset based on similarity."

    def __init__(self):
        super().__init__()
    
    def _run(self, query: str) -> str:
        try:
            from generate_candidate_reactions import generate_product

            response = generate_product(query)
            # save_images(response, 'similarity_reactions_E_Nu_Connected')
            return response
        except Exception as e:
            return str(e)
    
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()