from config import *
ignore_warnings()
from services.rag.settings import RagSettings
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama


class Generator:
    """
    A generator class for answer generation. Also acts as a rewriter class 
    that will take the initial input query and transform it into a better 
    input query for higher quality retrieval.

    Attributes:
        settings (RagSettings): The settings for the rewriter models.
        model (ChatOllama): The generative LLM.
        prompt_preparator (str): Preparatory instructions to build the prompt.
        question (str | None): The input query.
        prompt (str): The prompt to query the LLM.
    """

    def __init__(self, settings: RagSettings, 
                 question: str | None = None, 
                 rewriting=False) -> None:
        """
        Constructor.

        Args:
            settings (RagSettings): The settings object.
            question (str | None): The user query (for rewriting).
            rewriting (bool): Rewriter mode flag.

        Returns:
            None
        """
        
        # log activity
        rag_logger.info(f"Initialising {self.__class__.__name__}")

        self.settings: RagSettings = settings # hyperparameter input
        self.model: ChatOllama = ChatOllama(model=self.settings.model_choice) # perhaps this should go on QueryRag
        
        # defining question attribute for rewriting tasks
        self.question = question
        self.rewriting = rewriting

        if self.rewriting:
            # prompt preparator
            prompt_preparator = (
                "Please rewrite the query to maximise retrieval quality."
                "Return one single, concise, rewritten query with no explanations."
                "Preserve intent of the initial query."
            )
            self.prompt: str = self._set_prompt_rewrite(self.question, 
                                                        prompt_preparator)
            
            # log activity
            rag_logger.info(f"Prompt used for rewriting: {self.prompt}")

        else:
            self.prompt: str = self._set_prompt()


    def answer(self, question, context) -> str:
        """
        Main method to generate text based on provided prompt,
        or based on user question if rewriting mode.

        Args:
            question (str | None): The input query.
            context (str): The context upon which generate the answer.

        Returns:
            str
        """
        
        prompt_wrapper = ChatPromptTemplate.from_template(self.prompt)
        prompt = prompt_wrapper.format(context=context, question=question)
        answer = self.model.invoke(prompt)

        # log activity
        if self.rewriting:
            rag_logger.info(f"REWRITTEN query: {answer.content}")
        else:
            rag_logger.info(f"ANSWER to query: {answer.content}")

        return answer.content


    def _set_prompt(self) -> None:
        """
        Sets the final answer's prompt template.

        Returns:
            str
        """
        prompt = """
        Read the below context:
        {context}

        Answer the question based only on the information in the above context: 
        {question}
        """

        return prompt
    
    
    def _set_prompt_rewrite(self, question, prompt_preparator) -> None:
        """
        Builds a rewriter prompt by injecting 
        the prompt_prepper and domain hints.

        Args:
            query (str): The user query.
        
        Returns:
            str
        """

        prompt = (
            f"{prompt_preparator}\n\n"
            f"{self.settings.domain_hints}"
            f"User query: {question}\n"
        )
        return prompt  