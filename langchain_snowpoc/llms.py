import logging
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage
from snowflake import snowpark
from snowflake.connector import DictCursor
from snowflake.connector.connection import SnowflakeConnection
from snowflake.cortex import Complete

logger = logging.getLogger(__name__)


class Cortex(LLM):
    connection: SnowflakeConnection = None

    model: str = "mistral-7b"

    @property
    def _llm_type(self) -> str:
        return "cortex"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        res = Complete(
            self.model,
            prompt,
            snowpark.Session.builder.configs({"connection": self.connection}).create(),
        )
        return res

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"connection": self.connection, "model": self.model}


class SQLCortex(LLM):
    connection: SnowflakeConnection = None

    model: str = "mistral-7b"

    @property
    def _llm_type(self) -> str:
        return "sqlcortex"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        q = f"""SELECT SNOWFLAKE.CORTEX.COMPLETE('{self.model}', %(prompt)s) as COMPLETION"""

        res = list(self.connection.cursor(DictCursor).execute(q, {"prompt": prompt}))[
            0
        ]["COMPLETION"]
        return res

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "connection": self.connection,
            "model": self.model,
        }
