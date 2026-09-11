import unittest

from src.config import (
    LLM_KEEP_ALIVE,
    LLM_NUM_CTX,
    LLM_NUM_GPU,
    LLM_NUM_PREDICT,
    LLM_NUM_THREAD,
)
from src.llm import get_llm


class LlmConfigurationTests(unittest.TestCase):
    def test_llm_uses_configured_performance_options(self):
        llm = get_llm()

        self.assertEqual(LLM_NUM_CTX, llm.num_ctx)
        self.assertEqual(LLM_NUM_PREDICT, llm.num_predict)
        self.assertEqual(LLM_NUM_THREAD, llm.num_thread)
        self.assertEqual(LLM_NUM_GPU, llm.num_gpu)
        self.assertEqual(LLM_KEEP_ALIVE, llm.keep_alive)

    def test_model_and_temperature_can_be_overridden(self):
        llm = get_llm(model_name="modelo-teste", temperature=0.2)

        self.assertEqual("modelo-teste", llm.model)
        self.assertEqual(0.2, llm.temperature)


if __name__ == "__main__":
    unittest.main()
