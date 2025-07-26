from rag_fact_checker.model.fact_checker.llm_fact_checker import LLMFactChecker
from rag_fact_checker.model.fact_checker.llm_multishot_fact_checker import (
    LLMMultiShotFactChecker,
)
from rag_fact_checker.model.fact_checker.llm_multishot_split_fact_checker import (
    LLMMultiShotSplitFactChecker,
)
from rag_fact_checker.model.fact_checker.llm_split_fact_checker import (
    LLMSplitFactChecker,
)
from rag_fact_checker.model.hallucination_data_generator import (
    LLMHallucinationDataGenerator,
    LLMMultiShotHallucinationDataGenerator,
)
from rag_fact_checker.model.triplet_generator.llm_multishot_triplet_generator import (
    LLMMultiShotTripletGenerator,
)
from rag_fact_checker.model.triplet_generator.llm_triplet_generator import (
    LLMTripletGenerator,
)

model_name_class_mapping = {
    "triplet_generator": {
        "llm": LLMTripletGenerator,
        "llm_n_shot": LLMMultiShotTripletGenerator,
    },
    "fact_checker": {
        "llm": LLMFactChecker,
        "llm_split": LLMSplitFactChecker,
        "llm_n_shot": LLMMultiShotFactChecker,
        "llm_n_shot_split": LLMMultiShotSplitFactChecker,
    },
    "hallucination_data_generator": {
        "llm": LLMHallucinationDataGenerator,
        "llm_n_shot": LLMMultiShotHallucinationDataGenerator,
    },
}
__all__ = [
    "LLMTripletGenerator",
    "LLMMultiShotTripletGenerator",
    "LLMFactChecker",
    "LLMSplitFactChecker",
    "LLMMultiShotFactChecker",
    "LLMMultiShotSplitFactChecker",
    "model_name_class_mapping",
]
