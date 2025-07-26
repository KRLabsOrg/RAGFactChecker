# Rag Fact Checking System

A Python library for validating the factual accuracy of Large Language Model (LLM) responses against their source documents in Retrieval-Augmented Generation (RAG) systems. The raw textual inputs are converted into triplets (subject, predicate, object) to represent the sentences and then fact-checked against the reference documents.

This library offers you sentence-level fact checking granularity, with the possibility to extract the exact incorrect triplet from an LLM response.

The generic way of how it works is as follows:
- Generate the triplets from the answer of the LLM
- Generate the triplets from the reference documents
- Compare the triplets from the LLM answer with the triplets from the reference documents
- If the triplets from the LLM answer are not present in the reference documents, it is a hallucination


**NB**: This fact-checking system was built and validated in https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq dataset, specifically on the Thyroid topic. There is no guarantee that it will work for other datasets and might need more adjustments to be used for other datasets.

## Main Features

There are a couple of separate components in this library:

- **Triplet Generator**: Extracts factual relationships from the input text in the form of triplets, which consist of a subject, predicate, and object. This is done using LLMs.
- **Fact Checker**: Compares these triplets with those from a reference text to determine if they match (true/false).
- **Hallucination Generator**: An LLM generator that generates hallucinated triplets from the reference documents. This is done in order to synthetically generate a dataset of hallucinated triplets.
- **Answer-Based Hallucination Generator**: Generates controlled hallucinations by taking correct answers and injecting specific types of errors (factual, temporal, numerical, relational, contextual, omission) with configurable intensity levels.
- **Batch Processing**: All components support concurrent processing of multiple items using threading for improved performance.
- **Async Support**: Full asynchronous support with semaphore-based concurrency limiting for non-blocking operations.

## Installation

This is a pip installable package. To install it, run the following command:

```bash
pip install git+https://github.com/KRLabsOrg/RAGFactChecker.git@main
```
	
## Usage

Below is a sample that shows how to import and use the `LLMTripletValidator` class to execute fact checking. The inputs are:
- `input_text`: The answer from the LLM
- `reference_text`: The reference documents which were fed to the LLM from the RAG system

```python
import os
from rag_fact_checker import LLMTripletValidator
api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key")

triplet_validator = LLMTripletValidator(
  input_config = {"triplet_generator": "llm_n_shot", "fact_checker": "llm_n_shot"},
  openai_api_key=api_key
)

results = triplet_validator.validate_llm_triplets(
   input_text="The sky is green", 
   reference_text=["The sky is blue and the grass is green"]
)
```

Output:
```
DirectTextMatchOutput(
    input_triplets=[['The sky', 'is', 'green']], 
    reference_triplets=[['The sky', 'is', 'blue'], ['The grass', 'is', 'green']], 
    fact_check_prediction_binary={0: False} # this means that input triplet with idx 0 is incorrect
)
```


## Additional usages

Besides being used for fact-checking purposes only, the library can also be used for Information Extraction purposes - namely, given some text, to extract the triplets from it (all using LLMs).

```python
import os
from rag_fact_checker import LLMTripletValidator
api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key")


triplet_validator = LLMTripletValidator(
    input_config = {"triplet_generator": "llm_n_shot", "fact_checker": "llm_n_shot"},
    openai_api_key=api_key
)

results = triplet_validator.triplet_generation(
    input_text="The sky is green and the Eiffel Tower is blue. The Eiffel Tower is in Paris."
)
```

Another usecase is to use it for synthetic hallucinated data generation.

```python
import os
from rag_fact_checker import LLMTripletValidator
api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key")

triplet_validator = LLMTripletValidator(
    input_config = {"triplet_generator": "llm_n_shot", "fact_checker": "llm_n_shot"},
    openai_api_key=api_key
)

results = triplet_validator.generate_hlcntn_data(
    question="Which genes does thyroid hormone receptor beta1 regulate in the liver?",
    reference_text=["The carbohydrate response element-binding protein (ChREBP) and sterol response element-binding protein (SREBP)-1c, regulated by liver X receptors (LXRs), play central roles in hepatic lipogenesis. Because LXRs and thyroid hormone receptors (TRs) influence each other's transcriptional activity, researchers investigated whether TRs control ChREBP expression. They found that thyroid hormone (T3) and TR-beta1 upregulate ChREBP by binding direct repeat-4 elements (LXRE1/2), thereby fine-tuning hepatic lipid metabolism."]
)
```

### Answer-Based Hallucination Generation

For generating controlled hallucinations from correct answers:

```python
from rag_fact_checker.model.hallucination_data_generator import AnswerBasedHallucinationDataGenerator
from rag_fact_checker.data import Config, ErrorType
import logging

config = Config()
logger = logging.getLogger(__name__)
generator = AnswerBasedHallucinationDataGenerator(config, logger)

# Generate hallucination with specific error type and intensity
result = generator.generate_hallucination(
    correct_answer="Paris is the capital of France and has a population of 2.1 million.",
    error_type=ErrorType.FACTUAL,
    intensity=0.7
)
print(result.hallucinated_answer)  # Answer with injected factual errors
```

### Batch Processing

Process multiple items concurrently:

```python
# Batch triplet generation
input_texts = ["Text 1", "Text 2", "Text 3"]
batch_result = triplet_generator.forward_batch(input_texts)

# Batch fact checking
answer_triplets_batch = [[[["Subject", "predicate", "object"]], ...]]
reference_triplets_batch = [[[["Ref", "predicate", "object"]], ...]]
fact_check_result = fact_checker.forward_batch(answer_triplets_batch, reference_triplets_batch)

# Async batch processing
import asyncio
async def process_async():
    result = await triplet_generator.forward_batch_async(input_texts)
    return result
```


## Configuration

When creating the LLMTripletValidator, you can pass a dict to input_config to override default settings.
For example, to customize model names, logging level, etc.:

```python
custom_config = {
    "model": {
        "triplet_generator": {
            "model_name": "llm_n_shot"
        },
        "fact_checker": {
            "model_name": "llm"
        }
    },
    "logger_level": "DEBUG"
}
```

### Available Models
- **Triplet generator**: "llm", "llm_n_shot" 
- **Fact checker**: "llm", "llm_split", "llm_n_shot", "llm_n_shot_split"
- **Hallucination generator**: "llm", "llm_n_shot", "answer_based"
- **Logger level**: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"

### Batch Processing Configuration
```python
batch_config = {
    "simple_batch_config": {
        "max_workers": 5,          # Number of concurrent threads
        "max_concurrent": 10,      # Max concurrent async operations
        "timeout": 30.0           # Timeout per operation in seconds
    }
}
```

### Error Types for Answer-Based Hallucination
- **FACTUAL**: Incorrect facts and information
- **TEMPORAL**: Wrong dates, times, or temporal relationships  
- **NUMERICAL**: Incorrect numbers, quantities, measurements
- **RELATIONAL**: Wrong relationships between entities
- **CONTEXTUAL**: Information out of context or misapplied
- **OMISSION**: Missing critical information
