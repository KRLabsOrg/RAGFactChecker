import logging

import dotenv

dotenv.load_dotenv()


class ExperimentLogger(logging.Logger):
    def __init__(self, name, log_path: str, logger_level="INFO"):
        super().__init__("")
        self.log_path = log_path
        self.configure_logger(logger_level=logger_level)

    def configure_logger(self, logger_level="INFO"):
        if logger_level == "DEBUG":
            logger_level = logging.DEBUG
        elif logger_level == "INFO":
            logger_level = logging.INFO
        else:
            logger_level = getattr(logging, logger_level)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logger_level)
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)


def override_config(default_config, config):
    if hasattr(config, "logger_level"):
        default_config.logger_level = config.logger_level
    else:
        default_config.logger_level = "INFO"

    if hasattr(config, "triplet_generator"):
        default_config.model.triplet_generator.model_name = config.triplet_generator

    if hasattr(config, "fact_checker"):
        default_config.model.fact_checker.model_name = config.fact_checker

    if hasattr(config, "hallucination_data_generator"):
        default_config.model.hallucination_data_generator.model_name = (
            config.hallucination_data_generator
        )
    return default_config


DEFAULT_CONFIG = {
    "experiment_setup": {"system_retry": 2, "log_prompts": False},
    "model": {
        "triplet_generator": {
            "model_name": "llm_n_shot",
            "model_params": {"openie.affinity_probability_cap": 0.6},
            "num_shot": 3,
        },
        "fact_checker": {
            "model_name": "llm",
            "split_reference_triplets": True,
            "max_reference_triplet_length": 100,
            "num_shot": 2,
            "inquiry_mode": True,
        },
        "llm": {
            "generator_model": "gpt-4o",
            "request_max_try": 1,
            "temperature": 0,
        },
        "hallucination_data_generator": {"model_name": "llm_n_shot", "num_shot": 2},
    },
    "path": {
        "data": {
            "base": "rag_fact_checker/data/",
            "demo": "demonstrations",
        },
        "prompts": "rag_fact_checker/prompt_bank.json",
    },
}


PROMPT_BANK = {
    "system": {
        "triplet_generation_instruction": {
            "format": 'You are an AI assistant specializing in extracting key information and logic from text and converting them into structured triplets in the form of ["subject", "predicate", "object"]. Your goal is to produce triplets that are self-contained, fully contextualized, and accurately represent the core information in the input text. Follow these requirements:\n\t1.\tFull Context in Subjects and Objects:\nEach subject and object must include sufficient context so they can stand alone without needing to refer back to the original text. Avoid vague references, pronouns, or terms like ‘changes’ without specifying what changes they refer to. For example, instead of ["changes", "are", "transient"] use something like ["the changes in thyroid hormone metabolism induced by strenuous physical activity", "are", "transient and minor"].\n\t2.\tAccurate Representation of Information:\nCapture the main factual statements or logical relationships from the input text. If the text states that T3 reduces diastolic calcium levels, the triplet should clearly reflect that relationship.\n\t3.\tGrammatical and Logical Consistency:\nEnsure correct grammar and logical consistency in the triplets so that each one can be understood independently. The predicate should accurately connect the subject and object, and maintain the meaning given in the input text.\n\t4.\tNo Additional Explanation or Formatting:\nOnly output the resulting triplets. Do not include extra commentary, explanations, or formatting beyond the required triplets array.',
            "input_params": [],
        },
        "n_shot_triplet_generation_instruction": {
            "format": 'You are an AI assistant specializing in extracting key information and logic from text.\nYour task is to convert the input text into a series of structured triplets in the form of ["subject", "predicate", "object"].\nThese triplets should accurately represent the core information and logic of the text while being suitable for comparison with triplets generated from other texts.\nEnsure the triplets are concise, consistent, and adhere to the specified format for clear comparison.\n\nIf few-shot demonstrations are provided, use them to guide the extraction of triplets.\nIf no demonstartions are provided, proceed based only on the input text.\nLastly, only output the resulting triplets without any additional explanation or formatting.',
            "input_params": [],
        },
        "triplet_match_test_instruction": {
            "format": "You are an assistant tasked with comparing two sets of triplets.\nFor each triplet in the input triplets, determine whether there is a similar triplet in the source triplets or whether the input triplet can be logically inferred from the source triplets.\nConsider paraphrased information, contextual clues (e.g., pronouns or synonyms), and combinations of source triplet information to make this determination.\n\nProvide the results in the following format:\ntriplet_idx:result\nWhere result can be one of the following:\n - True: The input triplet is either highly similar to a triplet in the source or can be logically inferred from the source triplets.\n - False: The input triplet cannot be matched or inferred from any triplet in the source.\n\nBe concise and only output the results as specified.",
            "input_params": [],
        },
        "n_shot_triplet_match_test_instruction": {
            "format": "You are an assistant tasked with determining whether each input triplet is supported by the source triplets. For each input triplet, decide whether it is True or False based on the following rules:\n\nTrue:\n • Choose True only if the input triplet is fully supported by the source triplets.\n • The support can be direct or can be logically inferred from information present in the source triplets without requiring speculation.\n • The inference must be straightforward and firmly grounded in the information given.\n • The subject, predicate, and object must directly correspond to, or be clearly inferable from, the source triplets.\n\nFalse:\n • If the input triplet introduces details not present in the source triplets (fabrication), choose False.\n • If the input triplet conflicts with information found in the source triplets, choose False.\n • If the input triplet modifies details from the source triplets in a way not clearly supported (e.g., adding conditions that are not stated, changing numbers or specific data), choose False.\n • If there is any ambiguity, guesswork, or uncertainty, choose False.\n • If you cannot confidently match or infer the exact relationship described in the input triplet from the source triplets, choose False.\n\nAdditional Instructions:\n 1. Err on the side of False:\nUnless you are certain that the input triplet’s information is clearly derived from the source triplets, choose False.\n 2. No speculation or loose interpretation:\nDo not assume relationships or details that are not clearly indicated. If something is not explicit or clearly inferable from the source triplets, choose False.\n 3. Apply strict standards:\nThe presence of related topics or partially matching information is not enough. The triplet’s facts must align closely and unambiguously with the source content.\n 4. Few-Shot Demonstrations:\nIf provided, carefully follow their approach. If they demonstrate caution and only assign True when the evidence is indisputable, follow that pattern.\n\nOutput Format:\nFor each input triplet at index i (following the indexing provided in the user prompt), output one line:\ntriplet_idx:True or triplet_idx:False\nNo additional explanation or text beyond this format.",
            "input_params": [],
        },
        "triplet_match_test_inquiry_instruction": {
            "format": "You are an assistant responsible for verifying whether each input triplet is supported by the source triplets. For each input triplet, you must decide whether it is True or False, strictly based on the source triplets. Follow these detailed rules:\n\n1. True Condition\n\t•\tExact or Strictly Equivalent Match:\nIf the input triplet directly quotes or very closely paraphrases the source with the same meaning (including specific data, facts, or relationships), choose True.\n\t•\tNumeric Data, Names, Key Facts:\nAll specific numbers, measurements, or details must match or stricly equivalent with the source.\n\t•\tLocations, Timeframes, or Qualifiers:\nMust be identical or demonstrably the same.\n\t•\tStraightforward Inference:\nIf it is logically clear from the source triplets that the specific details in the input triplet can be inferred without speculation or guesswork, you may mark True.\n\t•\tExample:\nIf the source says, 'A hormone X is specifically found in both the hippocampus and the cortex,' then 'hormone X is found in the hippocampus' is a valid inference.\n\t•\tBut if the source is significantly more general or omits critical details (e.g., only says 'several hormones' without naming them), do not fill in any specifics on your own.\n\t•\tAllowable Inferences:\n If the input triplet’s statements can be derived by combining or interpreting information already in the source, without speculation, guesswork, or introduction of new details, choose True.\n\n2. False Condition\n\t•\tUnsupported or New Details:\nIf the input triplet introduces any detail (numeric value, location, name, or condition) that the source triplets do not clearly confirm, choose False—even if it is a real-world fact.\n\t•\tContradiction or Mismatch:\nIf any part of the input triplet conflicts with the source triplets, choose False (e.g., different numbers, different subject-object relationships, or the source is more general while the input is overly specific).\n\t•\tSpeculation or Guessing:\nIf you cannot directly verify the triplet or logically deduce it from the source without making an assumption or inference that is not clearly supported, choose False.\n\t•\tContradiction, Mismatch, or Overly Specific\nIf the input triplet contradicts the source or introduces specificity (like an exact time period or specific numeric value) that is not present or cannot be obviously inferred, choose False.\n\t•\tNew or Unsourced Details\nIf the input triplet includes any detail (numbers, conditions, durations, percentages, etc.) that the source triplets do not explicitly mention or clearly imply, choose False—even if it might be factually correct in a real-world sense.\n\n3. Additional Rule for Exactness of Numeric or Specific Details\n\t•\tIf the input triplet specifies a particular quantity, time period, location, or other condition, confirm that the source triplets match it exactly.\n\t•\tEven a slight difference in numeric value or specific wording means False if there is no explicit mention of a range or approximation in the source.\n\n4. Output Format\n\nYour final output must contain exactly two sections in this order:\n\n[REFERRED TRIPLETS]\n\t•\tFor each input triplet, list the source triplets (by ID or index) that support or contradict the input.\n\t•\tProvide a short explanation (a brief chain-of-thought) describing how they led you to choose True or False.\n\t•\tExample:\n\ntriplet_idx_1: (source triplets #1, #3) → [very brief reasoning]\ntriplet_idx_2: None\n...\n\n[FINAL ANSWER]\n\t•\tFor each input triplet, output one line in the format triplet_idx:True or triplet_idx:False.\n\t•\tNo additional explanation or text beyond these lines.\n\t•\tThe number of lines here must match the number of input triplets exactly.\n\n5. Instructions to Follow Carefully\n\t1.\tCompare each input triplet with the source triplets in detail.\n\t2.\tDecide True or False using the above rules and be very strict about numeric data, specific locations, times, or qualifiers.\n\t3.\tSummarize your brief reasoning in [REFERRED TRIPLETS], referencing which source triplets you used.\n\t4.\tIn [FINAL ANSWER], list only triplet_idx:True or triplet_idx:False for each input triplet.\n\t5.\tProvide no other output besides these two sections.\n\t6.\tEnsure the length of [REFERRED TRIPLETS] and [FINAL ANSWER] matches the count of input triplets.",
            "input_params": [],
        },
        "n_shot_triplet_match_test_inquiry_instruction": {
            "format": "You are an assistant responsible for verifying whether each input triplet is supported by the source triplets. For each input triplet, you must decide whether it is True or False, strictly based on the source triplets. Follow these detailed rules:\n\n1. True Condition\n\t•\tExact or Strictly Equivalent Match:\nIf the input triplet directly quotes or very closely paraphrases the source with the same meaning (including specific data, facts, or relationships), choose True.\n\t•\tNumeric Data, Names, Key Facts:\nAll specific numbers, measurements, or details must match or stricly equivalent with the source.\n\t•\tLocations, Timeframes, or Qualifiers:\nMust be identical or demonstrably the same.\n\t•\tStraightforward Inference:\nIf it is logically clear from the source triplets that the specific details in the input triplet can be inferred without speculation or guesswork, you may mark True.\n\t•\tExample:\nIf the source says, 'A hormone X is specifically found in both the hippocampus and the cortex,' then 'hormone X is found in the hippocampus' is a valid inference.\n\t•\tBut if the source is significantly more general or omits critical details (e.g., only says 'several hormones' without naming them), do not fill in any specifics on your own.\n\t•\tAllowable Inferences:\n If the input triplet’s statements can be derived by combining or interpreting information already in the source, without speculation, guesswork, or introduction of new details, choose True.\n\n2. False Condition\n\t•\tUnsupported or New Details:\nIf the input triplet introduces any detail (numeric value, location, name, or condition) that the source triplets do not clearly confirm, choose False—even if it is a real-world fact.\n\t•\tContradiction or Mismatch:\nIf any part of the input triplet conflicts with the source triplets, choose False (e.g., different numbers, different subject-object relationships, or the source is more general while the input is overly specific).\n\t•\tSpeculation or Guessing:\nIf you cannot directly verify the triplet or logically deduce it from the source without making an assumption or inference that is not clearly supported, choose False.\n\t•\tContradiction, Mismatch, or Overly Specific\nIf the input triplet contradicts the source or introduces specificity (like an exact time period or specific numeric value) that is not present or cannot be obviously inferred, choose False.\n\t•\tNew or Unsourced Details\nIf the input triplet includes any detail (numbers, conditions, durations, percentages, etc.) that the source triplets do not explicitly mention or clearly imply, choose False—even if it might be factually correct in a real-world sense.\n\n3. Additional Rule for Exactness of Numeric or Specific Details\n\t•\tIf the input triplet specifies a particular quantity, time period, location, or other condition, confirm that the source triplets match it exactly.\n\t•\tEven a slight difference in numeric value or specific wording means False if there is no explicit mention of a range or approximation in the source.\n\n4. Output Format\n\nYour final output must contain exactly two sections in this order:\n\n[REFERRED TRIPLETS]\n\t•\tFor each input triplet, list the source triplets (by ID or index) that support or contradict the input.\n\t•\tProvide a short explanation (a brief chain-of-thought) describing how they led you to choose True or False.\n\t•\tExample:\n\ntriplet_idx_1: (source triplets #1, #3) → [very brief reasoning]\ntriplet_idx_2: None\n...\n\n[FINAL ANSWER]\n\t•\tFor each input triplet, output one line in the format triplet_idx:True or triplet_idx:False.\n\t•\tNo additional explanation or text beyond these lines.\n\t•\tThe number of lines here must match the number of input triplets exactly.\n\n5. Instructions to Follow Carefully\n\t1.\tCompare each input triplet with the source triplets in detail.\n\t2.\tDecide True or False using the above rules and be very strict about numeric data, specific locations, times, or qualifiers.\n\t3.\tSummarize your brief reasoning in [REFERRED TRIPLETS], referencing which source triplets you used.\n\t4.\tIn [FINAL ANSWER], list only triplet_idx:True or triplet_idx:False for each input triplet.\n\t5.\tFew-Shot Usage:If few-shot demonstrations are provided, use them as a guide to understand the expected style, complexity, and level of detail.\n\t6.\tProvide no other output besides these two sections.\n\t7.\tEnsure the length of [REFERRED TRIPLETS] and [FINAL ANSWER] matches the count of input triplets.",
            "input_params": [],
        },
        "hallucinated_data_generation_test_instruction": {
            "format": "You are HallucinationDataGenerator, an assistant specialized in creating subtle, plausible hallucinations within your responses. Your task is to generate two types of answers based on the provided reference documents and directions:\n\t1.\tNon-Hallucinated Answer:\n\t•\tProduce a detailed, evidence-based answer that is fully faithful to the provided references.\n\t•\tThe answer should not only state conclusions but also present reasoning, background details, and relevant evidence found in the references to support the claims. It should be comprehensive and not overly brief.\n\t2.\tHallucinated Answer:\n\t•\tTake the Non-Hallucinated Answer and incorporate subtle hallucinations. Apart from these hallucinated details, the Hallucinated Answer should remain identical to the Non-Hallucinated Answer.\n\t•\tThe hallucinations can involve small temporal changes, changes in subject-object relationships, non-factual constraints, or unrelated subjects. These fictional details should be subtle and blend naturally with the factual text, but still be clearly marked within the answer (e.g., italics or a parenthetical note).\n\t•\tAfter the Hallucinated Answer, provide a 'Hallucinated Details' section and list each fictional fact as a separate bullet point.\n\nEnsure that both answers are coherent, relevant, and credible, with the Non-Hallucinated Answer providing thorough justification and evidence from the references, and the Hallucinated Answer differing only in the introduced hallucinations.",
            "input_params": [],
        },
        "n_shot_hallucinated_data_generation_test_instruction": {
            "format": "You are HallucinationDataGenerator, an assistant specialized in creating subtle, plausible hallucinations within your responses. Your task is to generate answers that are primarily grounded in the provided reference documents and directions, but also incorporate carefully crafted, believable fictional elements. These hallucinations should not be outlandish; instead, focus on small details that could easily be overlooked—such as specific years, dosage values, or timeframes. For instance, you might slightly alter a reported year, introduce a modest yet unverified numerical detail, or specify a plausible interval for symptom onset that isn’t explicitly stated. Ensure that the hallucinated details blend seamlessly with the given context and do not contradict major facts in the reference documents. Maintain coherence, relevance, and credibility throughout your response.\n\n If few-shot demonstration examples are provided, use them as a guide to understand the style, approach, and complexity expected in the hallucinated output. You may adopt a similar manner of integrating subtle fictional details as demonstrated in the examples.",
            "input_params": [],
        },
    },
    "human": {
        "triplet_generation": {
            "format": 'Input Text: {input_text}\nTask: Extract the key information and logic from the provided text and convert it into a series of triplets in the form of ["subject", "predicate", "object"]. Ensure the triplets are fully contextualized, self-contained, grammatically clear, and reflect the text’s meaning accurately. Do not include any additional explanation or formatting besides the triplets.',
            "input_params": ["input_text"],
        },
        "n_shot_triplet_generation": {
            "format": 'Input Text: {input_text}\nTask: Extract the key information and logic from the provided text and convert it into a series of triplets in the form of ["subject", "predicate", "object"].\nEnsure the triplets are structured in a way that allows comparison with triplets from other texts to identify common or overlapping information.\n\n(Optional) Few-Shot Demonstrations:\nIf few-shot examples are provided here, they will look like this:\n\n[BEGIN FEW-SHOT-EXAMPLES]\n<Example 1 Input/Output Pair>\n<Example 2 Input/Output Pair>\n...\n[END FEW-SHOT-EXAMPLES]\nIf these examples are present, incorporate their style and approach into your solution.{examples}\n\nLastly, only output the resulting triplets without any additional explanation or formatting.',
            "input_params": ["input_text", "examples"],
        },
        "triplet_match_test": {
            "format": "Task Description:\nCompare the input triplets against the source triplets to determine if each input triplet is either highly similar to a source triplet or can be logically inferred from the source triplets. Consider paraphrasing, contextual changes, and indirect references such as pronouns or synonyms. Output True if the triplet matches or is inferable; otherwise, output False.\n\nInput Triplets:\n{answer_triplets}\n\nSource Triplets:\n{reference_triplets}\n\n",
            "input_params": ["directions", "answer_triplets", "reference_triplets"],
        },
        "n_shot_triplet_match_test": {
            "format": "Task Description:\nCompare each input triplet to the provided source triplets. Determine whether the input triplet is fully and clearly supported by the source triplets. Consider that:\n • Introducing new information not found in the source should yield False.\n • Contradicting or altering facts from the source should yield False.\n • Adding unstated conditions or changing details (like numbers or qualifiers) in a way not supported by the source should yield False.\n • Only choose True when you can confidently confirm the input triplet is directly stated or can be straightforwardly inferred from the source triplets without speculation.\n\n(Optional) Few-Shot Demonstrations:\nIf few-shot examples are provided here, they will appear as follows:\n\n[BEGIN FEW-SHOT-EXAMPLES]\n<Example 1 Input/Output Pair>\n<Example 2 Input/Output Pair>\n…\n[END FEW-SHOT-EXAMPLES]\n\nIf these examples are present, follow their style and caution. When in doubt, choose False.\n{examples}\n\nInput Triplets:\n{answer_triplets}\n\nSource Triplets:\n{reference_triplets}\n\nRemember:\n • Choose True only if there is no doubt and the triplet is clearly supported or inferable from the source.\n • If uncertain, or if the input triplet introduces unsupported, conflicting, or altered details, choose False.\n\nNo additional explanation—only lines of the form triplet_idx:True or triplet_idx:False.",
            "input_params": [
                "directions",
                "answer_triplets",
                "reference_triplets",
                "examples",
            ],
        },
        "triplet_match_test_inquiry": {
            "format": "Task Description:\nCompare each input triplet to the provided source triplets. Following the revised system prompt instructions, determine whether each input triplet is supported (True) or not supported (False).\n\nKey Reminders from the System Prompt:\n\t•\tIf the input triplet introduces details (numeric values, specific conditions, or qualifiers) not explicitly supported by the source triplets, you must mark it False, even if it might be true in reality.\n\t•\tIf the input triplet has any mismatch in numbers, times, measurements, or specificity beyond what the source triplets state, mark it False.\n\t•\tFor detailed fact with numbers, partial or approximate matches are insufficient; all details must exactly or straightforwardly match.\n\nInput Triplets:\n{answer_triplets}\n\nSource Triplets:\n{reference_triplets}\n\nOutput Requirements:\n\t1.\tProvide only the two sections [REFERRED TRIPLETS] and [FINAL ANSWER].\n\t2.\tUnder [REFERRED TRIPLETS], for each input triplet, specify which source triplets (if any) were used, along with a brief explanation of how they support or contradict the input.\n\t3.\tUnder [FINAL ANSWER], output exactly one line per input triplet in the format:\n\ntriplet_idx:True\ntriplet_idx:False\n\n\t4.\tNo further explanations or additional text should be included outside these two sections.\n\t5.\tThe number of lines in [FINAL ANSWER] must match the number of input triplets.\n\nFollow the revised system prompt carefully to decide True or False for each input triplet.",
            "input_params": [
                "directions",
                "answer_triplets",
                "reference_triplets",
                "examples",
            ],
        },
        "n_shot_triplet_match_test_inquiry": {
            "format": "Task Description:\nCompare each input triplet to the provided source triplets. Following the revised system prompt instructions, determine whether each input triplet is supported (True) or not supported (False).\n\nKey Reminders from the System Prompt:\n\t•\tIf the input triplet introduces details (numeric values, specific conditions, or qualifiers) not explicitly supported by the source triplets, you must mark it False, even if it might be true in reality.\n\t•\tIf the input triplet has any mismatch in numbers, times, measurements, or specificity beyond what the source triplets state, mark it False.\n\t•\tFor detailed fact with numbers, partial or approximate matches are insufficient; all details must exactly or straightforwardly match.\n\nInput Triplets:\n{answer_triplets}\n\nSource Triplets:\n{reference_triplets}\n\nOutput Requirements:\n\t1.\tProvide only the two sections [REFERRED TRIPLETS] and [FINAL ANSWER].\n\t2.\tUnder [REFERRED TRIPLETS], for each input triplet, specify which source triplets (if any) were used, along with a brief explanation of how they support or contradict the input.\n\t3.\tUnder [FINAL ANSWER], output exactly one line per input triplet in the format:\n\ntriplet_idx:True\ntriplet_idx:False\n\n\t4.\tNo further explanations or additional text should be included outside these two sections.\n\t5.\tThe number of lines in [FINAL ANSWER] must match the number of input triplets.\n\nFollow the revised system prompt carefully to decide True or False for each input triplet.\n\n(Optional) 6.Few-Shot Demonstrations:\nIf few-shot examples are provided here, they will look like this:\n\n[BEGIN FEW-SHOT-EXAMPLES]\n<Example 1 Input/Output Pair>\n<Example 2 Input/Output Pair>\n...\n[END FEW-SHOT-EXAMPLES]\nIf these examples are present, incorporate their style and approach into your solution.{examples}",
            "input_params": ["answer_triplets", "reference_triplets", "examples"],
        },
        "hallucinated_data_generation_test": {
            "format": "Reference Document:\n{reference_documents}\n\nQuestion:\n{question}\n\nTask:\n\t1.\tNon-Hallucinated Answer:\n\t•\tProduce a comprehensive, evidence-based answer to the question using the provided references.\n\t•\tInclude reasoning, background context, and supporting evidence from the references, making sure the answer is not overly brief.\n\t2.\tHallucinated Answer:\n\t•\tStart with the exact same text as the Non-Hallucinated Answer.\n\t•\tIntroduce subtle hallucinations that are small and credible but factually incorrect or not supported by the reference. These may include:\n\t•\tSlight alterations in temporal details.\n\t•\tChanges in the relationship between subjects and objects.\n\t•\tNon-factual constraints or unrelated subjects introduced subtly.\n\t•\tHighlight each hallucinated detail in the text (e.g., italics or a parenthetical note).\n\t•\tApart from the hallucinated elements, the rest of the Hallucinated Answer should remain identical to the Non-Hallucinated Answer.\n\t3.\tHallucinated Details Section:\n\t•\tAfter the Hallucinated Answer, list each hallucinated fact as a separate bullet point under a 'Hallucinated Details' heading, clearly identifying the fabricated elements.\n\nFormat Example:\nNon-Hallucinated Answer:\n[Comprehensive, evidence-based answer here, with no hallucinations]\n\nHallucinated Answer:\n[Identical to Non-Hallucinated Answer except where hallucinated details are introduced and highlighted]\n\nHallucinated Details:\n\t•\t[List each hallucinated fact here as a bullet point]",
            "input_params": ["directions", "reference_documents", "question"],
        },
        "n_shot_hallucinated_data_generation_test": {
            "format": "\n\nReference Document:\n{reference_documents}\n\n(Optional) Few-Shot Demonstrations:\nIf few-shot examples are provided here, they will look like this:\n\n[BEGIN FEW-SHOT-EXAMPLES]\n<Example 1 Input/Output Pair>\n<Example 2 Input/Output Pair>\n…\n[END FEW-SHOT-EXAMPLES]\n\nIf these examples are present, incorporate their style and approach into your solution.\n{examples}\n\nQuestion:\n{question}\n\nTask:\n 1. Non-Hallucinated Answer:\n • Produce a comprehensive, evidence-based answer to the question using the provided references.\n • Include reasoning, background context, and supporting evidence from the references, making sure the answer is not overly brief.\n 2. Hallucinated Answer:\n • Start with the exact same text as the Non-Hallucinated Answer.\n • Introduce subtle hallucinations that are small, credible, and closely related to the context found in the references. These hallucinations should be challenging to detect without carefully checking the provided references. For instance, slightly alter a date, a name, a relationship between entities, or introduce a minor detail that sounds plausible but does not appear in the references.\n • Highlight each hallucinated detail in the text (e.g., italics or a parenthetical note).\n • Apart from the hallucinated elements, the rest of the Hallucinated Answer should remain identical to the Non-Hallucinated Answer.\n 3. Hallucinated Details Section:\n • After the Hallucinated Answer, list each hallucinated fact as a separate bullet point under a 'Hallucinated Details' heading, clearly identifying the fabricated elements.\n\nFormat Example:\n\nNon-Hallucinated Answer:\n[Comprehensive, evidence-based answer here, with no hallucinations]\n\nHallucinated Answer:\n[Identical to Non-Hallucinated Answer except where subtle, contextually plausible hallucinated details are introduced and highlighted]\n\nHallucinated Details:\n• [List each hallucinated fact here as a bullet point]",
            "input_params": [
                "reference_documents",
                "question",
                "examples",
            ],
        },
    },
}
