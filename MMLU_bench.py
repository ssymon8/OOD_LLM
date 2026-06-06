import torch
import torch.nn.functional as F
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config
from tqdm import tqdm
from datasets import load_dataset

import logging
import os
import sys
from pathlib import Path

class MMLUBench:
    def __init__(self, model_id: str):
        self.model_id = model_id
        try:
            self.device = torch.device("cuda")
        except:
            print("CUDA not available :(.")
            sys.exit(1)

        self.tokenizer = MistralCommonBackend.from_pretrained(model_id)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=FineGrainedFP8Config(dequantize=True)
        ).to(self.device)
        self.model.eval()

        self.choices = ["A", "B", "C", "D"]
        self.choice_ids = self._get_choice_token_ids()
    
    def _get_choice_token_ids(self):
        choice_token_ids = []
        for choice in self.choices:
            token_id = self.tokenizer.encode(choice)
            if token_id is not None:
                choice_token_ids.append(token_id[-1])
            else:
                raise ValueError(f"Token for choice '{choice}' not found in tokenizer.")
        return choice_token_ids

    def zero_shot_format_prompt(self, question: str, choices: list, subject: str): #zero-shot prompt formatting
        prompt = f"The following are multiple choice questions (with answers) about {subject}:\n\n"
        prompt += f"Question: {question}\n"
        for idx, choice in enumerate(choices):
            prompt += f"{self.choices[idx]}. {choice}\n"
        prompt += "Answer:"
        return prompt
    
    def five_shot_format_prompt(self, question: str, choices: list, subject: str): #five-shot prompt formatting
        dev_set = load_dataset("cais/mmlu", subject, split="dev")
        prompt = f"The following are multiple choice questions (with answers) about {subject}:\n\n"
        for i, sample in enumerate(dev_set):
            prompt += f"Question: Sample question {i+1}?\n"
            prompt += sample["question"] + "\n"
            for idx, choice in enumerate(sample["choices"]):
                prompt += f"{self.choices[idx]}. {choice}\n"
            prompt += f"Answer: {sample['answer']}\n\n"
        
        prompt += f"Final question: {question}\n"
        for idx, choice in enumerate(choices):
            prompt += f"{self.choices[idx]}. {choice}\n"
        prompt += "Answer:"
        return prompt

    def evaluate_subject(self, subject: str, split: str = "test", mode: str = "five-shot") -> float: 
        dataset = load_dataset("cais/mmlu", subject, split=split)
        correct = 0
        total = 0

        for sample in tqdm(dataset):
            if mode == "zero-shot":
                prompt = self.zero_shot_format_prompt(sample["question"], sample["choices"], sample["subject"])
            elif mode == "five-shot":
                prompt = self.five_shot_format_prompt(sample["question"], sample["choices"], sample["subject"])
            else:
                raise ValueError(f"Unsupported mode: {mode}")
                

            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                return_tensors="pt",
                return_dict=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=inputs["input_ids"].shape[1] + 512,
                    temperature=0.0,
                    do_sample=False
                )
            
            input_length = inputs["input_ids"].shape[1]

            logits = outputs.logits[:, -1, :]
            choice_logits = logits[:, self.choice_ids]

            answer_id = torch.argmax(choice_logits, dim=-1).item()

            if self.choices[answer_id] == sample["answer"]:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy for {subject} ({mode}): {accuracy:.4f}")
        return accuracy


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting MMLU evaluation...")
    try:
        # env variable for hf cache
        os.environ["HF_HOME"] = "/root/.cache/huggingface"

        MODEL_ID = "mistralai/Ministral-3-3B-Base-2512"

        mmlu_bench = MMLUBench(MODEL_ID)

        #copied those directly from the MMLU dataset card on HuggingFace
        subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']

        scores = []
        for subject in subjects:
            logger.info(f"Evaluating subject: {subject}")
            subject_score = mmlu_bench.evaluate_subject(subject, split="test", mode="five-shot")
            scores.append(subject_score)

        logger.info("MMLU evaluation completed!")
        logger.info(f"Mean score: {sum(scores) / len(scores):.4f}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()