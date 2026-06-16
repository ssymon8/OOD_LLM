import torch
import torch.nn.functional as F
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, FineGrainedFP8Config
from tqdm import tqdm
from datasets import load_dataset

import logging
import os
import sys
import argparse
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
                print(f"Token ID for choice '{choice}': {token_id[-1]}")  # Debugging: print the token ID for each choice
            else:
                raise ValueError(f"Token for choice '{choice}' not found in tokenizer.")
        return choice_token_ids

    def zero_shot_format_prompt(self, question: str, choices: list, subject: str): #zero-shot prompt formatting
        prompt = f"You are a evaluation harness. Your task is to answer the following multiple-choice question about {subject}:\n\n."
        prompt+= f"You MUST answer with only the letter corresponding to the correct choice (A, B, C, or D). \n\n"
        prompt += f"Question: {question}\n"
        for idx, choice in enumerate(choices):
            prompt += f"{self.choices[idx]}. {choice}\n"
        prompt += "Answer:"
        return prompt
    
    def five_shot_format_prompt(self, question: str, choices: list, subject: str, dev_set: list): #five-shot prompt formatting
        prompt = f"You are a precise evaluation harness. Your task is to answer the following multiple-choice questions about {subject}:\n\n."
        prompt+= f"You MUST answer with only the letter corresponding to the correct choice (A, B, C, or D).\n\n"
        prompt += "Here 5 example questions with their correct answers, followed by the final question for which you need to provide the correct answer.\n\n"
        for i, sample in enumerate(dev_set):
            prompt += f"Question: Sample question {i+1}?\n"
            prompt += sample["question"] + "\n"
            for idx, choice in enumerate(sample["choices"]):
                prompt += f"{self.choices[idx]}. {choice}\n"
            prompt += f"Answer: {sample['answer']}\n\n"
        
        prompt += f"Final question: {question}\n"
        for idx, choice in enumerate(choices):
            prompt += f"{self.choices[idx]}. {choice}\n"
        prompt += "Answer: "
        return prompt
        
    def get_layer_output(self,layer_idx):
        """
        A simpler hook to just print the output shape of the layer.
        """
        features = {"outputs": []}
    
        def hook(module, input, output):
            features["outputs"].append(output.detach().cpu())
        return hook, features

    def evaluate_subject(self, subject: str, split: str = "test", mode: str = "five-shot") -> float: 
        dataset = load_dataset("cais/mmlu", subject, split=split)
        dev_set = load_dataset("cais/mmlu", subject, split="dev")
        correct = 0
        total = 0

        random_indices = torch.randperm(len(dataset))[:100].tolist()  # Randomly select 100 indices for benchmarking
        test_set = dataset.select(random_indices)

        correct_outputs = [[] for i in range(26)]
        wrong_outputs = [[] for i in range(26)]

        for sample in tqdm(test_set):
            if mode == "zero-shot":
                prompt = self.zero_shot_format_prompt(sample["question"], sample["choices"], sample["subject"])
            elif mode == "five-shot":
                prompt = self.five_shot_format_prompt(sample["question"], sample["choices"], sample["subject"], dev_set)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
                

            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                return_tensors="pt",
                return_dict=True
            ).to(self.device)

            #Registering the hooks here (we'll go with a hook on each layer output to see how the model's opinion changes)
            hooks_and_features = [self.get_layer_output(i) for i in range(26)]
            handles= []
            for i in range(26):
                print(f"registering hook on layer {i}")
                handle = self.model.model.language_model.layers[i].register_forward_hook(hooks_and_features[i][0])
                handles.append(handle)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=inputs["input_ids"].shape[1] + 1,  # Only generate one token for the answer
                    temperature=0.0,
                    do_sample=False
                )

            for handle in handles:
                handle.remove()  # Remove the hook after use

            input_length = inputs["input_ids"].shape[1]
            logits = outputs.logits[:, -1, :]
            print(f"Logits for A, B, C, D: {logits[0, self.choice_ids]}")  # see the logits
            choice_logits = logits[0, self.choice_ids] #we isolate the logits corresponding to the tokens for A, B, C, D

            answer_id = torch.argmax(choice_logits, dim=-1).item()
            print(f"Model's predicted answer: {self.choices[answer_id]}")  # see the model's predicted answer
            print(f"Correct answer: {sample['answer']}")  # see the correct answer

            if answer_id == sample["answer"]:
                correct += 1
                for i, (hook, features) in enumerate(hooks_and_features):
                    correct_outputs[i].append(features["outputs"][0])
            else:
                for i, (hook, features) in enumerate(hooks_and_features):
                    wrong_outputs[i].append(features["outputs"][0])
            total += 1

            del inputs, outputs  # Free up memory
            torch.cuda.empty_cache()  # Clear GPU memory after each sample, just in case

        print(f"Total correct: {correct} out of {total}")
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy for {subject} ({mode}): {accuracy:.4f}")

        correct_outputs_files = [Path(f"./outputs/{subject}/correct_outputs_layer_{i}.pt") for i in range(26)]
        wrong_outputs_file = [Path(f"./outputs/{subject}/wrong_outputs_layer_{i}.pt") for i in range(26)]

        for i in range(26):
            torch.save(correct_outputs[i], correct_outputs_files[i])
            torch.save(wrong_outputs[i], wrong_outputs_files[i])
        return accuracy
    


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description=" Specify model Id and mode for the benchmarking")

    parser.add_argument("--model_id", type=str, default="mistralai/Ministral-3-3B-Base-2512", help="HuggingFace model ID to evaluate")
    parser.add_argument("--mode", type=str, choices=["zero-shot", "five-shot"], default="five-shot", help="Evaluation mode: zero-shot or five-shot")


    return parser.parse_args()

    
def main():
    args = parse_args()

    logger.info("Starting MMLU evaluation...")
    try:
        # env variable for hf cache
        os.environ["HF_HOME"] = "/root/.cache/huggingface"

        MODEL_ID = args.model_id

        mmlu_bench = MMLUBench(MODEL_ID)


        #copied those directly from the MMLU dataset card on HuggingFace
        subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
        
        random_subjects_id = torch.randperm(len(subjects))[:3].tolist()  # Randomly select 5 subjects for benchmarking

        random_subjects = [subjects[i] for i in random_subjects_id]


        scores = []
        for subject in random_subjects:
            logger.info(f"Evaluating subject: {subject}")
            Path(f"./outputs/{subject}").mkdir(parents=True, exist_ok= True)
            subject_score = mmlu_bench.evaluate_subject(subject, split="test", mode=args.mode)
            scores.append(subject_score)

        logger.info("MMLU evaluation completed!")
        logger.info(f"Mean score: {sum(scores) / len(scores):.4f}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()