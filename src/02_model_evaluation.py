import torch
import json
import re
import os
import time
import warnings
import gc

import pandas as pd

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field

from huggingface_hub import login 
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
torch.set_num_threads(2)

load_dotenv()
try:
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(hf_token)
    else:
        print("HF_TOKEN not found in environment variables. Please set it in .env file.")
except Exception as e:
    print(f"Failed to log in via token: {e}")

@dataclass
class EvalConfig:
    INPUT_PARQUET: str = "./data/Confidence_calibration_study_dataset.parquet"
    OUTPUT_DIR: str = "./results/"
    CHECKPOINT_DIR: str = "./results/checkpoints/"
      
    BATCH_SIZE: int = 16
    MAX_NEW_TOKENS: int = 128
    TEMPERATURE: float = 0.3
    DO_SAMPLE: bool = True
    TOP_P: float = 0.9
    
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD: float = 0.65
    
    USE_DUAL_GPU: bool = True
    PRIMARY_GPU: int = 0
    SECONDARY_GPU: int = 1
    
    CHECKPOINT_EVERY: int = 100
    RESUME_FROM_CHECKPOINT: bool = True
    
    SAVE_INTERMEDIATE: bool = False
    COMPRESS_OUTPUT: bool = True

    MODELS: List[str] = field(default_factory=lambda: [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Qwen/Qwen3.5-9B",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2-1.5B-Instruct",
            "google/gemma-3-12b-it",
            "google/gemma-7b-it",
            "microsoft/Phi-4-mini-instruct",
            "meta-llama/Llama-3.2-1B",
            "microsoft/phi-4",
            "HuggingFaceH4/zephyr-7b-beta",
            "deepseek-ai/deepseek-llm-7b-chat"
        ])
    def __post_init__(self):
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

config = EvalConfig()

def build_optimized_prompt(question: str, answer_type: str) -> str:
    
    type_instructions = {
        "boolean": "Answer must be exactly 'true' or 'false'.",
        "numeric": "Answer must be a number only (no text).",
        "text": "Answer in one clear sentence.",
    }
    
    instruction = type_instructions.get(answer_type, "")
    
    return f"""Question: {question}

{instruction}

Respond in JSON format with your answer and confidence (0.0 to 1.0):
{{"answer": "your_answer_here", "confidence": 0.85}}

JSON:"""

def parse_response_robust(text: str) -> Tuple[Optional[str], Optional[float]]:   
    json_patterns = [
        r'\{[^{}]*?"answer"[^{}]*?"confidence"[^{}]*?\}',
        r'\{[^{}]*?"confidence"[^{}]*?"answer"[^{}]*?\}',
    ]
    
    for pattern in json_patterns:
        matches = list(re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
        for match in reversed(matches):
            try:
                json_str = match.group()
                json_str = re.sub(r'[\n\r\t]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                
                data = json.loads(json_str)
                
                answer = data.get("answer") or data.get("Answer")
                confidence = data.get("confidence") or data.get("Confidence")
                
                if answer is not None and confidence is not None:
                    return str(answer).strip(), float(confidence)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    
    relaxed_pattern = r'\{[^{}]*answer[^{}]*confidence[^{}]*\}'
    for match in re.finditer(relaxed_pattern, text, re.IGNORECASE):
        try:
            json_str = match.group()
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)
            
            data = json.loads(json_str)
            answer = data.get("answer") or data.get("Answer")
            confidence = data.get("confidence") or data.get("Confidence")
            
            if answer is not None and confidence is not None:
                return str(answer).strip(), float(confidence)
        except:
            continue
    
    answer = None
    confidence = None
    
    lines = text.split('\n')
    for line in lines:
        if answer is None:
            ans_match = re.search(r'["\']?answer["\']?\s*:\s*["\']?([^"\'}\n,]+)["\']?', line, re.IGNORECASE)
            if ans_match:
                answer = ans_match.group(1).strip()
        
        if confidence is None:
            conf_match = re.search(r'["\']?confidence["\']?\s*:\s*([0-9.]+)', line, re.IGNORECASE)
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                except:
                    pass
    
    if answer and confidence:
        return answer, confidence
    
    conf_matches = re.findall(r'\b([0-9]\.[0-9]+)\b', text)
    if conf_matches:
        try:
            confidence = float(conf_matches[-1])
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if sentences:
                answer = sentences[-1][:200]
                return answer, confidence
        except:
            pass
    
    return None, None

class AnswerEvaluator:
    
    def __init__(self, device: str = 'cuda:1'):
        print(f"Loading evaluator on {device}...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
        self.cache = {}
        print("✓ Evaluator ready")
    
    def normalize(self, text: str) -> str:
        if text is None:
            return ""
        text = str(text).lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^(the answer is|answer:|a:)\s*', '', text, flags=re.IGNORECASE)
        return text
    
    def evaluate_boolean(self, pred: str, truth: str) -> int:
        pred, truth = self.normalize(pred), self.normalize(truth)
        
        pos = ['yes', 'true', 'correct', '1', 'right']
        neg = ['no', 'false', 'incorrect', '0', 'wrong']
        
        pred_pos = any(t in pred for t in pos)
        pred_neg = any(t in pred for t in neg)
        truth_pos = any(t in truth for t in pos)
        truth_neg = any(t in truth for t in neg)
        
        if pred_pos == pred_neg:
            return 0
        return int((pred_pos and truth_pos) or (pred_neg and truth_neg))
    
    def evaluate_numeric(self, pred: str, truth: str) -> int:
        pred, truth = self.normalize(pred), self.normalize(truth)
        
        pred_nums = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', pred)
        truth_nums = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', truth)
        
        if not pred_nums or not truth_nums:
            return 0
        
        try:
            pred_val = float(pred_nums[-1].replace(',', ''))
            truth_val = float(truth_nums[0].replace(',', ''))
            
            if abs(pred_val - truth_val) < 0.01:
                return 1
            if truth_val != 0 and abs((pred_val - truth_val) / truth_val) < 0.01:
                return 1
        except (ValueError, ZeroDivisionError):
            pass
        
        return 0
    
    def get_embedding(self, text: str, use_cache: bool = True):
        if use_cache and text in self.cache:
            return self.cache[text]
        emb = self.embedder.encode(text, convert_to_tensor=True)
        if use_cache:
            self.cache[text] = emb
        return emb
    
    def evaluate_semantic(self, pred: str, truth: str) -> float:
        pred, truth = self.normalize(pred), self.normalize(truth)
        if not pred or not truth:
            return 0.0
        
        pred_emb = self.get_embedding(pred, use_cache=False)
        truth_emb = self.get_embedding(truth, use_cache=True)
        return util.cos_sim(pred_emb, truth_emb).item()
    
    def evaluate(self, pred: str, truth: str, answer_type: str) -> int:
        if pred is None or truth is None:
            return 0
        
        pred, truth = self.normalize(pred), self.normalize(truth)
        if not pred or not truth:
            return 0
        
        # Type-specific
        if answer_type == "boolean":
            if self.evaluate_boolean(pred, truth) == 1:
                return 1
        elif answer_type == "numeric":
            if self.evaluate_numeric(pred, truth) == 1:
                return 1
        
        # Substring
        if truth in pred or pred in truth:
            return 1
        
        # Semantic
        if self.evaluate_semantic(pred, truth) >= config.SIMILARITY_THRESHOLD:
            return 1
        
        # Word overlap
        if answer_type == "text":
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are'}
            pred_words = set(pred.split()) - stopwords
            truth_words = set(truth.split()) - stopwords
            if truth_words and len(pred_words & truth_words) / len(truth_words) > 0.5:
                return 1
        
        return 0

class OptimizedInference:
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        print(f"\nLoading {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if config.USE_DUAL_GPU and torch.cuda.device_count() > 1:
            device_map = "auto"
            print(f"  Dual GPU mode")
        else:
            device_map = f"cuda:{config.PRIMARY_GPU}"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "14GB", 1: "14GB"}
        )
        self.model.eval()
        
        print(f"✓ Loaded")
        self._print_memory()
    
    def _print_memory(self):
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(f"cuda:{config.PRIMARY_GPU}") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=config.DO_SAMPLE,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        responses = []
        for i, output in enumerate(outputs):
            new_tokens = output[inputs['input_ids'][i].shape[0]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def cleanup(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"✓ Cleaned up")

@dataclass
class QuestionResult:
    id: int
    dataset: str
    type: str
    model: str
    prediction: str
    confidence: float
    correct: int
    parse_success: bool
    inference_time: float
    
    def to_dict(self) -> Dict:
        return asdict(self)

class CheckpointManager:
    
    def __init__(self, model_name: str, checkpoint_dir: str):
        self.model_name = model_name.replace('/', '_')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_path = self.checkpoint_dir / f"{self.model_name}_checkpoint.json"
        self.results_path = self.checkpoint_dir / f"{self.model_name}_results.parquet"
    
    def save_checkpoint(self, question_idx: int, results: List[QuestionResult]):
        checkpoint = {
            "model": self.model_name,
            "last_question_idx": question_idx,
            "total_results": len(results),
            "timestamp": time.time()
        }
        
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
        
        if results:
            df = pd.DataFrame([r.to_dict() for r in results])
            df.to_parquet(self.results_path)
    
    def load_checkpoint(self) -> Tuple[int, List[QuestionResult]]:
        if not self.checkpoint_path.exists():
            return 0, []
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            if self.results_path.exists():
                df = pd.read_parquet(self.results_path)
                results = [QuestionResult(**row) for _, row in df.iterrows()]
            else:
                results = []
            
            print(f"✓ Resumed from checkpoint: question {checkpoint['last_question_idx']}")
            return checkpoint['last_question_idx'], results
        except Exception as e:
            print(f"⚠ Checkpoint load failed: {e}")
            return 0, []
    
    def clear_checkpoint(self):
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        if self.results_path.exists():
            self.results_path.unlink()

def evaluate_model(
    model_name: str,
    questions_df: pd.DataFrame,
    evaluator: AnswerEvaluator,
    checkpoint_manager: CheckpointManager
) -> pd.DataFrame:
    """Evaluate model"""
    
    print(f"\n{'='*80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*80}")
    
    start_idx, existing_results = (0, []) if not config.RESUME_FROM_CHECKPOINT else checkpoint_manager.load_checkpoint()
    
    inference = OptimizedInference(model_name)
    
    questions = questions_df.to_dict('records')
    total_questions = len(questions)
    results = existing_results
    batch_size = config.BATCH_SIZE
    
    pbar = tqdm(total=total_questions, initial=start_idx, desc=f"Evaluating")
    
    for batch_start in range(start_idx, total_questions, batch_size):
        batch_end = min(batch_start + batch_size, total_questions)
        batch = questions[batch_start:batch_end]
        
        prompts = [build_optimized_prompt(q['question'], q['type']) for q in batch]
        
        start_time = time.time()
        try:
            responses = inference.generate_batch(prompts)
            inference_time = (time.time() - start_time) / len(batch)
        except Exception as e:
            print(f"\n⚠ Batch {batch_start}-{batch_end} failed: {e}")
            pbar.update(len(batch))
            continue
        
        for question, response in zip(batch, responses):
            pred, conf = parse_response_robust(response)
            
            parse_success = pred is not None and conf is not None
            
            if parse_success:
                conf = max(0.0, min(1.0, conf))
                correct = evaluator.evaluate(pred, question['answer'], question['type'])
            else:
                pred = ""
                conf = 0.0
                correct = 0
            
            results.append(QuestionResult(
                id=question['id'],
                dataset=question['dataset'],
                type=question['type'],
                model=model_name,
                prediction=pred,
                confidence=conf,
                correct=correct,
                parse_success=parse_success,
                inference_time=inference_time
            ))
        
        pbar.update(len(batch))
        
        if (batch_end % config.CHECKPOINT_EVERY == 0) or (batch_end == total_questions):
            checkpoint_manager.save_checkpoint(batch_end, results)
    
    pbar.close()
    inference.cleanup()
    
    results_df = pd.DataFrame([r.to_dict() for r in results])
    
    print(f"\n✓ Complete")  
    checkpoint_manager.clear_checkpoint()
    return results_df

def main():
    questions_df = pd.read_parquet(config.INPUT_PARQUET)
    
    evaluator = AnswerEvaluator(device=f'cuda:{config.SECONDARY_GPU}')
    all_results = []
    for model_idx, model_name in enumerate(config.MODELS, 1):
        print(f"\n{'='*80}")
        print(f"MODEL {model_idx}/{len(config.MODELS)}")
        print(f"{'='*80}")
        
        try:
            checkpoint_mgr = CheckpointManager(model_name, config.CHECKPOINT_DIR)
            
            results_df = evaluate_model(model_name, questions_df, evaluator, checkpoint_mgr)

            all_results.append(results_df)

            if config.SAVE_INTERMEDIATE:
                model_short = model_name.replace('/', '_')
                output_path = f"{config.OUTPUT_DIR}{model_short}_results.csv"
                
                if config.COMPRESS_OUTPUT:
                    output_path += ".gz"
                    results_df.to_csv(output_path, index=False, compression='gzip')
                else:
                    results_df.to_csv(output_path, index=False)
            
        except Exception as e:
            print(f"\n✗ Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)
            
    if all_results:
        print(f"\n{'='*80}")
        print("COMBINING RESULTS")
        print(f"{'='*80}")
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        final_path = f"{config.OUTPUT_DIR}evaluation_results_combined.csv"
        if config.COMPRESS_OUTPUT:
            final_path += ".gz"
            combined_df.to_csv(final_path, index=False, compression='gzip')
        else:
            combined_df.to_csv(final_path, index=False)
        
        print(f"✓ Saved combined results: {final_path}")
        print(f"\nFinal Statistics:")
        print(f"  Total predictions: {len(combined_df)}")
        print(f"  Models evaluated: {combined_df['model'].nunique()}")
        print(f"  Overall accuracy: {combined_df['correct'].mean():.3f}")
        print(f"  Parse success rate: {combined_df['parse_success'].mean()*100:.1f}%")
        
        # Per-model summary
        print(f"\nPer-Model Summary:")
        summary = combined_df.groupby('model').agg({
            'correct': 'mean',
            'confidence': 'mean',
            'parse_success': 'mean',
            'id': 'count'
        }).round(3)
        summary.columns = ['Accuracy', 'Avg_Confidence', 'Parse_Success', 'N_Questions']
        summary['Overconf_Gap'] = (summary['Avg_Confidence'] - summary['Accuracy']).round(3)
        print(summary.to_string())
        
        # Save summary
        summary_path = f"{config.OUTPUT_DIR}model_summary.csv"
        summary.to_csv(summary_path)
        print(f"\n✓ Saved summary: {summary_path}")
        
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE!")
        print(f"{'='*80}")        
        return combined_df
    
    else:
        print("\n✗ No results generated")
        return None

torch.manual_seed(62)
main()