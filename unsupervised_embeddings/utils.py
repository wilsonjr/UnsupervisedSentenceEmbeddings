import csv 
import gzip

from sentence_transformers import SentenceTransformers, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from datetime import datetime

def evaluate_embeddings(model_path: str, dataset_path: str, delimiter: str='\t') -> None:

    samples = []

    with gzip.open(dataset_path, 'r', encoding='utf8') as f_eval:
        reader = csv.DictReader(f_eval, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'test':
                score = float(row['score'])
                samples.append(InputExample(texts=[row['sentence1'], row['sentence1']], label=score))
    
    model = SentenceTransformers(model_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(samples, batch_size=16, name='eval-output')
    test_evaluator(model, output_path=model_path)
    