from sentence_transformers import models, losses
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from datetime import datetime

from typing import Any, List

import gzip
import csv


class SimCSE:

    def __init__(self, model_name: str) -> None:

        self.output_dir = "output/simcse_{}-{}".format(self.model_name.replace("/", "_"),  
                                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self.embedding_model = models.Transformer(model_name, max_seq_length=124)
        self.pooling_model = models.Pooling(self.embedding_model.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[self.embedding_model, self.pooling_model])

        
    def set_datasets(self, train_path: str, dev_path: str) -> Any:
    
        self.training_dataset = []
        self.development_dataset = []

        with open(train_path, 'r', encoding='utf8') as f_train:
            for line in f_train:
                line = line.strip()
                if len(line) >= 10:
                    self.training_dataset.append(InputExample(texts=[line, line]))


        with gzip.open(dev_path, 'r', encoding='utf8') as f_dev:
            reader = csv.DictReader(f_dev, delimite='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                if row['split'] == 'dev':
                    score = float(row['score']) / 5.0
                    self.development_dataset.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        
        self.dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(self.development_dataset, batch_size=16, name='sts-eval')
        

        return self


    def train(self, epochs: int, batch_size: int, info_steps: int) -> Any:

        if not self.training_dataset or len(self.training_dataset) == 0:
            raise Exception("Please, provide training and development datasets using .set_datasets()")

        if not self.dev_evaluator:
            raise Exception("Something wrong with the Dev Evaluator.") 
            
                   

        return self

    def save(self) -> Any:
        
        return self