import csv 
import gzip

import umap
import pandas as pd

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from datetime import datetime

from .mlm import MaskedLanguageModeling
from .sim_cse import SimCSE

def evaluate_embeddings(model_path: str, dataset_path: str, delimiter: str='\t') -> None:

    samples = []
    embedding_sentences = []

    with open(dataset_path, 'rt', encoding='utf8') as f_eval:
        reader = csv.DictReader(f_eval, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'test':
                score = float(row['score'])
                samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

            embedding_sentences.append(row['sentence2'])
    
    model = SentenceTransformer(model_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(samples, batch_size=16, name='eval-output')
    test_evaluator(model, output_path=model_path)

    embeddings = model.encode(embedding_sentences)
    embeddings2d = umap.UMAP().fit_transform(embeddings)

    df = pd.DataFrame({
        'Name': model_path, 
        'UMAP 1': embeddings2d[:, 0],
        'UMAP 2': embeddings2d[:, 1]
    })
    df.to_csv(model_path+'projection.csv', index=False)



def consecutive_training(train_path: str, mlm_dev_path: str, sim_cse_dev_path: str, model_name: str='distilbert-base-uncased', mlm_epochs: int=3, sim_cse_epochs: int=7, batch_size: int=32, info_steps: int=2000):

    mlm = MaskedLanguageModeling(model_name, output_path=f'output/mlm_{mlm_epochs}')
    mlm.set_datasets(train_path, mlm_dev_path) \
        .train(epochs=mlm_epochs, batch_size=batch_size, info_steps=info_steps) \
            .save()

    if sim_cse_epochs > 0:
        sim_cse = SimCSE(mlm.output_dir if mlm_epochs > 0 else model_name, output_path=f'output/mlm_{mlm_epochs}_sim_cse_{sim_cse_epochs}')
        sim_cse.set_datasets(train_path, sim_cse_dev_path) \
            .train(epochs=sim_cse_epochs, batch_size=batch_size, info_steps=info_steps)

        return sim_cse.output_dir
    else:
        return mlm.output_dir

    # utils.evaluate_embeddings(sim_cse.output_dir, 'notebooks/stsbenchmark.tsv')

def perform_experiment(
    train_path: str, 
    mlm_dev_path: str, 
    sim_cse_dev_path: str, 
    test_path: str,
    model_name: str='distilbert-base-uncased',
    max_epochs: int=10,
    batch_size: int=32
):
    
    for epochs_mlm in range(max_epochs):
        epochs_simcse = max_epochs-epochs_mlm

        model_path = consecutive_training(train_path, mlm_dev_path, \
            sim_cse_dev_path, model_name, \
            mlm_epochs=epochs_mlm, sim_cse_epochs=epochs_simcse, batch_size=batch_size)

        evaluate_embeddings(model_path, test_path)


