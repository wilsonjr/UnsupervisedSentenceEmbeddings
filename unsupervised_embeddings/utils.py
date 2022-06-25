import csv 
import gzip

import umap
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

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
    _ = test_evaluator(model, output_path=model_path)

    embeddings = model.encode(embedding_sentences)
    embeddings2d = umap.UMAP().fit_transform(embeddings)

    df = pd.DataFrame({
        'Name': model_path, 
        'UMAP 1': embeddings2d[:, 0],
        'UMAP 2': embeddings2d[:, 1]
    })
    df.to_csv(model_path+'/projection.csv', index=False)



def consecutive_training(train_path: str, mlm_dev_path: str, sim_cse_dev_path: str, model_name: str='distilbert-base-uncased', mlm_epochs: int=3, sim_cse_epochs: int=7, batch_size: int=32, info_steps: int=2000):

    if mlm_epochs > 0:
        mlm = MaskedLanguageModeling(model_name, output_path=f'output/mlm_{mlm_epochs}')
        mlm.set_datasets(train_path, mlm_dev_path) \
            .train(epochs=mlm_epochs, batch_size=batch_size, info_steps=info_steps) \
                .save()

    if sim_cse_epochs > 0:
        input_simcse_model = mlm.output_dir if mlm_epochs > 0 else model_name
        print(">> {}".format(input_simcse_model))

        sim_cse = SimCSE(input_simcse_model, output_path=f'output/mlm_{mlm_epochs}_sim_cse_{sim_cse_epochs}')
        sim_cse.set_datasets(train_path, sim_cse_dev_path) \
            .train(epochs=sim_cse_epochs, batch_size=batch_size, info_steps=info_steps)

        return sim_cse.output_dir
    else:
        return mlm.output_dir


def perform_experiment(
    train_path: str, 
    mlm_dev_path: str, 
    sim_cse_dev_path: str, 
    test_path: str,
    model_name: str='distilbert-base-uncased',
    max_epochs: int=10,
    batch_size: int=32
):

    paths = []
    
    # for epochs_mlm in range(1, max_epochs+1):
    for epochs_mlm in range(0, max_epochs+1):
        epochs_simcse = (max_epochs)-epochs_mlm
        print("----Training with {} epochs for MLM and {} epochs for SimCSE".format(epochs_mlm, epochs_simcse))

        model_path = consecutive_training(train_path, mlm_dev_path, \
            sim_cse_dev_path, model_name, \
            mlm_epochs=epochs_mlm, sim_cse_epochs=epochs_simcse, batch_size=batch_size)

        paths.append(model_path)
        evaluate_embeddings(model_path, test_path)
    
    df_projections = []

    df_analysis = []

    for path in paths:
      
        df = pd.read_csv(path+'/projection.csv')
        df_projections.append(df)

        df = pd.read_csv(path+'/similarity_evaluation_eval-output_results.csv')
        df['Config'] = path
        df_analysis.append(df)

    df_projections = pd.concat(df_projections)
    df_analysis = pd.concat(df_analysis)
    df_projections.to_csv(f'output/experiment_epochs_{max_epochs}_projections.csv', index=False)
    df_analysis.to_csv(f'output/experiment_{max_epochs}_statistics.csv', index=False)



def show_test_eval(path: str) -> None:

    statistics = pd.read_csv(path)
    statistics['Config'] = statistics['Config'].apply(lambda x: x.split('/')[1])

    statistics['MLM Epochs'] = statistics['Config'].apply(lambda x: x.split('_')[1]).astype(int)
    statistics['SimCSE Epochs'] = statistics['Config'].apply(lambda x: x.split('_')[-1]).astype(int)

    ax = sns.barplot(x="MLM Epochs", y="cosine_pearson", data=statistics)
    ax.set(yscale="log")

    plt.show()


def show_projections(path: str) -> None:

    projections = pd.read_csv(path)
    projections['Name'] = projections['Name'].apply(lambda x: x.split('/')[1])
    projections['MLM Epochs'] = projections['Name'].apply(lambda x: x.split('_')[1]).astype(int)
    projections['SimCSE Epochs'] = projections['Name'].apply(lambda x: x.split('_')[-1]).astype(int)

    plt.figure(figsize=(18, 16))

    sns.relplot(data=projections, x='UMAP 1', y='UMAP 2', col='SimCSE Epochs', kind='scatter')
    plt.show()