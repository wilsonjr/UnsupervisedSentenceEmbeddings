.. -*- mode: rst -*-

=====
Unsupervised training method for Sentence Embeddings
=====


-----------
Requirements
-----------

* `UMAP <https://github.com/lmcinnes/umap>`_
* `transformers <https://github.com/huggingface/transformers>`_
* `sentence-transformers <https://github.com/UKPLab/sentence-transformers>`_


-----------
Instalation
-----------

.. code:: bash

    python setup.py bdist_wheel 
    pip install dist/unsupervised_embeddings-0.0.1-py3-none-any.whl

-----------
Usage Examples
-----------

**Unsupervised training with MLM and SimCSE**

.. code:: python
    
    from unsupervised_embeddings import utils

    # these will generate series of models combinations
    # for max. of 5 epochs:
    # 0 mlm + 5 simcse
    # 1 mlm + 4 simcse
    # ...
    # see v0.0.1 release for these datasets

    utils.perform_experiment( 
        train_path='datasets/train_nli.csv', 
        mlm_dev_path='datasets/dev_nli.csv',
        sim_cse_dev_path='datasets/stsbenchmark.tsv',
        test_path='datasets/stsbenchmark.tsv',
        model_name='distilbert-base-uncased', 
        max_epochs=5,
        batch_size=124
    )


**Reproduce generated experiments**

.. code:: python

    # Pearson correlation
    utils.show_test_eval('./personal/UnsupervisedSentenceEmbeddings/experiment_output/experiment_5_statistics.csv')

    # Projections of test set with training models
    utils.show_projections('./personal/UnsupervisedSentenceEmbeddings/experiment_output/experiment_epochs_5_projections.csv')

-----------
Support 
-----------

Please, if you have any questions feel free to contact me at wilson_jr@outlook.com
