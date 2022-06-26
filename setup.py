from setuptools import find_packages, setup 


setup(
	name='unsupervised_embeddings',
	packages=find_packages(include=['unsupervised_embeddings']),
	version='0.0.1',
	description='Unsupervised approaches for sentence embeddings',
    long_description_content_type='text/markdown',
	author='Wilson Estecio Marcilio Junior',
	author_email='wilson_jr@outlook.com',
	url='https://github.com/wilsonjr/UnsupervisedSentenceEmbeddings',
	license='MIT',
	install_requires=['transformers', 'sentence-transformers', 'umap'],
)