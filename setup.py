from setuptools import setup, find_packages

setup(
    name='Graph Retrieval System',
    version='0.1',
    description='Graph retrieval',
    long_description='Graph retrieval',
    author='JVNK',
    author_email='jaya11vibhav@gmail.com',
    url='https://github.com/jayavibhavnk/Graph-retrieval-system',
    packages=find_packages(),
    install_requires=[
      'langchain_openai',
      'langchain',
      'sentence_transformers'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha'
    ],
)
