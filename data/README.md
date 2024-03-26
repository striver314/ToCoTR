## Data Description

This folder contains the query and qrel files used in our paper.

- For ANTIQUE, we include a qrel file, a query file, and its corresponding typo query files. The qrel and query file can be found in [ir-datasets](https://ir-datasets.com/antique.html). For the manually validated typoed queries, the sourced data is in [Link](https://github.com/Guzpenha/query_variation_generators/blob/main/data/variations_antique_labeled.csv).
- For MSMARCO dev, and TREC 2019 dataset, we directly used the [CharacterBERT+ST](https://dl.acm.org/doi/pdf/10.1145/3477495.3531951) open-sourced data, which you can find in the [Link](https://github.com/ielab/CharacterBERT-DR/tree/main/data). Besides, in our work we also use [Google Search API](6https://developers.google.com/customsearch/v1/introduction) to obtain spelling correction queries. We make this data public in the marco_dev directory.
