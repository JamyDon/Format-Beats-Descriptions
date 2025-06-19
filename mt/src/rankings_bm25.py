# from utils import load_samples
import logging
import os
from retriv import SearchEngine
import random
from typing import List, Dict
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)

def load_samples(filepath) -> List[str]:
    return open(filepath, 'r').read().splitlines()

def get_collection_data_structure(samples):
    collection = []
    for id, sample in tqdm(enumerate(samples), desc='Generating collection data structure'):
        current = {"id": id, "text": sample}
        collection.append(current)
    return collection

def get_bm25_ranking(train_src_path, test_src_path):    
    queries = load_samples(test_src_path)
    logging.info('loading corpus...')
    training_samples = load_samples(train_src_path)
    logging.info('loaded corpus...')
    logging.info('number of samples is: {}'.format(len(training_samples)))

    collection = get_collection_data_structure(training_samples)
    se = SearchEngine("new-index").index(collection)
    logging.info('collection indexed...')

    # capture random ranking 
    random_ranking = []
    random.seed(42)
    for i in tqdm(range(10000), desc='Generating random ranking'):
        random_ranking.append({'index': random.randint(0, len(training_samples)), 'score': 0 })

    result = {}
    for ind, query in tqdm(enumerate(queries), desc='Generating bm25 ranking', total=len(queries)):
        # logging.info('indexing query: {}'.format(ind))
        ranking = se.search(query=query, return_docs=True, cutoff=100)
        new_ranking = []
        for item in ranking:
            new_ranking.append({'index': int(item['id']), 'score': round(float(item['score']), 2)})
        
        # sometimes due to small query, we get no rankings
        if len(new_ranking) == 0:
            result[ind] = random_ranking
        else:
            result[ind] = new_ranking
    
    return result

def write_to_file(data: Dict, path_to_index, path_to_score):
    logging.info("writing to file...")
    indexes = []
    scores = []
    for _, examples in data.items():
        indexes.append(" ".join([str(item['index']) for item in examples]) + "\n")
        scores.append(" ".join([str(item['score']) for item in examples]) + "\n")

    if not os.path.exists(os.path.dirname(path_to_index)):
        os.makedirs(os.path.dirname(path_to_index))

    with open(path_to_index, 'w') as f:
        f.writelines(indexes)
    with open(path_to_score, 'w') as f:
        f.writelines(scores)

    logging.info("done!")
