from rankings_bm25 import get_bm25_ranking, write_to_file
import logging
import os
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

log_dir = "../log/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir) 


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Time usage: {elapsed_time:.4f} seconds\n")
        return result
    return wrapper


@timing_decorator
def write_ranking(lang, is_outof=False):
    test_split = 'test'

    train_path = f"../data/{lang}/train.{lang}" if not is_outof else f"../data/{lang}/train.en"
    test_path = f"../data/{lang}/{test_split}.{lang}" if not is_outof else f"../data/{lang}/{test_split}.en"
    path_to_index = f"../data/{lang}/index/{test_split}/into/bm25.index" if not is_outof else f"../data/{lang}/index/{test_split}/outof/bm25.index"
    path_to_score = f"../data/{lang}/index/{test_split}/into/bm25.score" if not is_outof else f"../data/{lang}/index/{test_split}/outof/bm25.score"

    logger.info(f"Processing paths:\n{train_path}\n{test_path}\n{path_to_index}\n{path_to_score}")

    result = get_bm25_ranking(train_src_path=train_path, test_src_path=test_path)
    write_to_file(result, path_to_index, path_to_score)
    return result


def run(lang):
    log_file = f"{log_dir}{lang}.log"
    fileHandler = logging.FileHandler(log_file, "w")
    fileHandler.setLevel(logging.INFO)
    logger.addHandler(fileHandler)

    logger.info(f"Starting processing for language: {lang}")
    write_ranking(lang, is_outof=False)
    
    logger.info(f"Starting out-of processing for language: {lang}")
    write_ranking(lang, is_outof=True)

    logger.info(f"Finished processing for language: {lang}")
    logger.removeHandler(fileHandler)


if __name__ == "__main__":
    for lang in ['de', 'fr', 'ru']:
        run(lang)
    logger.info('Done!')
