import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np
from tqdm import tqdm
import json
import csv
import logging

logger = logging.getLogger(__name__)

# Used for passage retrieval
def load_passages(path):
    if not os.path.exists(path):
        logger.info(f"{path} does not exist")
        return
    logger.info(f"Loading passages from: {path}")
    passages = []
    with open(path) as fin:
        if path.endswith(".jsonl"):
            for k, line in enumerate(fin):
                ex = json.loads(line)
                passages.append(ex)
        else:
            reader = csv.reader(fin, delimiter="\t")
            for k, row in enumerate(reader):
                if not row[0] == "id":
                    ex = {"id": row[0], "title": row[2], "text": row[1]}
                    passages.append(ex)
    return passages


class Indexer(object):

    def __init__(self, vector_sz, n_subquantizers=0, n_bits=8, use_gpu=False, gpu_id=0):
        if n_subquantizers > 0:
            # For cosine similarity, we use IndexPQ with METRIC_INNER_PRODUCT on L2-normalized vectors.
            self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            # For cosine similarity, we use IndexFlatIP on L2-normalized vectors.
            self.index = faiss.IndexFlatIP(vector_sz)
        self.index_id_to_db_id = []


        self.use_gpu = use_gpu
        if self.use_gpu:
            self.gpu_id = gpu_id
            self.gpu_resources = faiss.StandardGpuResources()
        if use_gpu:
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, gpu_id, self.index)
        

    def index_data(self, ids, embeddings):
        self._update_id_mapping(ids)
        embeddings = embeddings.astype('float32')
        # Normalize vectors to use cosine similarity with IndexFlatIP
        faiss.normalize_L2(embeddings)
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)
        print(f'Total data indexed {len(self.index_id_to_db_id)}')

    def search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size: int = 2048) -> List[Tuple[List[object], List[float]]]:
        query_vectors = query_vectors.astype('float32')
        # Normalize query vectors to use cosine similarity with IndexFlatIP
        faiss.normalize_L2(query_vectors)
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        
        for k in tqdm(range(nbatch)):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            
            # 如果没有ID映射，直接使用FAISS内部索引
            if not self.index_id_to_db_id:
                db_ids = [[str(i) if i != -1 else "-1" for i in query_top_idxs] for query_top_idxs in indexes]
            else:
                # 有ID映射时的安全转换
                db_ids = []
                for query_top_idxs in indexes:
                    query_db_ids = []
                    for i in query_top_idxs:
                        if i == -1:  # FAISS返回-1表示未找到
                            query_db_ids.append("-1")
                        elif 0 <= i < len(self.index_id_to_db_id):
                            query_db_ids.append(str(self.index_id_to_db_id[i]))
                        else:
                            logger.warning(f"Index {i} out of range, using raw index")
                            query_db_ids.append(str(i))
                    db_ids.append(query_db_ids)
            
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        
        return result

    def serialize(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        print(f'Serializing index to {index_file}, meta data to {meta_file}')

        cpu_index = faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index
        faiss.write_index(cpu_index, index_file)

        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path, index_filename="index.faiss"):
        # 使用指定的索引文件名或默认名称
        index_file = os.path.join(dir_path, index_filename)
        # 从索引文件名生成元数据文件名，保持一致的命名方式
        meta_filename = index_filename.replace('.faiss', '_meta.faiss')
        if meta_filename == index_filename:  # 如果没有.faiss后缀
            meta_filename = index_filename + '_meta'
        meta_file = os.path.join(dir_path, meta_filename)
        
        print(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file)
        print(f'Loaded index of type {type(self.index)} and size {self.index.ntotal}')
        if self.use_gpu:
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_id, self.index)
            print('index_cpu_to_gpu successful!')



    def _update_id_mapping(self, db_ids: List):
        self.index_id_to_db_id.extend(db_ids)