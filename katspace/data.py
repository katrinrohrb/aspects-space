import json
import stanza 
import pathlib
from pathlib import Path
import os 
import pandas as pd
import numpy as np

def results_from_json(filenames, results_dir):

  if isinstance(filenames, str):
    filename = filenames
    json_file_path = Path(results_dir, Path(filename).stem + '-result.json')

    if json_file_path.exists():
        with open(json_file_path, 'r', encoding="utf-8") as f:
            json_results = json.load(f)
        return json_results
    else:
        return None
  elif isinstance(filenames, list):
    return {filename : results_from_json(filename, results_dir) for filename in filenames}
  

def chunker_alt(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def chunk_lengths(total_sents, num_chunks):
  base_chunk_size, remainder = divmod(total_sents, num_chunks)
  return [base_chunk_size + 1] * remainder + [base_chunk_size] * (num_chunks - remainder)

def chunker(seq, size = None, num_chunks = None):
  if (num_chunks and (len(seq) < num_chunks)) or (size and (len(seq) < size)):
     raise("sequence is too short to be chunked in this way")
  if (size != None) and (num_chunks == None):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]
  elif (size == None) and (num_chunks != None):
    lengths = chunk_lengths(len(seq), num_chunks)
    sums = [sum(lengths[0:i]) for i in range(0, num_chunks + 1)]
    chunk_borders = zip(sums[:-1], sums[1:])
    return [seq[start:end] for start, end in chunk_borders]
  else:
    raise("chunker should not be called with both size and num_chunks args")
  


class Results(): 
    
    @classmethod
    def convert_json_to_parquet(cls, json_fns, json_dir, labels_pqt_fn, scores_pqt_fn):
      from .core import drive_dir

      results = results_from_json(json_fns, json_dir)
      max_len = max([len(results[filename]) for filename in results.keys() if results[filename] != None])
       
      labels_padded = [pd.Series(data = \
      [result["label"] for result in results[filename]] + [""] * (max_len - len(results[filename])), \
                                name = filename) for filename in results.keys() if results[filename] != None]

      labels_df = pd.DataFrame(labels_padded)
      labels_df = labels_df.T

      fp_labels_df = Path(drive_dir, labels_pqt_fn)
      labels_df.to_parquet(fp_labels_df)


      scores_padded = [pd.Series(data = \
      [result["score"] for result in results[filename]] + [np.NaN] * (max_len - len(results[filename])), \
                                name = filename) for filename in results.keys() if results[filename] != None]

      scores_df = pd.DataFrame(scores_padded)

      scores_df = scores_df.T

      fp_scores_df = Path(drive_dir, scores_pqt_fn)
      scores_df.to_parquet(fp_scores_df)
    
    def __init__(self):
       pass
  

    