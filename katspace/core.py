import stanza

from stanza import Document
import re

import pandas as pd

import os
import time
from pathlib import Path
import json
import datetime
from tqdm import tqdm
from collections import Counter

import katspace
from random import randint, seed
from datasets import Dataset
import logging

import seaborn as sns

logger = logging.getLogger('katspace.core')


logger.debug(str(__file__))

FILE_DIR = Path(__file__).parent
DEFAULT_CONFIG_FILE = Path(FILE_DIR.parent, "katspace_config.json")

DRIVE_DIR = FILE_DIR.parent
MODEL_DIR = Path(DRIVE_DIR, "model")
RESULTS_DIR = Path(DRIVE_DIR, "data/results")
DATA_DIR = Path(DRIVE_DIR, "data")

SPACE_TYPES = ("perceived_space", "action_space", "visual_space", "descriptive_space", "no_space")
COLORS = {
  "black" : '\033[0m', #black
  "red" : '\033[91m', #red
  "green" : '\033[92m', #green
  
  "cyan" : '\033[96m', #cyan
  "purple" : '\033[94m', #purple

  "reset" : "\u001b[0m",
  "reverse" : "\u001b[7m"
}

session = None

space_types = ["perceived_space", "action_space", "visual_space", "descriptive_space", "no_space"]
space_types_ext = space_types + ["all_space"]
space_types_pos = ["perceived_space", "action_space", "visual_space", "descriptive_space"]


space_types2colors = {"perceived_space" : "purple" , "action_space" : "red", "visual_space" : "green", "descriptive_space" : "cyan", "no_space" : "black"}

space_type_palette=["#dd8452", "#4c72b0", "#c44e52", "#55a868", "#9467bd", "green"]

space_type_order = {"action_space" : 1,
                    "perceived_space" : 2 , 
                    "descriptive_space" : 3, 
                    "visual_space" : 4, 
                    "no_space" : 5,
                    "all_space" : 6}


color_indexer = {'action_space': 1,
 'perceived_space': 0,
 'visual_space': 2,
 'descriptive_space': 3,
 'all_space': 4}

palette = {space_type : sns.color_palette().as_hex()[i]for space_type, i in color_indexer.items()}


seed(round(time.time()*1000))


def space_type_sort(space_type_series):
  return space_type_series.map(space_type_order)

def results_from_json(filename, results_dir):
  if results_dir == None: 
     results_dir = katspace.session.results_dirs[0]
  filename = Path(filename).stem + '-result.json'
  with Path(results_dir,filename).open('r', encoding="utf-8") as f:
    json_results = json.load(f)
    return json_results
    
class Text:
  pickle_dir_tokenized = "Datasets/pickle/tokenized_sents/"
  pickle_dir_dep_parsed = "Datasets/pickle/dep_parsed/"

  def __init__(self, doc = None, filename = None, corpus = None, txt_dirs = None, results_dirs = None):
    self.doc = doc
    self.results_dirs = results_dirs
    if self.results_dirs == None:
      self.results_dirs = katspace.session.results_dirs

    filename = os.path.basename(filename)
    if filename.endswith(".pickle"):
      filename = filename[:-7]
    self.filename = filename

    #self.txt_dirs = txt_dir
    if txt_dirs == None:
      self.txt_dirs = katspace.session.txt_dirs
    else: 
      self.txt_dirs = txt_dirs
      

    if self.doc == None: 

      path = Path(filename).absolute()
      if path.exists(): 
        self.dir = str(path.parent)
        logger.debug(f"File found: {str(path)}")
      else: 
        logger.debug("Searching txt_dirs ...  ")
        self.dir = Text.find_file(filename, txt_dirs, throw_error=False)
        logger.debug(f"{filename} found in {self.dir}")

      logger.debug(f"self.dir = {self.dir}")
      logger.debug(f"session.results_dir_dict = {katspace.session.results_dir_dict}")

      if self.dir in katspace.session.results_dir_dict:
        self.results_dir = katspace.session.results_dir_dict[self.dir]
        logger.debug(f"Inferred {self.results_dir} from directory {self.dir}")
      else:
        self.results_dir = Text.find_file(filename + "-results.json", results_dirs, throw_error=False)
        logger.debug(f"Results for {filename} found in {self.results_dir}")
     
      path = Path(self.dir, filename)
     
      if path.exists():
        logger.debug(f"Path exists: {str(path)}")
        with path.open(mode="r", encoding="utf8") as f: 
          self.sentences = [KatSentence(line) for line in f]
      else:
        logger.warning(f"Cannot find: {str(path)}")

    elif self.doc != None:
      self.sentences = self.doc.sentences

    #This needs to fail gracefully if no results yet
    self.results = results_from_json(self.filename, self.results_dir)
    
    self.num_sents = len(self.sentences)
    self.is_colored = [False] * self.num_sents
    
  
  @classmethod
  def load_tokenized(cls, filename, pickle_dir = None):
    global drive_dir
    if pickle_dir == None:
      pickle_dir = Text.pickle_dir_tokenized
    if not filename.endswith(".pickle"):
      filename = filename + ".pickle"
    if not filename.startswith("/"):
      #print(f"Assuming source directory: {pickle_dir}")
      print(f"Trying to read: {drive_dir}, {pickle_dir}, {filename}")
      filename = Path(drive_dir, pickle_dir, filename)
    with open(filename, "rb") as f:
      doc = Document.from_serialized(f.read())
    return Text(doc, filename, txt_dirs = pickle_dir)

  @classmethod
  def find_file(cls, filename, dirs = None, throw_error = True):
    if type(dirs) == list: 
      failed_list = []
      for dir in dirs: 
        path = Path(dir, filename)
        if path.exists(): 
          return dir
        else: 
          failed_list.append(path)
      
      if throw_error:
        raise OSError(f"File {filename} not found in any of: {failed_list}")
      else: 
        return None 
    if type(dirs) != list: 
      return dirs
  

  @classmethod
  def load_txt_file(cls, filename, txt_dirs = None, results_dirs = None):
    logger.debug(f"load_txt_file(filename = {filename}, txt_dirs = {txt_dirs}, results_dir = {results_dirs})")
    return Text(filename = filename, txt_dirs = txt_dirs, results_dirs = results_dirs)
  
  def sents_by_index(self, start_i, end_i = None, output_string = False, color = False):
    output_list = True
    if end_i == None:
      output_list = False
      end_i = start_i +1

    sentences = self.sentences[start_i : end_i]
    
    if output_string:
      if color:
        return " ".join([ sent for sent in sentences])
      else: 
        return " ".join([ sent.text for sent in sentences])
    else:
      if output_list == False:
        return sentences[0]
      else:
        return sentences
  
  def formatted_by_index(self, start_i, end_i = None, output_string = False, color = False):
    output_list = True
    if end_i == None:
      output_list = False
      end_i = start_i +1
    
    sentences = [sent.text for sent in self.sentences[start_i : end_i]] 
   
    if (self.results != None) and color:
      for i in range(start_i, end_i):
        space_type = self.results[i]["label"]
        if space_type == "no_space": #or self.is_colored[i]:
          continue
        color_sent = space_types2colors[space_type]
        sentences[i - start_i] =   COLORS[color_sent] + f"<{space_type.rstrip('_space')}> " + \
                                            sentences[i - start_i] \
                                        + f" <\{space_type.rstrip('_space')}> " + COLORS["black"]
    if output_string:
      if color:
        return " ".join([ sent for sent in sentences])
      else: 
        return " ".join([ sent.text for sent in sentences])
    else:
      if output_list == False:
        return sentences[0]
      else:
        return sentences

  def get_context(self, sent_i, trailing_window = 1, word_window_size = 512, output_type = "bert"):
    if output_type == "browse":
      color = True
      highlight = COLORS["reverse"]
      unhighlight = COLORS["reset"]
    elif output_type == "annotate":
      color = False
      highlight = COLORS["cyan"]
      unhighlight = COLORS["black"]

    start_i = sent_i
    end_i = sent_i + trailing_window + 1
    num_words = sum([len(sent.words) for sent in self.sents_by_index(sent_i, end_i)])
    while num_words < word_window_size and start_i > 0:
      start_i = start_i - 1
      num_words = num_words + len(self.sents_by_index(start_i).words)
    while num_words < word_window_size and end_i < self.num_sents:
      end_i = end_i + 1
      num_words = num_words + len(self.sents_by_index(end_i).words)
    
    if output_type == "annotate" or output_type == "browse":
      preceding_window = self.formatted_by_index(start_i, sent_i, color = color)
      sentence = self.formatted_by_index(sent_i, color = color)
      trailing_window = self.formatted_by_index(sent_i + 1, end_i + 1, color = color)
      string = " ".join([sent for sent in preceding_window]) + " " +\
      highlight + \
      sentence + " " +\
      unhighlight + \
      " ".join([sent for sent in trailing_window])
      return string
    elif output_type == "stanza_obj":
      preceding_window = self.sents_by_index(start_i, sent_i)
      sentence = self.sents_by_index(sent_i)
      trailing_window = self.sents_by_index(sent_i + 1, end_i + 1)
      return num_words, preceding_window, sentence, trailing_window
    elif output_type == "bert":
      preceding_window = self.sents_by_index(start_i, sent_i)
      sentence = self.sents_by_index(sent_i)
      trailing_window = self.sents_by_index(sent_i + 1, end_i + 1)
      return [sent.text for sent in preceding_window], sentence.text, [sent.text for sent in trailing_window]

  def get_random_sentences(self, size = 1, return_ids = False):
    ids = [randint(0, self.num_sents -1) for _ in range(size)]
    if return_ids:
      return ids 
    else:
      return [self.sents_by_index(id, output_string=True) for id in ids]
      
  
  # for space BERT
  def get_random_dataset(self, size):
    dict = {"text": self.get_random_sentences(size)}
    return Dataset.from_pandas(pd.DataFrame(dict))
  
  def get_dataset(self, id_start, id_end):
    dict = {"text": [sent.text for sent in self.sents_by_index(id_start, id_end)]}
    return Dataset.from_pandas(pd.DataFrame(dict))

class Corpus:
  def __init__(self, drive_dir = None, corpus_dir_rel = ""):
    if drive_dir == None: 
       self.drive_dir = katspace.session.drive_dir
    else:
      self.drive_dir = drive_dir 

    #careful! directory names as constants likely to cause trouble 
    self.pickle_dir_tokenized = self.drive_dir + "Datasets/pickle/tokenized_sents/"
    self.pickle_dir_dep_parsed = self.drive_dir + "Datasets/pickle/dep_parsed/"
    self.corpus_dir = self.drive_dir + corpus_dir_rel
   
    self.filenames_pickled_tk = os.listdir(self.pickle_dir_tokenized)
    self._df = None 

  @classmethod
  def parse_filename(cls, name, return_type = "tuple"):
    re_match=re.match('([^-]*)-(.*)[(](.*)[)][.]txt',name)
    if not re_match:
        print(f"file not matching regexp: {name}\n")
    else:
      try:
          name_tup = re_match.group(1,2,3)
          cleaned_tuple = tuple(map(cls.clean_entry,name_tup))
          cleaned_tuple = cls.parse_name(cleaned_tuple[0]) + cleaned_tuple[1:] + (name,)
      except:
          print(re_match.group(0))
          raise
      if return_type == "tuple":
        return cleaned_tuple
      else:
        return {"Author_last" : cleaned_tuple[0], 
        "Author_first": cleaned_tuple[1],	
        "Author": cleaned_tuple[2],	
        "Title": cleaned_tuple[3],	
        "Year": cleaned_tuple[4],	
        "Filename": cleaned_tuple[5]}

  @classmethod
  def clean_entry(cls, entry):
    re_match= re.match('_*(.*[^_])_*$', entry)
    try:
        entry= re_match.group(1)
    except:
        print(entry)
        raise
    entry= entry.replace("_", " ")
    
    return entry

  @classmethod
  def parse_name(cls, name):
    name_parts = name.split()
    last_name = name_parts[-1]
    first_names = ' '.join(name_parts[:-1])
    sort_name = last_name + ", " + first_names
    return (last_name , first_names, sort_name)
  
  #should be named make_df
  def _load_df(self):
    filenames = os.listdir(self.corpus_dir)
    parsed = [ self.parse_filename(name) for name in filenames if self.parse_filename(name) is not None ]
    self._df = pd.DataFrame(parsed, columns=['Author_last', 'Author_first', 'Author', 'Title','Year', 'Filename'])


  def get_random_book(self):
    i = randint(0, len(self.filenames_pickled_tk) -1)
    filename = self.filenames_pickled_tk[i]
    return Text.load_tokenized(self.pickle_dir_tokenized + filename)
  
def make_date_dir(name, path = None):
  if path == None:
    path = drive_dir
  date_str = datetime.today().strftime('%Y%m%d')
  results_dir_rel = f"{name}{date_str}"

  results_dir = Path(path + results_dir_rel)
  if not results_dir.exists():
    return results_dir.mkdir()
  
def sum_results(labels):
  counter = Counter(labels)
  return counter

def results_into_df(df  : pd.DataFrame, results_dir = None) -> pd.DataFrame :
  corpus_sel = df

  for space_type in space_types:
    corpus_sel.loc[:,space_type] = [0]*corpus_sel.shape[0]

  for index, row in tqdm(corpus_sel.iterrows()):
    filename = row["filename"]
    results = results_from_json(filename, results_dir = results_dir) 
    if results is not None:
      labels = [result["label"] for result in results]
      result = sum_results(labels)
      for space_type in space_types:
        corpus_sel.loc[index, space_type] = result[space_type]
  return corpus_sel

class KatSentence():
  def __init__(self, txt_str : str):
    self.text = txt_str
    self.words = self.text.split()

class Session():
  def __init__(self, config_file = None):
    logger.debug("Session initialized")
    
    self.txt_dirs = None
    self.results_dirs = None
    self.DEFAULT_CONFIG = {"default" : {
                                        "drive_dir" : ".",
                                        "txt_dirs"  : ".", 
                                        "results_dirs" : "."
                                        },
                          "colab" : {"drive_dir" : "/content/drive/MyDrive/"}}
    
    self.config = self.DEFAULT_CONFIG.copy()

    if config_file != None: 
      self.config_file = config_file
    else:
      self.config_file = DEFAULT_CONFIG_FILE 

    path = Path(self.config_file)
    if path.exists():
      with path.open() as config_file: 
        self.config.update(json.load(config_file))
        self.config_file = config_file
    else: 
      print(f"Configuration file {self.config_file} not found in {Path.cwd()}")
    
    for key in self.config["default"]:
      self.__dict__[key] = self.config["default"][key]
      logger.debug(f"Session.__init__: set {key} to {self.__dict__[key]}")
    
  
  def set_type(self, session_type : str = None):
    if session_type in self.config.keys(): 
      for key in self.config[session_type]:
        self.__dict__[key] = self.config[session_type][key]
        self.session_type = session_type
    else:
      raise RuntimeError(f"Session type {session_type} not configured in {self.config_file}!")
    

      
       
      
  


