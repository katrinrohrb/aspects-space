from katspace.core import Session
from pathlib import Path

import logging

# create logger
logger = logging.getLogger('katspace')
#logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
#ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

#logger.debug(str(__file__), str(type(__file__)))

session = Session()


