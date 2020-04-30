'''
basic modifications on android.
OPERATOR = {
    #insert
    0 : "insertion",
    #delete
    1 : "removal"
}
'''
import os
import sys
from abc import ABCMeta, abstractmethod

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_root)
from config import config, logging
logger = logging.getLogger("attacker.modification")

MANIFEST = "AndroidManifest.xml"  # type: str

class DroidModification(object):
    """Abstract base class for all attack classes."""
    __metaclass__ = ABCMeta

    def __init__(self, disassembly_root, verbose):
        self.disassembly_root = disassembly_root
        self.verbose = verbose

    @abstractmethod
    def insert(self, elem_name, mod_count = 1):
        """Insert an specified element"""
        raise NotImplementedError

    @abstractmethod
    def remove(self, elem_name, mod_count = 1):
        """
        delete an specified element
        mod_count = -1 indicates that all the corresponding elements will be changed
        """
        raise NotImplementedError