'''
modifications on the string in smali files
'''
import os

from attacker.modification import exception as e

from attacker.modification import dex_util as du
from attacker.modification import droid_modification
from attacker.modification.droid_modification import logger

class DroidStringModification(droid_modification.DroidModification):
    """Modification for string encryption"""
    def __init__(self, disassembly_root, verbose):
        super(DroidStringModification, self).__init__(disassembly_root, verbose)
        self.all_smali_paths = du.get_smali_paths(os.path.join(disassembly_root, 'smali'))
        if len(self.all_smali_paths) > 0:
            self.partial_smali_paths = du.filter_smali_paths(self.all_smali_paths, ['java'], 'R$')
        else:
            self.partial_smali_paths = []

    def insert(self, elem_name, mod_count = 1):
        '''Insert the elem_name into a random smali file'''
        smali_path_selected = []
        if len(self.partial_smali_paths) > 0:
            smali_path_selected = du.select_smali_file(self.partial_smali_paths, num = 1)[0]

        if len(smali_path_selected) == 0:
            raise e.ModifyFileException("Cannot find an avaiable file to insert string {}".format(elem_name))

        if mod_count < 0:
            raise ValueError("The amount of insertion cannot be smaller than 0.")

        try:
            du.insert_const_string(smali_path_selected, elem_name, mod_count)
        except Exception as ex:
            logger.exception(ex)
            logger.error("Failed to insert string '{}' into file '{}'".format(elem_name,
                                                                              smali_path_selected))
            raise e.ModifyFileException(str(ex))
        logger.info("String insert: Successfully insert the string '{}' in to file '{}'".format(elem_name,
                                                                                                smali_path_selected))

        if self.verbose:
            print("String insert: Successfully insert the string '{}' in to file '{}'".format(elem_name,
                                                                                                smali_path_selected))

    def remove(self, elem_name, mod_count = 1):
        '''
        Encrypt the string in smali files
        mod_count = -1 indicates that all the corresponding elements will be changed
        '''
        try:
            is_modified, files_modified = du.encrypt_string(self.all_smali_paths, elem_name, mod_count)
            if is_modified:
                logger.info("Successfully modify APK: string encryption for '{}' of files '{}'".format(elem_name,
                                                                                                       '\n'.join(files_modified)))
                if self.verbose:
                    print("Successfully modify APK: string encryption for '{}'".format(elem_name))
            else:
                logger.info("No string encryption for '{}' of files '{}'".format(elem_name,
                                                                                '\n'.join(files_modified)))
                if self.verbose:
                    print("No string encryption for '{}'".format(elem_name))

        except e.ModifyFileException as ex:
            logger.exception(str(ex))
            logger.error("Failed to encrypt '{}' for file '{}'".format(elem_name,
                                                                       self.disassembly_root))
            raise ("Failed to Modify APK: string encryption, error " + str(ex))


def _main():
    disassembly_root = "/path/to/disassmbly_apk_dir/"
    string_enc = DroidStringModification(disassembly_root, verbose=True)
    elem_name = 'http://10.0.0.172'
    string_enc.insert(elem_name)

if __name__ == '__main__':
    _main()