'''
modifications on the permissions, hardware.
'''
import os

from attacker.modification import xml_util as xu
from attacker.modification import droid_modification
from attacker.modification.droid_modification import logger

PERMTAG='uses-permission'
HW = 'uses-feature'

class DroidPermModification(droid_modification.DroidModification):
    """Modification for permission and hardware"""
    def __init__(self, disassembly_root, verbose):
        super(DroidPermModification, self).__init__(disassembly_root, verbose)
        self.comp_type = PERMTAG

    def insert(self, specific_name, mod_count = 1):
        '''Insert an permission based on 'specfic_name' into manifest.xml file'''
        if not isinstance(specific_name, str) and not os.path.isdir(self.disassembly_root):
            raise ValueError("Type error: str variable meets the requirement.")

        if mod_count < 0:
            raise ValueError("The amount of insertion cannot be smaller than 0.")

        manifest_tree = xu.get_xmltree_by_ET(os.path.join(self.disassembly_root, droid_modification.MANIFEST))

        info, flag, new_manifest_tree = xu.insert_perm_manifest(manifest_tree, self.comp_type, specific_name, mod_count)

        xu.dump_xml(os.path.join(self.disassembly_root, droid_modification.MANIFEST), new_manifest_tree)

        if self.verbose:
            if flag:
                logger.info(info)
                logger.info(
                    'Permission insertion: Successfully insert \'{}\' into \'{}/androidmanifest.xml\''.format(
                        specific_name,
                        os.path.basename(self.disassembly_root)))
            else:
                logger.warn(info)

    def remove(self, elem_name, mod_count = 1):
        raise NotImplementedError("Risk the functionality.")

class DoridHardwareModification(droid_modification.DroidModification):
    def __init__(self, disassembly_root, verbose):
        super(DoridHardwareModification, self).__init__(disassembly_root, verbose)
        self.comp_type = HW

    def insert(self, specific_name, mod_count = 1):
        '''Insert an hardware based on 'specfic_name' into manifest.xml file'''
        if not isinstance(specific_name, str) and not os.path.isdir(self.disassembly_root):
            raise TypeError("Type error:")

        if mod_count < 0:
            raise ValueError("The amount of insertion cannot be smaller than 0.")

        manifest_tree = xu.get_xmltree_by_ET(os.path.join(self.disassembly_root, droid_modification.MANIFEST))

        info, flag, new_manifest_tree = xu.insert_perm_manifest(manifest_tree, self.comp_type, specific_name, mod_count)

        xu.dump_xml(os.path.join(self.disassembly_root, droid_modification.MANIFEST), new_manifest_tree)

        if self.verbose:
            if flag:
                logger.info(info)
                logger.info(
                    'Hardware insertion: Successfully insert \'{}\' into \'{}/androidmanifest.xml\''.format(
                        specific_name,
                        os.path.basename(self.disassembly_root)))
            else:
                logger.warn(info)

    def remove(self, elem_name, mod_count = 1):
        raise NotImplementedError("Risk the functionality.")