'''
modifications on the android components (activity, service, provider, receiver):
We here modify the component name, the others (e.g., label, icon) are unavailable.
'''
import os

from attacker.modification import xml_util as xu
from attacker.modification import dex_util as du
from attacker.modification import droid_modification
from attacker.modification.droid_modification import logger
from attacker.modification import exception as e


class DroidCompModification(droid_modification.DroidModification):
    """Modification for components"""
    def __init__(self, disassembly_root, component_type, verbose):
        super(DroidCompModification, self).__init__(disassembly_root, verbose)
        self.comp_type = component_type

    def insert(self, specific_name, mod_count = 1):
        """
        Insert an component based on 'specfic_name' into manifest.xml file
        """
        if not isinstance(specific_name, str) and not os.path.isdir(self.disassembly_root):
            raise ValueError("Value error: require str type of variables.")

        manifest_tree = xu.get_xmltree_by_ET(os.path.join(self.disassembly_root, droid_modification.MANIFEST))

        info, flag, new_manifest_tree = xu.insert_comp_manifest(manifest_tree, self.comp_type, specific_name, mod_count)

        xu.dump_xml(os.path.join(self.disassembly_root, droid_modification.MANIFEST), new_manifest_tree)

        if self.verbose:
            if flag:
                logger.info(info)
                logger.info(
                    "'{}' insertion: Successfully insert activity '{}' into '{}'/androidmanifest.xml".format(
                        self.comp_type,
                        specific_name,
                        os.path.basename(self.disassembly_root)))
            else:
                logger.warn(info)
                raise e.ModifyFileException(info)

    def _rename_files(self, smali_paths, activity_name, new_activity_name):
        for smali_path in smali_paths:
            orginal_name = activity_name.split('.')[-1]
            modified_name = new_activity_name.split('.')[-1]
            if orginal_name in smali_path and \
                    modified_name not in smali_path:
                du.rename_smali_file(smali_path, activity_name, new_activity_name)

    def _rename_folders(self, smali_dirs, activity_name, new_activity_name):
        for smali_dir in smali_dirs:
            if du.name2path(activity_name) in smali_dir or \
                    os.path.dirname(du.name2path(activity_name)) in smali_dir:
                du.rename_smali_dir(smali_dir, activity_name, new_activity_name)

    def remove(self, specific_name, mod_count = -1):
        """
        change the denoted component name to a random string
        mod_count = -1 indicates that all the corresponding elements will be changed
        """

        # step 1: modify the corresponding name in AndroidManifest.xml

        if not isinstance(specific_name, str) and not os.path.exists(self.disassembly_root):
            raise ValueError("Value error:")

        # mod_count = -1 change name of all the specified components

        manifest_tree = xu.get_xmltree_by_ET(os.path.join(self.disassembly_root, droid_modification.MANIFEST))

        info, flag, new_comp_name, new_manifest_tree = xu.rename_comp_manifest(manifest_tree,
                                                                               self.comp_type,
                                                                               specific_name)
        xu.dump_xml(os.path.join(self.disassembly_root, droid_modification.MANIFEST), new_manifest_tree)

        if self.verbose:
            if flag:
                logger.info(info)
                logger.info(
                    "'{}' name changing: Successfully change name '{}' to '{}' of '{}'/androidmanifest.xml".format(
                        self.comp_type,
                        specific_name,
                        new_comp_name,
                        os.path.basename(self.disassembly_root)
                    ))
            else:
                logger.warn(info  + ": {}/androidmanifest.xml".format(os.path.basename(self.disassembly_root)))
                return

        # step 2: modify .smali files accordingly
        package_name = manifest_tree.getroot().get('package')
        smali_paths = du.get_smali_paths(self.disassembly_root)
        related_smali_paths = set(du.find_smali_w_name(smali_paths, specific_name))
        du.change_source_name(related_smali_paths, specific_name, new_comp_name)
        changed_class_names = set(du.change_class_name(related_smali_paths,
                                                       specific_name,
                                                       new_comp_name,
                                                       package_name))

        # Change class instantiation
        if len(changed_class_names) > 0:
            du.change_instantition_name(smali_paths,
                                        changed_class_names,
                                        specific_name,
                                        new_comp_name,
                                        package_name)

        if self.verbose:
            logger.info("'{}' name changing: Successfully change '{}' name in smali files".format(
                self.comp_type,
                specific_name))

        # step 3: modify all .xml files accordingly
        if len(changed_class_names) > 0:
            xml_paths = xu.get_xml_paths(self.disassembly_root)
            # todo: change asset xml
            # xu.change_xml(xml_paths, changed_class_names,
            #              specific_name, new_comp_name, package_name)

        if self.verbose:
            logger.info("'{}' name changing: Successfully change '{}' name in xml files".format(
                self.comp_type,
                specific_name))

        # step 4: modify folder and file names
        self._rename_files(smali_paths, specific_name, new_comp_name)

        # smali_dirs = du.get_smali_dirs(self.disassembly_root)
        # self._rename_folders(smali_dirs, specific_name, new_comp_name)

class DroidActivityModification(DroidCompModification):
    """Modification for activity"""
    def __init__(self, disassembly_root, verbose, component_type = 'activity'):
        super(DroidActivityModification, self).__init__(disassembly_root, component_type, verbose)

class DroidServiceModification(DroidCompModification):
    """Modification for Service"""
    def __init__(self, disassembly_root, verbose, component_type = 'service'):
        super(DroidServiceModification, self).__init__(disassembly_root, component_type, verbose)

class DroidReceiverModification(DroidCompModification):
    """Modification for receiver"""
    def __init__(self, disassembly_root, verbose, component_type = 'receiver'):
        super(DroidReceiverModification, self).__init__(disassembly_root, component_type, verbose)

class DroidProviderModification(DroidCompModification):
    """Modification for providier"""
    def __init__(self, disassembly_root, verbose, component_type = 'provider'):
        super(DroidProviderModification, self).__init__(disassembly_root, component_type, verbose)

    def insert(self, name = "_"):
        raise NotImplementedError("Not implemented.")

class DroidIntentModification(droid_modification.DroidModification):
    def __init__(self, disassembly_root, verbose):
        super(DroidIntentModification, self).__init__(disassembly_root, verbose)

    def insert(self, specific_name, mod_count = 1):
        """
        Insert an intent-filter based on 'specfic_name' into manifest.xml file
        """
        if not isinstance(specific_name, str) and not os.path.isdir(self.disassembly_root):
            raise ValueError("Value error: require str type of variables.")

        manifest_tree = xu.get_xmltree_by_ET(os.path.join(self.disassembly_root, droid_modification.MANIFEST))


        info, flag, new_manifest_tree = xu.insert_intent_manifest(manifest_tree, 'activity', specific_name, mod_count)

        xu.dump_xml(os.path.join(self.disassembly_root, droid_modification.MANIFEST), new_manifest_tree)

        if self.verbose:
            if flag:
                logger.info(info)
                logger.info(
                    "intent-filter insertion: Successfully insert intent-filter '{}' into '{}'/androidmanifest.xml".format(
                        specific_name,
                        os.path.basename(self.disassembly_root)))
            else:
                logger.error(info)
                raise e.ModifyFileException("Error of intent-filter '{}' insertion for {}:{}".format(specific_name,
                                                                                               self.disassembly_root,
                                                                                                     info))

    def remove(self, elem_name, mod_count = 1):
        raise NotImplementedError


def _test_main():
    act_mod = DroidActivityModification("/path/to/disassmbly_apk_dir/", True)
    act_name4 = ''
    act_mod.remove(act_name4)
    return


if __name__ == "__main__":
    _test_main()