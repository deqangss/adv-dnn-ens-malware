'''
modifications on the system api: We here modify the system api (i.e.,method) name
'''
import os

from attacker.modification import exception as e

from attacker.modification import dex_util as du
from attacker.modification import droid_modification
from attacker.modification.droid_modification import logger

curr_dir = os.path.dirname(os.path.realpath(__file__))
APISMALIMAP_PATH = os.path.join(curr_dir, 'res/Api22SmaliMapping.json')
#

SIMPLE_API_TEMPLATE = '''.method public invoke_{methodName}{randomString}({LclassName}{Lparams})V
    .locals {paramNum}
{varContent}
    .prologue
    :try_start_0
    invoke-virtual {{{Vars}}}, {LclassName}->{methodName}({Lparams})V
    :try_end_0
    .catch Ljava/lang/Exception; {{:try_start_0 .. :try_end_0}} :catch_0

    :goto_0
    return-void

    :catch_0
    move-exception v0

    .local v0, "ex":Ljava/lang/Exception;
    const-string v1, "Insert method error:"

    invoke-virtual {{v0}}, Ljava/lang/Exception;->toString()Ljava/lang/String;

    move-result-object v2

    invoke-static {{v1, v2}}, Landroid/util/Log;->v(Ljava/lang/String;Ljava/lang/String;)I

    goto :goto_0
.end method

'''

PARAM_Content_TEMP = '    .param p{order}, \"{varName}\"    # {paramName}'


class DroidSysAPIModification(droid_modification.DroidModification):
    """Modification for android api"""

    def __init__(self, disassembly_root, verbose):
        super(DroidSysAPIModification, self).__init__(disassembly_root, verbose)
        self.all_smali_paths = du.get_smali_paths(os.path.join(disassembly_root, 'smali'))
        if len(self.all_smali_paths) > 0:
            self.partial_smali_paths = du.filter_smali_paths(self.all_smali_paths, ['java'], 'R$')
        else:
            self.partial_smali_paths = []
        self.api_smali_mapping_dict = du.load_json(APISMALIMAP_PATH)

    def _generate_method_var_types(self, smali_param_types):
        """
        generate the variable type of API method
        :rtype: basestring
        :param smali_param_types: a list of smali parameter types, e.g., Landroid/...; IBZ etc.
        :return:
        """
        assert isinstance(smali_param_types, list)
        if len(smali_param_types) == 0:
            return ''

        return ''.join(smali_param_types)

    def generate_generic_smali(self, class_name, method_name, params, is_smali=True):
        if params == None:
            params = ''  # void
        import random
        number = random.choice(range(23456))
        random_str = du.random_string(code=class_name + method_name + params + str(number))
        param_types = du.get_param_smali_type(params, is_smali=is_smali)
        var_types = self._generate_method_var_types(param_types)  #
        param_num = 3  # class_name, param number, try..catch...

        if '.' in class_name:
            class_name = class_name.replace('.', '/')
        if not class_name.startswith('L'):
            class_name = 'L' + class_name
        if not class_name.endswith(';'):
            class_name = class_name + ';'

        variables_content = PARAM_Content_TEMP.format(order=1,
                                                      varName=du.random_name(),
                                                      paramName=class_name + '\n')

        # the following is the debug info
        param_variables = ['p1']

        if len(param_types) > 0:
            for i in range(0, len(param_types)):
                param_idx = i + 2
                variable_name = du.random_name(2345 + param_idx)
                variables_content = variables_content + PARAM_Content_TEMP.format(
                    order=param_idx,
                    varName=du.random_string(variable_name),
                    paramName=param_types[i] + '\n'
                )
                param_variables.append('p{:d}'.format(param_idx))
        param_variable = ', '.join(param_variables)

        generic_smali_block = SIMPLE_API_TEMPLATE.format(methodName=method_name,
                                                         randomString=random_str,
                                                         LclassName=class_name,
                                                         Lparams=var_types,
                                                         paramNum=param_num,
                                                         varContent=variables_content,
                                                         Vars=param_variable
                                                         )
        return generic_smali_block

    def insert(self, class_name=None, method_name=None, params=None, is_params_smali=False, mod_count=1, force=True,
               elem_name=None):
        """
        insert samli code based on class name and method
        :param class_name: the class name,
        :param method_name: method belongs to the class
        :param is_smali: Does the params has the smali type, e.g., Landroid/... or I, B, Z etc.
        :param mod_count: the maximum modification number
        :param force: enforce the insertion even that there is no smali code provided
        :param elem_name: negect
        """
        if elem_name:
            raise ValueError(" elem_name is neglected, please fill the arguments 'class_name' and 'method_name'.")

        mod_count = int(mod_count)
        if mod_count < 0:
            raise ValueError(" The amount of insertion cannot be smaller than 0.")

        # get java fashion
        if '/' in class_name:
            class_name = class_name.replace('/', '.')
        if class_name.startswith('L'):
            class_name = class_name.lstrip('L')
        if class_name.endswith(';'):
            class_name = class_name.rstrip(';')

        key = class_name + '+' + method_name
        if key in self.api_smali_mapping_dict.keys():
            smali_block_list = self.api_smali_mapping_dict[key]
        else:
            logger.warning("No such key '{}.{}'".format(class_name, method_name))
            if force:
                logger.warning("Forcing implementation: Trying generic template for public method.")
                smali_block_list = [self.generate_generic_smali(class_name, method_name, params, is_params_smali)]
            else:
                logger.warn("No obfuscated code generated.")
                print("No obfuscated code generated.")
                return 1

        smali_path_selected = []
        if len(self.partial_smali_paths) > 0:
            smali_path_selected = du.select_smali_file(self.partial_smali_paths, num=1)[0]

        if len(smali_path_selected) == 0:
            logger.warn("No files provided.")
            print("No files provided.")
            raise e.ModifyFileException("Cannot find an avaiable file to insert string {}".format(elem_name))

        try:
            import random
            random.seed(2345)
            assert isinstance(smali_block_list, list)
            used_random_numbers = []

            for t in range(mod_count):
                block_idx = random.choice(range(len(smali_block_list)))

                rdm_number = random.choice(range(23456))
                count = 0
                while rdm_number in used_random_numbers and count < 10:
                    rdm_number = random.choice(range(23456))
                    count = count + 1
                used_random_numbers.append(rdm_number)
                smali_block_code = du.change_method_name(smali_block_list[block_idx], rdm_number)
                du.insert_dead_code(smali_path_selected, smali_block_code)
                if self.verbose:
                    print("Times:{}/{}".format(t + 1, mod_count))
                    print("API Insert: Successfully insert the method '{}.{}' in to file '{}'".format(class_name,
                                                                                                      method_name,
                                                                                                      smali_path_selected))

        except Exception as ex:
            raise e.ModifyFileException(str(ex) + ":Failed to insert method '{}' into file '{}'".format(method_name,
                                                                                                        smali_path_selected))
        logger.info("API Insert: Successfully insert the method '{}.{}' in to file '{}'".format(class_name,
                                                                                                method_name,
                                                                                                smali_path_selected))

    def remove(self, class_name=None,
               method_name=None,
               elem_name=None,
               mod_count=1):
        """
        hide the api call by reflection + string encryption
        :param class_name: the class contains the api, do not support the array
        :param method_name: api method
        :param elem_name: neglected
        :param mod_count: how much api will be modified; if -1 , all elements will be chagned
        """
        if not isinstance(class_name, str) and not isinstance(method_name, str):
            raise TypeError("The type of string is supported")

        if '[' in class_name and (class_name in du.javaBasicT_to_SmaliBasicT.values()) and \
                (class_name in du.javaBasicT_to_SmaliBasicT.keys()):
            raise NotImplementedError("array '[' and basic type (e.g. int, float) are not supported.")

        if '.' in class_name:
            class_name = class_name.replace('.', '/')  # format the class into smali fashion
        if not class_name.startswith('L'):
            class_name = 'L' + class_name
        if not class_name.endswith(';'):
            class_name = class_name + ';'

        try:
            if self.verbose:
                print("Java reflection ...")
            is_modified, files_modified = du.method_reflection(self.all_smali_paths,
                                                               class_name,
                                                               method_name,
                                                               mod_count)
            if is_modified:
                logger.info("Successfully modify APK: Java reflection for '{}' of files '{}'".format(
                    class_name + "->" + method_name,
                    '\n'.join(files_modified)))
                if self.verbose:
                    print("Successfully modify APK: Java reflection for '{}'".format(class_name + "->" + method_name))
            else:
                logger.info("No modification of Java reflection for '{}' of files '{}'".format(
                    class_name + "->" + method_name,
                    '\n'.join(files_modified)))
                if self.verbose:
                    print("No modification of Java reflection for '{}'".format(class_name + "->" + method_name))

            if self.verbose:
                print("String encryption ...")

            is_str_modified, _ = du.encrypt_string(files_modified, method_name, mod_count=-1)
            if is_str_modified:
                logger.info("Successfully modify APK: Java string encryption for '{}' of files '{}'".format(
                    class_name + "->" + method_name,
                    '\n'.join(files_modified)))
                if self.verbose:
                    print("Successfully modify APK: encryption for '{}'".format(class_name + "->" + method_name))
            else:
                logger.info(
                    "No string encryption for '{}' of files '{}'".format(
                        class_name + "->" + method_name,
                        '\n'.join(files_modified)))
                if self.verbose:
                    print("No encryption for '{}'".format(class_name + "->" + method_name))

        except e.ModifyFileException as ex:
            logger.exception(str(ex))
            logger.error("Failed to implement reflection for '{}' of file '{}'".format(class_name + "->" + method_name,
                                                                                       self.disassembly_root))
            raise ("Failed to Modify APK: java reflection, error " + str(ex))


def _main():
    disassembly_dir = "/path/to/disassmbly_apk_dir/"
    api_modify = DroidSysAPIModification(disassembly_dir, True)
    api_modify.remove(class_name='android.widget.VideoView', method_name='stopPlayback')
    return

if __name__ == "__main__":
    _main()
