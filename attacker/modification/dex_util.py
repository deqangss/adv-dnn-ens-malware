from __future__ import print_function

import time
from tools.base_util import *
curr_dir = os.path.dirname(os.path.realpath(__file__))

CONST_STR = 'android/content/res/' # the append smali files will be put at the folder smali/android/contect/res of new APK
ANNOTATION_REF = '''
    .annotation system Ldalvik/annotation/Throws;
        value = {
            Ljava/lang/reflect/InvocationTargetException;,
            Ljava/lang/IllegalAccessException;,
            Ljava/lang/NoSuchMethodException;
        }
    .end annotation
'''
FIELD_TEMPLATE = '.field private static final {stringName}:Ljava/lang/String; = \"{stringValue}\"'
EMPTY_METHOD = '''.method private static final {methodName}()V
{methodBody}

    return-void
.end method
'''

def get_smali_paths(directory):
    try:
        return retrive_files_set(directory, "", ".smali")
    except IOError as ex:
        raise IOError("Failed to load all smali files.")

def get_smali_dirs(directory):
    try:
        return retrive_all_dirs(directory, "smali")
    except IOError as es:
        raise IOError("Failed to load all smali directories")

def filter_smali_paths(paths, signs=['java'], spec_str = 'R$'):
    try:
        paths_retrain = []
        if not isinstance(signs, list):
            signs = [signs]

        for path in paths:
            folder_names = path.split('/')
            flag = True
            for n in folder_names[:-1]:
                if n in signs:
                    flag = False

            if spec_str in folder_names[-1]:
                flag = False

            if flag:
                paths_retrain.append(path)

        return paths_retrain
    except ValueError as ex:
        raise ValueError("The input is incorrect.")

javaBasicT_to_SmaliBasicT = {
    'boolean' : 'Z',
    'byte' : 'B',
    'short' : 'S',
    'char' : 'C',
    'int' : 'I',
    'float' : 'F',
    'long' : 'J',
    'double' : 'D'
}

def get_param_smali_type(params, is_smali = True):
    """Get arugments' smali type of specified params which is used by a method"""
    param_types_smali = []
    if params == '':
        return param_types_smali

    if is_smali:
        class_param_types = params.split(';')
        for pt in class_param_types:  # type: string
            pt = pt.strip().replace(' ', '')
            if pt is '':
                continue

            tmp_prefix = []
            for i, c in enumerate(pt):
                if c in javaBasicT_to_SmaliBasicT.values():
                    if len(tmp_prefix) > 0:
                        param_types_smali.append(''.join(tmp_prefix) + c)
                        tmp_prefix = []
                    else:
                        param_types_smali.append(c)
                elif c == '[':
                    tmp_prefix.append(c)
                elif c == 'L':
                    if len(tmp_prefix) > 0:
                        param_types_smali.append(''.join(tmp_prefix) + pt[i:] + ";")
                    else:
                        param_types_smali.append(pt[i:] + ";")
                    break
                else:
                    #print("okfdfd ", pt, i, c, params)
                    raise ValueError("The symbol '{}' are negected".format(str(c)))
        return param_types_smali
    else:
        #here, the parameter type is split by symbol ','
        param_types_java = params.split(',')
        assert isinstance(param_types_java, list)
        for param_type in param_types_java:
            if '[' not in param_type:  #we here only do not consider the advance variable type, e.g., generic type '<T>'
                if param_type in javaBasicT_to_SmaliBasicT.keys():
                    param_types_smali.append(javaBasicT_to_SmaliBasicT[param_type])
                else:
                    param_type = param_type.replace('.', '/')
                    param_types_smali.append('L' + param_type +  ';')
            else:
                raw_param_type = param_type.split('[')[-1]
                if raw_param_type in javaBasicT_to_SmaliBasicT.keys():
                    param_types_smali.append(param_type.replace(raw_param_type, '')+javaBasicT_to_SmaliBasicT[raw_param_type])
                else:
                    prefix_array = param_type.replace(raw_param_type, '')
                    raw_param_type = raw_param_type.replace('.', '/')
                    param_types_smali.append(prefix_array + 'L' + raw_param_type + ';')
        return param_types_smali

def insert_const_string(smali_path_selected, string_value, mod_count = 1):
    if not os.path.exists(smali_path_selected):
        return False

    smali_codes = read_full_file(smali_path_selected)
    smali_method_blocks = smali_codes.split('.method ')

    if len(smali_method_blocks) > 1:
        flag = False
        used_random_numbers = []
        for t in range(mod_count):
            block_idx = random.choice(range(1, len(smali_method_blocks)))

            targeted_method_block = smali_method_blocks[block_idx]
            block_lines = targeted_method_block.splitlines()

            for i, line in enumerate(block_lines):
                if '.locals' in line:
                    reg_match = re.search(r'^[ ]*?(.locals)[ ]*?(?P<regNumber>\d)', line)
                    if reg_match is not None:
                        reg_number = int(reg_match.group('regNumber'))
                        if reg_number <= 0:
                            pass
                            # block_lines[i] = '    .locals {}'.format(reg_number + 1) # this line may trigger some issue
                        else:
                            block_lines[i] = '    .locals {}'.format(reg_number)
                        block_lines.insert(i + 1, "    const-string v{}, \"{}\"".format(0, string_value))
                    flag = True
            modified_block = '\n'.join(block_lines)
            smali_method_blocks[block_idx] = modified_block

            if not flag:
                rdm_number = random.choice(range(23456))
                count = 0
                while rdm_number in used_random_numbers and count < 100:
                    rdm_number = random.choice(range(23456))
                    count = count + 1
                method_name = random_name(rdm_number)
                method_body = '\n' + '    .locals 1' + '\n' + \
                              "    const-string v{}, \"{}\"".format(0, string_value)
                empty_method = EMPTY_METHOD.format(methodName=method_name,
                                                   methodBody = method_body)
                smali_method_blocks.append(empty_method.replace('.method ', ''))
                flag = True

        if not flag:
            smali_method_blocks[0] = smali_method_blocks[0] + '\n\n' + FIELD_TEMPLATE.format(stringName = string.upper(random_string(string_value)),
                                                                                             stringValue = string_value) + '\n\n'

        modified_smali_code = '\n.method '.join(smali_method_blocks)

        write_whole_file(modified_smali_code, smali_path_selected)

        return True
    else:
        return False


def encrypt_line(smali_line, name_string, encryption_class_name):
    """encrypt the 'name_string' in the 'smali_line', codes are adapted from:https://github.com/necst/aamo"""
    is_modified = False
    if name_string == '':
        print(smali_line.rstrip())
        return is_modified

    enc_name_string = apply_encryption(name_string)
    if '.field' in smali_line:
        field_match = re.search(r'^[ ]*?\.field(.*?) static final (?P<fieldName>([^ ]*?)):Ljava\/lang\/String\; = \"(?P<fieldValue>[^\"]*?)\"$', smali_line)
        if field_match is not None:
            field_var_name = field_match.group('fieldName')
            field_value = field_match.group('fieldValue')
            if field_var_name == name_string and '$' not in name_string:
                print(smali_line.replace('"' + field_value + '"', '"' + enc_name_string + '"').rstrip())
            else:
                print(smali_line.rstrip())
        else:
            print(smali_line.rstrip())

    elif 'const-string' in smali_line:
        const_string_match = re.search(
            r'^([ ]*?)(?P<constType>const\-string|const\-string\/jumbo) (?P<regType>v|p)(?P<regNum>\d{1,2}), \"(?P<stringConst>.*?)\"$',
            smali_line)

        if const_string_match is None:
            print(smali_line.rstrip())
            return is_modified

        if name_string not in const_string_match.group('stringConst'):
            print(smali_line.rstrip())
            return is_modified

        v_string = const_string_match.group('regType') + const_string_match.group('regNum')
        print('    ' + const_string_match.group('constType') + ' ' + v_string + ', "' + enc_name_string + '"')
        print('')
        print('    invoke-static/range {' + v_string + ' .. ' + v_string + '}, ' + 'L' + CONST_STR + encryption_class_name + ';->convertToString(Ljava/lang/String;)Ljava/lang/String;')
        print('')
        print('    move-result-object ' + v_string)
        is_modified = True
    else:
        print(smali_line.rstrip())

    return is_modified

def encrypt_string(all_smali_paths, name_string, mod_count=1):
    """Encrypt the corresponding string in the smali files"""
    if len(all_smali_paths) == 0:
        return False, []

    if isinstance(all_smali_paths, str):
        all_smali_paths = [all_smali_paths]

    if not (isinstance(all_smali_paths, (list, set, str))):
        raise ValueError("Input value is incorrect.")

    is_modified = False
    modified_files = []
    change_count = 0
    count_lock = bool(mod_count > 0)
    enc_class_name = random_name()

    for smali_file_path in all_smali_paths:
        if not os.path.exists(smali_file_path):
            continue

        fh = read_file_by_fileinput(smali_file_path, inplace = True)

        for smali_line in fh:
            if change_count > mod_count and count_lock:
                print(smali_line.rstrip())
                continue
            if name_string in smali_line: # need to accomodate the 'ascii' encoded binary array
                if encrypt_line(smali_line, name_string, enc_class_name):
                    is_modified = True
                    change_count = change_count + 1
            else:
                print(smali_line.rstrip())
        if is_modified:
            modified_files.append(smali_file_path)

        fh.close()

    if is_modified:
        smali_root_path = list(all_smali_paths)[0].split('/smali/')[0] + '/smali'
        write_decryption_smali_dir = os.path.join(smali_root_path, CONST_STR)

        if not os.path.exists(write_decryption_smali_dir):
            os.makedirs(write_decryption_smali_dir)
        decryption_class = read_full_file(os.path.join(curr_dir, 'res/DecryptString.smali'))

        decryption_class = decryption_class.replace('DecryptString', enc_class_name)
        dst_path = os.path.join(write_decryption_smali_dir,enc_class_name+".smali")

        if not os.path.exists(dst_path):
            write_whole_file(decryption_class, dst_path)

    return is_modified, modified_files

def name2path(name):
    return name.replace('.', '/')

def abs_path_comp(path, pkg_path):
    if pkg_path in path:
        abs_path = path
    elif (path.startswith('/')) or (len(path.split('/')) == 1):
        abs_path = (pkg_path + '/' + path).replace('//', '/')
    else:
        abs_path = path
    return abs_path

def find_smali_w_name(smali_paths, source_name):
    path_ext = name2path(source_name)
    for smali_file_path in smali_paths:
        if path_ext in smali_file_path:
            yield smali_file_path

def change_source_name(smali_paths, act_source_name, act_dst_name):
    """
    change .source "XXXX.java"
    :param smali_paths: .samli files
    :param act_source_name: XXXX
    :param act_dst_name: targeted name
    :return:
    """
    src_subnames = act_source_name.split('.')
    dst_subnames = act_dst_name.split('.')
    try:
        assert len(src_subnames) == len(dst_subnames)
    except Exception:
        raise AssertionError("Alignment error: source name does not align with destination name.")

    for sf in smali_paths:
        for smali_line in read_file_by_fileinput(sf, inplace=True):
            source_match = re.search(r'^([ ]*?)\.source(.*?)\"(?P<source_name>([^\"]*?))\"', smali_line)

            if source_match is not None:
                naive_source_name = source_match.group('source_name')
                niv_src_subnames = naive_source_name.split('.')
                new_name = naive_source_name
                for _, niv_name in enumerate(niv_src_subnames):
                    for i, src_name in enumerate(src_subnames):
                        if niv_name in src_name:
                            new_name = new_name.replace(niv_name, dst_subnames[i])
                smali_line = smali_line.replace(naive_source_name, new_name)
            print(smali_line.rstrip())

def change_class_name(smali_paths, source_name, dst_name, pkg_name):
    """
    change class name of definition
    :param smali_paths: set of smali paths
    :param source_name: original class name
    :param dst_name: modified class name
    :return: class name found based on the original class name
    """
    # related_smali = set(find_smali_w_name(smali_paths, source_name))
    pkg_path = name2path(pkg_name)
    act_src_path = abs_path_comp(name2path(source_name), pkg_path)
    act_dst_path = abs_path_comp(name2path(dst_name), pkg_path)
    for sf in smali_paths:
        fi_sf = read_file_by_fileinput(sf, inplace=True)
        for smali_line in fi_sf:
            class_match = re.search(r'^([ ]*?)\.class(.*?)(?P<class_name>L([^;\(\) ]*?);)', smali_line)
            if class_match is not None:
                class_name = class_match.group('class_name')
                if act_src_path in class_name:
                    smali_line = smali_line.replace(act_src_path, act_dst_path)
                    yield class_name
                print(smali_line.rstrip()) # append and write the smali line
            else:
                print(smali_line.rstrip())
        fi_sf.close()

def change_instantition_name(smali_paths, related_class, source_name, dst_name, pkg_name):
    """
    change instantiated class name (e.g., new-instance, invoke-type, arguments, etc)
    :param smali_paths: set of smali paths
    :param related_class: class names based on the source name
    :param source_name: original class name
    :param dst_name: modified class name
    """
    pkg_path = name2path(pkg_name)
    src_path = abs_path_comp(name2path(source_name), pkg_path)
    dst_path = abs_path_comp(name2path(dst_name), pkg_path)

    for smali_path in smali_paths:
        for smali_line in read_file_by_fileinput(smali_path, inplace=True):
            if re.search(r'L([^;\(\) ]*?);', smali_line) is None:
                print(smali_line.rstrip())
            else:
                for class_name in related_class:
                    if class_name in smali_line:
                        smali_line = smali_line.replace(
                            src_path,
                            dst_path
                        )
                        # break
                print(smali_line.rstrip())

def rename_smali_file(smali_path, activity_name, new_activity_name):
    dir_name, file_name, ext_name = path_split(smali_path)
    act_name = activity_name.split('.')[-1]
    new_act_name = new_activity_name.split('.')[-1]

    if not os.path.exists(smali_path) or (act_name not in smali_path):
        return
    #try:
    #    assert(act_name in file_name)
    #except AssertionError:
    #    raise AssertionError("assert error:{}".format(smali_path))

    src = os.path.join(dir_name, file_name + ext_name)
    if act_name == file_name:
        dst = os.path.join(dir_name, new_act_name + ext_name)
    else:
        file_name = fix_invalid_id(file_name)
        inner_names = file_name.split('$')
        for idx, name in enumerate(inner_names):
            if act_name == name:
                inner_names[idx] = new_act_name
        new_file_name = '$'.join(inner_names)
        dst = os.path.join(dir_name, new_file_name + ext_name)

    rename_file(src,dst)


def rename_smali_dir(smali_dir, activity_name, new_activity_name):
    # type: (string, string, string) -> void
    if not os.path.exists(smali_dir):
        return

    def rename(src_path, new_path):
        if src_path is not '' and src_path in smali_dir:
            new_smali_dir = smali_dir.replace(src_path, new_path)
            rename_tree_dir(smali_dir, new_smali_dir)
            return True
        else:
            return False

    act_path1 = os.path.dirname(name2path(activity_name))
    new_act_path = os.path.dirname(name2path(new_activity_name))
    rename(act_path1, new_act_path)

    act_path2 = name2path(activity_name)
    new_act_path = name2path(new_activity_name)
    rename(act_path2, new_act_path)


def select_smali_file(smali_paths, num = 1):
    try:
        selected_paths = []
        # random.seed(seed)
        for i in range(num):
            # idx = random.choice(range(len(smali_paths)))
            trial_times = 0
            while trial_times < 50000:
                trial_times = trial_times + 1
                path = random.choice(list(smali_paths))
                if '.method ' in read_full_file(path):
                    selected_paths.append(path)
                    break

        return selected_paths
    except ValueError as ex:
        raise ValueError(str(ex))

def change_method_name(block_smali_method, rdm_number = 2):
    smali_lines = block_smali_method.split('\n')
    new_smali_lines = []
    for smali_line in smali_lines:
        method_def_match = re.match(r'^([ ]*?)\.method(.*?) (?P<methodName>([^ ]*?))\((?P<methodArg>(.*?))\)(?P<methodRtn>(.*?))$', smali_line)
        if method_def_match is None:
            new_smali_lines.append(smali_line)
        else:
            method_name = method_def_match.group('methodName')
            new_smali_lines.append(smali_line.replace(method_name+'(', method_name + random_name(rdm_number) + '('))

    return '\n'.join(new_smali_lines)



def insert_dead_code(smali_file_path, smali_block):
    smali_codes = read_full_file(smali_file_path)

    smali_method_chuchs = smali_codes.split('.method ')
    if len(smali_method_chuchs) > 1:
        insert_idx = random.choice(range(1, len(smali_method_chuchs)))
    else:
        insert_idx = 1
    smali_block = smali_block.replace('.method ', '')
    smali_method_chuchs.insert(insert_idx, smali_block)
    re_smali_codes = '.method '.join(smali_method_chuchs)

    write_whole_file(re_smali_codes, smali_file_path)

def is_specfic_exsit(desired_str, src):
    """justify desired string in the src"""
    if desired_str in src:
        return True
    else:
        return False

def split_invoke_argument(invoke_argument):
    """Match an invocation type of the parameter"""
    for find_item in re.findall(r'(L[^;]*?;)|(\[L[^;]*?;)|(\[V)|(\[Z)|(\[B)|(\[S)|(\[C)|(\[I)|(\[J)|(\[F)|(\[D)|(V)|(Z)|(B)|(S)|(C)|(I)|(J)|(F)|(D)', invoke_argument):
        for find_match in find_item:
            if find_match != '':
                yield find_match

basic_class_dict = {
    'I': 'Ljava/lang/Integer;->TYPE:Ljava/lang/Class;',
    'Z': 'Ljava/lang/Boolean;->TYPE:Ljava/lang/Class;',
    'B': 'Ljava/lang/Byte;->TYPE:Ljava/lang/Class;',
    'S': 'Ljava/lang/Short;->TYPE:Ljava/lang/Class;',
    'J': 'Ljava/lang/Long;->TYPE:Ljava/lang/Class;',
    'F': 'Ljava/lang/Float;->TYPE:Ljava/lang/Class;',
    'D': 'Ljava/lang/Double;->TYPE:Ljava/lang/Class;',
    'C': 'Ljava/lang/Character;->TYPE:Ljava/lang/Class;'
}

basic_object_obtaining_dict = {
    'I': 'Ljava/lang/Integer;->valueOf(I)Ljava/lang/Integer;',
    'Z': 'Ljava/lang/Boolean;->valueOf(Z)Ljava/lang/Boolean;',
    'B': 'Ljava/lang/Byte;->valueOf(B)Ljava/lang/Byte;',
    'S': 'Ljava/lang/Short;->valueOf(S)Ljava/lang/Short;',
    'J': 'Ljava/lang/Long;->valueOf(J)Ljava/lang/Long;',
    'F': 'Ljava/lang/Float;->valueOf(F)Ljava/lang/Float;',
    'D': 'Ljava/lang/Double;->valueOf(D)Ljava/lang/Double;',
    'C': 'Ljava/lang/Character;->valueOf(C)Ljava/lang/Character;'
}

basic_type = {
    'I': 'Ljava/lang/Integer;',
    'Z': 'Ljava/lang/Boolean;',
    'B': 'Ljava/lang/Byte;',
    'S': 'Ljava/lang/Short;',
    'J': 'Ljava/lang/Long;',
    'F': 'Ljava/lang/Float;',
    'D': 'Ljava/lang/Double;',
    'C': 'Ljava/lang/Character;'
}

reverse_cast_dict = {
    'I': 'Ljava/lang/Integer;->intValue()I',
    'Z': 'Ljava/lang/Boolean;->booleanValue()Z',
    'B': 'Ljava/lang/Byte;->byteValue()B',
    'S': 'Ljava/lang/Short;->shortValue()S',
    'J': 'Ljava/lang/Long;->longValue()J',
    'F': 'Ljava/lang/Float;->floatValue()F',
    'D': 'Ljava/lang/Double;->doubleValue()D',
    'C': 'Ljava/lang/Character;->charValue()C'
}

def is_class(class_name_smali):
    if re.search(r'L[^;]*?;', class_name_smali).group() == class_name_smali:
        return True
    return False


def is_wide_type(invoke_type):
    if invoke_type == 'J' or invoke_type == 'D':
        return True
    else:
        return False

def is_void(invoke_return):
    if invoke_return == 'V':
        return True
    return False


def is_wide(invoke_return):
    if invoke_return == 'J' or invoke_return == 'D':
        return True
    return False


def is_obj(invoke_return):
    if re.search(r'L([^;]*?);|\[|\[L([^;]*?);', invoke_return) is not None:
        return True
    return False

def change_invoke_by_ref(new_class_name, method_fh, ivk_type, ivk_param, ivk_object, ivk_method, ivk_argument, ivk_return):
    """change invocation by java reflection"""
    identifier = ivk_type + ivk_object + ivk_method + ivk_argument + ivk_return
    new_method_name = string_on_code(identifier)

    is_range = is_specfic_exsit('range', ivk_type)

    is_static = is_specfic_exsit('static', ivk_type)

    extra_argument = ''
    if not is_static:
        extra_argument = ivk_object

    new_invoke_type = 'invoke-static'
    if is_range:
        new_invoke_type = new_invoke_type + '/range'

    # put the original invoked object to arguments if ivk_type is public
    new_smali_line = '    ' + new_invoke_type + ' {' + ivk_param + '}, ' + new_class_name + '->' + new_method_name + '(' + extra_argument + ivk_argument + ')' + ivk_return
    print(new_smali_line.rstrip())

    # generate the new method
    method_declaration = '.method public static ' + new_method_name + '(' + extra_argument + ivk_argument + ')' + ivk_return
    if method_declaration in method_fh:
        return method_fh

    method_fh += '\n'
    method_fh = method_fh + method_declaration + '\n'

    list_arguments = list(split_invoke_argument(ivk_argument))
    count_arguments = len(list_arguments)

    method_fh = method_fh + '    ' + '.locals ' + str(5+count_arguments) + '\n'
    method_fh = method_fh + ANNOTATION_REF + '\n'

    # reflection for method
    method_fh = method_fh + '    ' + 'const-class v0, ' + ivk_object + '\n\n'
    method_fh = method_fh + '    ' + 'const-string v1, \"' + ivk_method + '\"' + '\n\n'

    # handle class
    method_fh = method_fh + '    ' + 'const v2, ' + hex(count_arguments) + '\n\n' #array length
    method_fh = method_fh + '    ' + 'new-array v3, v2, [Ljava/lang/Class;' + '\n\n' #array
    for idx, arg in enumerate(list_arguments):
        # obtain class
        class_type = basic_class_dict.get(arg, '')
        if class_type != '':
            method_fh = method_fh + '    ' + 'sget-object v4, ' + class_type + '\n\n'
        else:
            method_fh = method_fh + '    ' + 'const-class v4, ' + arg + '\n\n'
        # assign value for the array
        method_fh = method_fh + '    ' + 'const v{}, '.format(5 + idx) + hex(idx) + '\n\n' #index
        method_fh = method_fh + '    ' + 'aput-object v4, v3, v{}'.format(5 + idx) + '\n\n'

    # call 'getMethod'
    method_fh = method_fh + '    ' + 'invoke-virtual {v0, v1, v3}, Ljava/lang/Class;->getMethod(Ljava/lang/String;[Ljava/lang/Class;)Ljava/lang/reflect/Method;' + '\n\n'
    method_fh = method_fh + '    ' + 'move-result-object v0' + '\n\n'


    # handle object (class instantiation)
    method_fh = method_fh + '    ' + 'new-array v1, v2, [Ljava/lang/Object;' + '\n\n'
    if is_static:
        param_idx = 0 #no need p0
    else:
        param_idx = 1

    for idx, arg in enumerate(list_arguments):
        object_tail_invocation = basic_object_obtaining_dict.get(arg, '')
        if object_tail_invocation != '':
            if is_wide_type(arg):
                method_fh = method_fh + '    ' + 'invoke-static/range {p' + str(param_idx) + ' .. p' + str(param_idx + 1) + '}, '
            else:
                method_fh = method_fh + '    ' + 'invoke-static/range {p' + str(param_idx) + ' .. p' + str(param_idx) + '}, '
            method_fh = method_fh + object_tail_invocation + '\n\n'
            method_fh = method_fh + '    ' + 'move-result-object v2' + '\n\n'
            #put into object array
            method_fh = method_fh + '    ' + 'aput-object v2, v1, v{}'.format(idx + 5) + '\n\n'
            if is_wide_type(arg):
                param_idx = param_idx + 1 # wide type of value occupys two registers
        else:
            method_fh = method_fh + '    ' + 'aput-object p' + str(param_idx) + ', v1, v{}'.format(idx+5) + '\n\n'
        param_idx = param_idx + 1

    # call Invoke, note that static method has no p0
    if is_static:
        method_fh = method_fh + '    ' + 'const-class v2, ' + ivk_object + '\n\n' #v2 could be null (i.e., const v2, 0x0)?
        method_fh = method_fh + '    ' + 'invoke-virtual {v0, v2, v1}, Ljava/lang/reflect/Method;->invoke(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;' + '\n\n'
    else:
        method_fh = method_fh + '    ' + 'invoke-virtual {v0, p0, v1}, Ljava/lang/reflect/Method;->invoke(Ljava/lang/Object;[Ljava/lang/Object;)Ljava/lang/Object;' + '\n\n'

    is_void_value = is_void(ivk_return)
    is_obj_value = is_obj(ivk_return)
    is_wide_value = is_wide(ivk_return)

    if is_void_value:
        method_fh = method_fh + '    ' + 'return-void' + '\n\n'
    elif is_obj_value:
        method_fh = method_fh + '    ' + 'move-result-object v0' + '\n\n'
        method_fh = method_fh + '    ' + 'check-cast v0, ' + ivk_return + '\n\n'
        method_fh = method_fh + '    ' + 'return-object v0' + '\n\n'
    else:
        rt_type = basic_type.get(ivk_return, '')
        method_fh = method_fh + '    ' + 'move-result-object v0' + '\n\n'
        method_fh = method_fh + '    ' + 'check-cast v0, ' + rt_type + '\n\n'
        basic_type_value_getting = reverse_cast_dict.get(ivk_return, '')
        method_fh = method_fh + '    ' + 'invoke-virtual/range {v0 .. v0}, ' + basic_type_value_getting + '\n\n'
        if is_wide_value:
            method_fh = method_fh + '    ' + 'move-result-wide v0' + '\n\n'
            method_fh = method_fh + '    ' + 'return-wide v0' + '\n\n'
        else:
            method_fh = method_fh + '    ' + 'move-result v0' + '\n\n'
            method_fh = method_fh + '    ' + 'return v0' + '\n\n'
    method_fh = method_fh + '.end method' + '\n\n'
    return method_fh


def method_reflection(all_smali_paths, class_name_smali, method_name_smali, mod_count=1):
    """
    apply the java reflection to Android API method
    :param all_smali_paths: smali file paths provided, should not be null
    :param class_name_smali: class name (smali type, e.g., Landroid/.../..;) of API
    :param method_name_smali: API method name
    :param mod_count: the maximum mount of modification
    :return: are some smali file modified, and what are they
    """
    if len(all_smali_paths) == 0:
        return False, []
    assert is_class(class_name_smali)

    new_method_fh = read_full_file(os.path.join(curr_dir, 'res/MethodReflection.smali'))
    raw_name = 'Ref' + random_name(seed=int(time.time()), code=class_name_smali)
    new_class_name = 'L' + CONST_STR + raw_name + ';'
    new_method_fh = new_method_fh.replace('MethodReflection', raw_name)
    is_modified = False
    change_count = 0
    count_lock = bool(mod_count>0)
    file_changed_list = []

    new_method_body = new_method_fh
    for idx, smali_path in enumerate(all_smali_paths):
        smali_fh = read_file_by_fileinput(smali_path, inplace=True)

        for smali_line in smali_fh:
            if re.match(r'^([ ]*?)invoke-((virtual)|(static)|(super))', smali_line) is None:
                print(smali_line.rstrip())
                continue

            if change_count >= mod_count and count_lock:
                print(smali_line.rstrip())
                continue

            invoke_match = re.search(
                r'^([ ]*?)(?P<invokeType>invoke\-([^ ]*?)) {(?P<invokeParam>([vp0-9,. ]*?))}, (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$',
                smali_line)

            if invoke_match is None:
                print(smali_line.rstrip())
            elif invoke_match.group('invokeObject') == class_name_smali and invoke_match.group('invokeMethod') == method_name_smali:
                new_method_body = change_invoke_by_ref(new_class_name,
                                                       new_method_body,  # append method
                                                       invoke_match.group('invokeType'),
                                                       invoke_match.group('invokeParam'),
                                                       invoke_match.group('invokeObject'),
                                                       invoke_match.group('invokeMethod'),
                                                       invoke_match.group('invokeArgument'),
                                                       invoke_match.group('invokeReturn')
                                                       )
                is_modified = True
                change_count = change_count + 1
                file_changed_list.append(smali_path)
            else:
                print(smali_line.rstrip())
        smali_fh.close()

    # write to disk
    if is_modified:
        smali_root_path = list(all_smali_paths)[0].split('/smali/')[0] + '/smali'
        write_ref_smali_dir = os.path.join(smali_root_path, CONST_STR)

        if not os.path.exists(write_ref_smali_dir):
            os.makedirs(write_ref_smali_dir)

        dst_path = os.path.join(write_ref_smali_dir, raw_name + ".smali")
        if not os.path.exists(dst_path):
            write_whole_file(new_method_body, dst_path)
        file_changed_list.append(dst_path)

    return is_modified, file_changed_list


def _main():

    plain_text = 'ui.finishscreen.buttons.exit.caption'
    encrypted_text = 'QQgbVlFXXBAJFwIXUVJbHwMXEBEKWRFKBE0KEUsCU0dACFpe'
    get_text = apply_encryption(plain_text)

    print(get_text == encrypted_text)
    print(get_text)


if __name__ == "__main__":
    _main()