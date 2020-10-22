'''
Further work: modifier serves as a service
'''

import os
import shutil
import sys
import multiprocessing
import subprocess
import time
from collections import defaultdict

sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())

from attacker.modification import *
from tools import utils, progressbar_wrapper
from config import config, COMP, logging, ErrorHandler

logger = logging.getLogger("attacker.modifier")
logger.addHandler(ErrorHandler)

OPERATOR = {
    # insert
    0: "insert",
    # remove
    1: "remove"
}

INSTR_ALLOWED = {
    OPERATOR[0]: [COMP['Permission'],
                  COMP['Activity'],
                  COMP['Service'],
                  COMP['Receiver'],
                  COMP['Hardware'],
                  COMP['Intentfilter'],
                  COMP['Android_API'],
                  COMP['User_String']
                  ],
    OPERATOR[1]: [COMP['Activity'],
                  COMP['Service'],
                  COMP['Receiver'],
                  COMP['Provider'],
                  COMP['Android_API'],
                  COMP['User_String']
                  ]
}

MetaInstTemplate = "{Operator}##{Component}##{SpecName}##{Count}"
APIInstrSpecTmpl = "{ClassName}::{ApiName}::{ApiParameter}"
MetaDelimiter = '##'
SpecDelimiter = '::'


def name_adv_file(name):
    name = os.path.splitext(os.path.basename(name))[0]
    return name + "_adv"


def get_original_name(name):
    name = os.path.splitext(os.path.basename(name))[0]
    if "_adv" in name:
        return name.rsplit("_adv", 1)[0]
    else:
        raise ValueError


def _assemble_apk(src_dir, dst_file, verbose=True):
    try:
        if verbose:
            sys.stdout.write("Encoding {0} files to {1}.\n".format(src_dir, dst_file))
            command_run = subprocess.call("apktool b " + src_dir + " -o " + dst_file, shell=True)
        else:
            command_run = subprocess.call("apktool -q b " + src_dir + " -o " + dst_file, shell=True)

        if command_run == 0:
            return dst_file, True
        else:
            currentTime = time.time()
            logger.error("Encoding: " + src_dir + " processing failed in " + str(currentTime) + "s...")
            sys.stderr.write(src_dir + ": " + " processing failed in " + str(currentTime) + "s...")
            return dst_file, False
    except Exception as ex:
        logger.exception(ex)
        logger.error("Cannot build foler {}:{}".format(src_dir, str(ex)))
        return dst_file, False


def assemble_apks(apk_paths, outpur_dir, work_dir, proc_number=4, verbose=True):
    pool = multiprocessing.Pool(int(proc_number))
    results = []
    for i, path in enumerate(apk_paths):
        name, ext = os.path.splitext(os.path.basename(path))
        src = os.path.join(work_dir, name)
        if ext == '':
            ext = ".apk"
        dst = outpur_dir + "/" + name_adv_file(path) + ext
        results.append(pool.apply_async(_assemble_apk, args=[src, dst, verbose]))

    pool.close()
    adv_apk_paths = []
    for i, res in enumerate(results):
        new_path, status = res.get()
        sys.stdout.write(new_path + " result: {0}.\n".format(status))
        if status:
            adv_apk_paths.append(new_path)
    pool.join()
    return adv_apk_paths


def _disassemble_apk(src_file, dst_dir, verbose=True):
    try:
        if verbose:
            sys.stdout.write("decoding {0} apk file to {1}.\n".format(src_file, dst_dir))
            command_run = subprocess.call("apktool d " + src_file + " -o " + dst_dir, shell=True)
        else:
            command_run = subprocess.call("apktool -q d " + src_file + " -o " + dst_dir, shell=True)
        if command_run == 0:
            return src_file, True
        else:
            currentTime = time.time()
            logger.error("Decoding: " + src_file + " processing failed in " + str(currentTime) + "s...")
            return src_file, False
    except Exception as ex:
        logger.exception(ex)
        logger.error("Cannot disassemble apk file {}:{}".format(src_file, str(ex)))
        return src_file, False


def disassemble_apks(malware_samples, work_dir, proc_number=6, verbose=True):
    """
    disassemble apks
    :param malware_samples: list of malware sample paths
    :param work_dir: directory for conducting disassembly
    :param verbose: print the info on the shell
    :return: True if perform does successfully, False if not
    """
    pool = multiprocessing.Pool(int(proc_number))

    pbar = progressbar_wrapper.ProgressBar()
    process_results = []
    for i, path in enumerate(malware_samples):
        src = path
        dst = os.path.join(work_dir, os.path.splitext(os.path.basename(path))[0])
        if os.path.exists(dst):
            shutil.rmtree(dst)
        process_results = pool.apply_async(_disassemble_apk,
                                           args=[src, dst, verbose],
                                           callback=pbar.CallbackForProgressBar
                                           )
    pool.close()
    if process_results:
        pbar.DisplayProgressBar(process_results, len(malware_samples), type = 'hour')
    pool.join()

    for file_name, res in pbar.TotalResults:
        sys.stdout.write("Disassemble: " + file_name + " result: {0}.\n".format(res))
    return 0


def _sign_apk(sample_file_name, verbose=True):  # Sign an apk file with a SHA1 key
    if verbose:
        sys.stdout.write("Sign: " + sample_file_name)
    command_run = \
        subprocess.call("jarsigner -sigalg MD5withRSA -digestalg SHA1 -keystore " + os.path.join(
            config.get("DEFAULT", 'project_root'),
            "attacker/modification/res/resignKey.keystore") + " -storepass resignKey " + sample_file_name + ' resignKey',
                        shell=True)
    if command_run == 0:
        return sample_file_name, True
    else:
        currentTime = time.time()
        logger.error("Signing: " + sample_file_name + " processing failed in " + str(currentTime) + "s...\n")
        return sample_file_name, False


def sign_apks(file_names, proc_number=6, verbose=True):
    """
    sign apks
    :param file_names: apks
    :param verbose: print info
    """
    pool = multiprocessing.Pool(int(proc_number))
    results = []
    for i, path in enumerate(file_names):
        if os.path.isfile(path):
            results.append(pool.apply_async(_sign_apk, args=[path, verbose]))
    for i, res in enumerate(results):
        name, status = res.get()
        sys.stdout.write(name + " result: {0}.\n".format(status))


# @Unchangeable method name
def insert_activity(specific, mod_count, disassemble_root, verbose=True):
    try:
        if verbose:
            print("Insert '{}' activity in the manifest file".format(specific))
        act_mod = DroidActivityModification(disassemble_root, verbose)
        act_mod.insert(specific, mod_count)
    except Exception as ex:
        logger.error("Insert '{}' activity in the manifest file '{}': {}".format(specific, disassemble_root, str(ex)))


# @Unchangeable method name
def remove_activity(specific, mod_count, disassemble_root, verbose=True):
    try:
        if verbose:
            print("Change '{}' activity in the manifest file".format(specific))
        act_mod = DroidActivityModification(disassemble_root, verbose)
        act_mod.remove(specific, mod_count=-1)  # change all the file accordingly
    except Exception as ex:
        logger.error("Failed to change '{}' activity name in manifest file '{}': {}".format(
            specific,
            disassemble_root,
            str(ex)
        ))


# @Unchangeable method name
def insert_service(specific, mod_count, disassemble_root, verbose=True):
    try:
        if verbose:
            print("Insert '{}' service in the manifest file".format(specific))
        svc_mod = DroidServiceModification(disassemble_root, verbose)
        svc_mod.insert(specific, mod_count)
    except Exception as ex:
        logger.error("Insert '{}' service in the manifest file of '{}': {}".format(specific, disassemble_root, str(ex)))


# @Unchangeable method name
def remove_service(specific, mod_count, disassemble_root, verbose=True):
    try:
        if verbose:
            print("Change '{}' service in the manifest file".format(specific))
        svc_mod = DroidServiceModification(disassemble_root, verbose)
        svc_mod.remove(specific, -1)  # change all the file accordingly
    except Exception as ex:
        logger.error("Failed to change '{}' service in the manifest file of '{}': {}".format(specific, disassemble_root,
                                                                                             str(ex)))


# @Unchangeable method name
def insert_receiver(specific, mod_count, disassemble_root, verbose=True):
    try:
        if verbose:
            print("Insert '{}' receiver in the manifest file".format(specific))
        rx_mod = DroidReceiverModification(disassemble_root, verbose)
        rx_mod.insert(specific, mod_count)
    except Exception as ex:
        logger.error(
            "Insert '{}' receiver in the manifest file of '{}': {}".format(specific, disassemble_root, str(ex)))


# @Unchangeable method name
def remove_receiver(specific, mod_count, disassemble_root, verbose=True):
    try:
        if verbose:
            print("Change '{}' receiver in the manifest file".format(specific))
        rx_mod = DroidReceiverModification(disassemble_root, verbose)
        rx_mod.remove(specific, mod_count=-1)  # change all the file accordingly
    except Exception as ex:
        logger.error(
            "Failed to change '{}' receiver in the manifest file of '{}': {}".format(specific, disassemble_root,
                                                                                     str(ex)))


# @Unchangeable method name
def remove_provider(specific, mod_count, disassemble_root, verbose=True):
    try:
        if verbose:
            print("Change '{}' provider in the manifest file".format(specific))
        pvd_mod = DroidProviderModification(disassemble_root, verbose)
        pvd_mod.remove(specific, mod_count=-1)  # change all the file accordingly
    except Exception as ex:
        logger.error(
            "Failed to change '{}' provider in the manifest file of '{}': {}".format(specific, disassemble_root,
                                                                                     str(ex)))


# @Unchangeable method name
def insert_permission(specific, mod_count, disassemble_root, verbose=True):
    try:
        if verbose:
            print("Insert '{}' permission in the manifest file".format(specific))
        perm_mod = DroidPermModification(disassemble_root, verbose)
        perm_mod.insert(specific, mod_count)
    except Exception as ex:
        logger.error(
            "Insert '{}' permission in the manifest file of '{}': {}".format(specific, disassemble_root, str(ex)))


# @Unchangeable method name
def insert_hardware(specific, mod_count, disassemble_root, verbose=True):
    try:
        if verbose:
            print("Insert '{}' hardware in the manifest file".format(specific))
        perm_mod = DoridHardwareModification(disassemble_root, verbose)
        perm_mod.insert(specific, mod_count)
    except Exception as ex:
        logger.error(
            "Insert '{}' hardware in the manifest file of '{}': {}".format(specific, disassemble_root, str(ex)))


# @Unchangeable method name
def insert_intent_filter(specific, mod_count, disassemble_root, verbose=True):
    try:
        if verbose:
            print("Insert '{}' intent-filter in the manifest file".format(specific))
        intent_mod = DroidIntentModification(disassemble_root, verbose)
        intent_mod.insert(specific, mod_count)
    except Exception as ex:
        logger.error(
            "Insert '{}' intent-filter in the manifest file of '{}': {}".format(specific, disassemble_root, str(ex)))


# @Unchangeable method name
def insert_const_string(specific, mod_count, disassemble_root, verbose=True):
    try:
        if verbose:
            print("Insert '{}' in the dalvik file".format(specific))
        str_mod = DroidStringModification(disassemble_root, verbose)
        str_mod.insert(specific, mod_count)
    except Exception as ex:
        logger.error("Insert '{}' in the dalvik file: {}".format(specific, str(ex)))


# @Unchangeable method name
def remove_const_string(specific, mod_count, disassemble_root, verbose=True):
    try:
        if verbose:
            print("Change string '{}' in the dalvik file".format(specific))
        str_mod = DroidStringModification(disassemble_root, verbose)
        str_mod.remove(specific, mod_count)
    except Exception as ex:
        logger.error("Change string '{}' in the dalvik file: {}".format(specific, str(ex)))


# @Unchangeable method name
def insert_android_api(specific, mod_count, disassemble_root, verbose=True):
    try:
        if verbose:
            print("Insert api '{}' in the dalvik file".format(specific))
        str_mod = DroidSysAPIModification(disassemble_root, verbose)
        class_name, method_name, param = specific.split(SpecDelimiter)
        str_mod.insert(class_name=class_name, method_name=method_name, params=param, is_params_smali=True,
                       mod_count=mod_count)
    except Exception as ex:
        logger.error("Insert api '{}' in the dalvik file: {}".format(specific, str(ex)))


# @Unchangeable method name
def remove_android_api(specific, mod_count, disassemble_root, verbose=True):
    try:
        mod_count = -1
        if verbose:
            print("Change api '{}' in the dalvik file".format(specific))
        str_mod = DroidSysAPIModification(disassemble_root, verbose)
        class_name, method_name, param = specific.split(SpecDelimiter)
        str_mod.remove(class_name=class_name, method_name=method_name, mod_count=mod_count)
    except Exception as ex:
        logger.error("Change api'{}' in the dalvik file: {}".format(specific, str(ex)))


def _morpher(apk_path, decomp_root_dir, meta_instrs, verbose=True):
    """
    modify the disassembled apk based on the instructions
    :param apk_path: absolute path of apks
    :param decomp_root_dir: directory for disassembly
    :param meta_instrs: instructions
    :param verbose: print information
    :return: success flag, invalid instructions
    """
    work_dir = os.path.join(decomp_root_dir, os.path.splitext(os.path.basename(apk_path))[0])
    invalid_instrs = []

    try:
        for meta_instr in meta_instrs:
            elements = meta_instr.strip().split(MetaDelimiter)
            operator = elements[0]
            comp = elements[1]
            specific = elements[2]
            count = int(elements[3])
            if comp in INSTR_ALLOWED[operator]:
                globals()[operator + '_' + comp.replace('-', '_')](specific, count, work_dir, verbose)
            else:
                logger.warn(
                    "The morpher does not support the execution of '{}' operation on '{}' component.".format(operator,
                                                                                                             comp))
                invalid_instrs.append(meta_instr)

        return True, invalid_instrs
    except Exception as ex:
        currentTime = time.time()
        logger.exception(ex)
        logger.error(str(ex))
        logger.error(
            "Modify: " + os.path.basename(decomp_root_dir) + " processing failed in " + str(currentTime) + "s...")
        return False, invalid_instrs


def _morpher_wrapper(pargs):
    return _morpher(*pargs)


def modify_disassembly(pristine_apk_paths, decomp_file_dir, meta_instrs, proc_number=6, verbose=True):
    """
    modify the apk's disassembly according to the corresponding meta_instrs
    :param pristine_apk_paths: pristine apk paths
    :param decomp_file_dir: the corresponding disassembled folder named by the apk name
    :param meta_instrs: the corresponding instructions for modification
    :param save_dir: directory for saving the adversarial apks
    :param verbose: print the details
    :return: True if this method performs thoroughly, otherwise False
    """

    pool = multiprocessing.Pool(int(proc_number))

    # unparallel
    # for i, apk_path in enumerate(pristine_apk_paths):
    #    _morpher_wrapper([apk_path, decomp_file_dir, meta_instrs[i], verbose])

    processing_results = []
    processing_result = None
    scheduled_tasks = []
    progrss_bar = progressbar_wrapper.ProgressBar()

    for i, apk_path in enumerate(pristine_apk_paths):
        scheduled_tasks.append(apk_path)
        processing_result = pool.apply_async(_morpher,
                                             args=(apk_path, decomp_file_dir, meta_instrs[i], verbose),
                                             callback=progrss_bar.CallbackForProgressBar
                                             )
        processing_results.append(processing_result)
    pool.close()
    if (processing_result):
        progrss_bar.DisplayProgressBar(processing_result, len(scheduled_tasks), type='hour')
        for i, res in enumerate(processing_results):
            if res is None:
                continue
            _flag, invalid_instructions = res.get()
            if _flag:
                print("Modifications are done")
                if len(invalid_instructions) > 0:
                    logger.warning("Not all instructons are performed.\n")
                    logger.info("\t" + '\n'.join(invalid_instructions) + '\n')
            else:
                print("Modify disassembly files failed.")
    pool.join()  # good for getting exceptions
    return True


def modify_sample(instructions, save_dir="/tmp/adv_apks", proc_number=4, vb=True):
    """
    Modify the APK based on the given instructions {apk_path:[meta_instruction1, ...], ...}
    :param instructions: a list of meta-instr (APK_path:Operator$$Comp$$Specific name$$count)
    :param save_dir:
    :return
    """

    if not isinstance(instructions, (dict, defaultdict)):
        logger.error("Incorrect instrctions.\n")
        return 1

    # step 1: data preparation
    apk_names = list(instructions.keys())  # abs path
    meta_instrs = list(instructions.values())

    # step 2: disassembly
    tmp_work_dir = os.path.join("/tmp", "apk_disassembly")
    try:
        if not os.path.exists(tmp_work_dir):
            utils.mkdir(tmp_work_dir)
        disassemble_apks(apk_names, tmp_work_dir, proc_number, verbose=vb)
    except Exception as ex:
        logger.exception(str(ex))
        logger.error("apk disassembly error: " + str(ex) + "\n")
        raise Exception("APK disassembly error: " + str(ex) + "\n")

    # step 3: modification
    modify_disassembly(apk_names, tmp_work_dir, meta_instrs, proc_number, verbose=vb)

    # step 4: assembly
    utils.mkdir(save_dir)
    try:
        new_apk_names = assemble_apks(apk_names, save_dir, tmp_work_dir, proc_number, verbose=vb)
    except Exception as ex:
        logger.exception(str(ex))
        logger.error("apk assembly error: " + str(ex) + "\n")
        raise Exception("APK assembly error: " + str(ex) + "\n")

    # step 5: sign apks
    try:
        sign_apks(new_apk_names, proc_number, vb)
    except Exception as ex:
        logger.exception(str(ex))
        logger.error("apk signing error: " + str(ex) + "\n")
        raise Exception("APK signing error: " + str(ex) + "\n")

    return 0  # execute successfully