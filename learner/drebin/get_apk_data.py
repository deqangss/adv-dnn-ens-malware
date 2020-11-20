from __future__ import print_function
import os
import sys
import time
import multiprocessing

import lxml.etree as etree
from xml.dom import minidom
import re
import collections

from tools import progressbar_wrapper
from androguard.misc import AnalyzeAPK, APK
from .PermAPIMapping import AxplorerMapping
from . import BasicBlockAttrBuilder

DREBIN_FEAT_INFO = {
    'S1': ['HardwareComponentsList'],
    'S2': ['RequestedPermissionList'],
    'S3': ['ActivityList', 'ServiceList', 'ContentProviderList', 'BroadcastReceiverList'],
    'S4': ['IntentFilterList'],
    'S5': ['RestrictedApiList'],
    'S6': ['UsedPermissionsList'],
    'S7': ['SuspiciousApiList'],
    'S8': ['URLDomainList']
}


def GetFromXML(ApkDirectoryPath, ApkFile):
    '''
    Get requested permission etc. for an ApkFile from Manifest files.
    :param String ApkDirectoryPath
    :param String ApkFile
    :return RequestedPermissionSet
    :rtype Set([String])
    :return ActivitySet
    :rtype Set([String])
    :return ServiceSet
    :rtype Set([String])
    :return ContentProviderSet
    :rtype Set([String])
    :return BroadcastReceiverSet
    :rtype Set([String])
    :return HardwareComponentsSet
    :rtype Set([String])
    :return IntentFilterSet
    :rtype Set([String])
    '''
    ApkDirectoryPath = os.path.abspath(ApkDirectoryPath)
    xml_tmp_dir = "/tmp/drod_xml_files"
    if not os.path.exists(xml_tmp_dir):
        os.mkdir(xml_tmp_dir)

    ApkName = os.path.splitext(os.path.basename(ApkFile))[0]

    RequestedPermissionList = []
    ActivityList = []
    ServiceList = []
    ContentProviderList = []
    BroadcastReceiverList = []
    HardwareComponentsList = []
    IntentFilterList = []
    try:
        ApkFile = os.path.abspath(ApkFile)
        a = APK(ApkFile)
        f = open(os.path.join(xml_tmp_dir, ApkName + ".xml"), "w")
        if sys.version_info.major > 2:
            xmlstring = etree.tounicode(a.xml["AndroidManifest.xml"], pretty_print=True)
        else:
            xmlstring = etree.tostring(a.xml["AndroidManifest.xml"], pretty_print=True, encoding='utf-8')
        f.write(str(xmlstring))
        f.close()
    except Exception as e:
        print(str(e))
        print("Executing Androlyze on " + ApkFile + " to get AndroidManifest.xml Failed.")
        return

    try:
        f = open(os.path.join(xml_tmp_dir, ApkName + ".xml"), "r")
        Dom = minidom.parse(f)
        DomCollection = Dom.documentElement

        DomPermission = DomCollection.getElementsByTagName("uses-permission")
        for Permission in DomPermission:
            if Permission.hasAttribute("android:name"):
                RequestedPermissionList.append(Permission.getAttribute("android:name"))

        DomActivity = DomCollection.getElementsByTagName("activity")
        for Activity in DomActivity:
            if Activity.hasAttribute("android:name"):
                ActivityList.append(Activity.getAttribute("android:name"))

        DomService = DomCollection.getElementsByTagName("service")
        for Service in DomService:
            if Service.hasAttribute("android:name"):
                ServiceList.append(Service.getAttribute("android:name"))

        DomContentProvider = DomCollection.getElementsByTagName("provider")
        for Provider in DomContentProvider:
            if Provider.hasAttribute("android:name"):
                ContentProviderList.append(Provider.getAttribute("android:name"))

        DomBroadcastReceiver = DomCollection.getElementsByTagName("receiver")
        for Receiver in DomBroadcastReceiver:
            if Receiver.hasAttribute("android:name"):
                BroadcastReceiverList.append(Receiver.getAttribute("android:name"))

        DomHardwareComponent = DomCollection.getElementsByTagName("uses-feature")
        for HardwareComponent in DomHardwareComponent:
            if HardwareComponent.hasAttribute("android:name"):
                HardwareComponentsList.append(HardwareComponent.getAttribute("android:name"))

        DomIntentFilter = DomCollection.getElementsByTagName("intent-filter")
        DomIntentFilterAction = DomCollection.getElementsByTagName("action")
        for Action in DomIntentFilterAction:
            if Action.hasAttribute("android:name"):
                IntentFilterList.append(Action.getAttribute("android:name"))

    except Exception as e:
        print(str(e))
        print("Cannot resolve " + ApkFile + "'s AndroidManifest.xml File!")
        return RequestedPermissionList, ActivityList, ServiceList, ContentProviderList, BroadcastReceiverList, HardwareComponentsList, IntentFilterList
    finally:
        f.close()
        return RequestedPermissionList, ActivityList, ServiceList, ContentProviderList, BroadcastReceiverList, HardwareComponentsList, IntentFilterList


def GetFromInstructions(ApkDirectoryPath, ApkFile, PMap, RequestedPermissionList):
    '''
    Get required permissions, used Apis and HTTP information for an ApkFile.
    Reloaded version of GetPermissions.

    :param String ApkDirectoryPath
    :param String ApkFile
    :param PScoutMapping.PScoutMapping PMap
    :param RequestedPermissionList List([String])
    :return UsedPermissions
    :rtype Set([String])
    :return RestrictedApiSet
    :rtype Set([String])
    :return SuspiciousApiSet
    :rtype Set([String])
    :return URLDomainSet
    :rtype Set([String])
    '''

    UsedPermissions = []
    RestrictedApiList = []
    SuspiciousApiList = []
    URLDomainList = []
    try:
        ApkFile = os.path.abspath(ApkFile)
        a, dd, dx = AnalyzeAPK(ApkFile)
    except Exception as e:
        print(str(e))
        print("Executing Androlyze on " + ApkFile + " Failed.")
        return

    if not isinstance(dd, list):
        dd = [dd]

    for i, d in enumerate(dd):
        for method in d.get_methods():
            g = dx.get_method(method)
            for BasicBlock in g.get_basic_blocks().get():
                Instructions = BasicBlockAttrBuilder.GetBasicBlockDalvikCode(BasicBlock)
                Apis, SuspiciousApis = BasicBlockAttrBuilder.GetInvokedAndroidApis(Instructions)
                Permissions, RestrictedApis = BasicBlockAttrBuilder.GetPermissionsAndApis(Apis, PMap,
                                                                                          RequestedPermissionList,
                                                                                          SuspiciousApiList)
                UsedPermissions.extend(Permissions)
                RestrictedApiList.extend(RestrictedApis)
                SuspiciousApiList.extend(SuspiciousApis)
                for Instruction in Instructions:
                    URLSearch = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                                          Instruction,
                                          re.IGNORECASE)

                    if URLSearch:
                        URL = URLSearch.group()
                        Domain = re.sub(r"(.*://)?([^/?]+).*", "\g<1>\g<2>", URL)
                        URLDomainList.append(Domain)

    # Got Set S6, S5, S7 described in Drebin paper

    return UsedPermissions, RestrictedApiList, SuspiciousApiList, URLDomainList


def DumpFeatures(AbsolutePath, Content):
    '''
    Export something to json file.
    Will automatic convert Set content into List.

    :param String AbsolutePath: absolute path to store the json file
    :param Variant Content: something you want to export
    '''
    try:
        if (isinstance(Content, set)):
            Content = list(Content)
        # if(isinstance(Content, collections.defaultdict)):
        #    Content = dict(Content)
        with open(AbsolutePath, "w") as f:
            # json.dump(Content, f, indent=4)
            for Key, Val in Content.items():
                for V in Val:
                    f.write(str(Key) + '_' + str(V) + '\n')

    except Exception as e:
        print("Json data writing Failed:{}.".format(str(e)))
        if "f" in dir():
            f.close()

def ProcessingDataForGetApkData(ApkDirectoryPath, ApkFile, PMap, saveDir):
    '''
    Produce .data file for a given ApkFile.

    :param String ApkDirectoryPath: absolute path of the ApkFile directory
    :param String ApkFile: absolute path of the ApkFile
    :param PMap: axplorer for API mapping

    :return Tuple(String, Boolean)  ProcessingResult: The processing result, (ApkFile, True/False)
    True means successful. False means unsuccessful.
    '''
    try:
        StartTime = time.time()
        print("Start to process " + ApkFile + "...")
        DataDictionary = {}
        RequestedPermissionList, \
        ActivityList,\
        ServiceList, \
        ContentProviderList, \
        BroadcastReceiverList, \
        HardwareComponentsList, \
        IntentFilterList = GetFromXML(ApkDirectoryPath, ApkFile)
        DataDictionary["RequestedPermissionList"] = RequestedPermissionList
        DataDictionary["ActivityList"] = ActivityList
        DataDictionary["ServiceList"] = ServiceList
        DataDictionary["ContentProviderList"] = ContentProviderList
        DataDictionary["BroadcastReceiverList"] = BroadcastReceiverList
        DataDictionary["HardwareComponentsList"] = HardwareComponentsList
        DataDictionary["IntentFilterList"] = IntentFilterList
        # Got Set S2 and others

        UsedPermissions, \
        RestrictedApiSet, \
        SuspiciousApiSet, \
        URLDomainSet = GetFromInstructions(ApkDirectoryPath,ApkFile, PMap,RequestedPermissionList)
        UsedPermissionsList = list(UsedPermissions)
        RestrictedApiList = list(RestrictedApiSet)
        SuspiciousApiList = list(SuspiciousApiSet)
        URLDomainList = list(URLDomainSet)
        DataDictionary["UsedPermissionsList"] = UsedPermissionsList
        DataDictionary["RestrictedApiList"] = RestrictedApiList
        DataDictionary["SuspiciousApiList"] = SuspiciousApiList
        DataDictionary["URLDomainList"] = URLDomainList
        # Set S6, S5, S7, S8
        name = os.path.basename(ApkFile)

        new_path = os.path.join(saveDir, name)

        DumpFeatures(os.path.splitext(new_path)[0] + ".data", DataDictionary)

    except Exception as e:
        FinalTime = time.time()
        print(ApkFile + " processing failed in " + str(FinalTime - StartTime) + "s...")
        return ApkFile, False
    else:
        FinalTime = time.time()
        print(ApkFile + " processed successfully in " + str(FinalTime - StartTime) + "s")
        return ApkFile, True


def GetApkData(ApkFileList, saveDir, ProcessNumber=4):
    '''
    Get Apk data dictionary for all Apk files under ApkDirectoryPath and store them in ApkDirectoryPath
    Used for next step's classification

    :param list ApkDirectoryPaths: absolute path of the directories contained Apk files
    '''
    if len(ApkFileList) <= 0:
        return

    ''' Change current working directory to import the mapping '''
    PMap = AxplorerMapping()
    pool = multiprocessing.Pool(int(ProcessNumber))
    ProcessingResults = []
    ScheduledTasks = []
    ProgressBar = progressbar_wrapper.ProgressBar()
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    for i, ApkFile in enumerate(ApkFileList):

        if not os.path.exists(os.path.join(saveDir, os.path.splitext(os.path.basename(ApkFile))[0] + ".data")):
            # ProcessingDataForGetApkData(ApkDirectoryPath, ApkFile, PMap)
            ApkDirectoryPath = os.path.split(ApkFile)[0]
            ScheduledTasks.append(ApkFile)
            ProcessingResults = pool.apply_async(ProcessingDataForGetApkData,
                                                 args=(ApkDirectoryPath, ApkFile, PMap, saveDir),
                                                 callback=ProgressBar.CallbackForProgressBar)

    pool.close()
    if (ProcessingResults):
        ProgressBar.DisplayProgressBar(ProcessingResults, len(ScheduledTasks), type="hour")
    pool.join()

    return


def load_features(data_dir, order_seqence=None):
    feature_list = []
    apk_name_list = []

    if not os.path.isdir(data_dir):
        return feature_list, apk_name_list

    file_names = os.listdir(data_dir)
    if order_seqence is None:
        for fn in file_names:
            if '.data' in fn:
                with open(os.path.join(data_dir, fn), 'r') as rh:
                    features = rh.read().splitlines()
                    feature_list.append(features)
                    apk_name_list.append(fn)
        return feature_list, apk_name_list
    else:
        assert len(file_names) <= len(order_seqence)

        for follewed_name in order_seqence:
            for fn in file_names:
                if os.path.splitext(fn)[0] in follewed_name and '.data' in fn:
                    with open(os.path.join(data_dir, fn), 'r') as rh:
                        features = rh.read().splitlines()
                        feature_list.append(features)
                        apk_name_list.append(fn)
        return feature_list, apk_name_list


def get_incap_instances(apk_name_list, order_sequence):
    _tmp_name_list = []
    apk_names = []
    assert len(apk_name_list) >= 0
    assert len(order_sequence) >= 0
    for apk_name in apk_name_list:
        _name = os.path.splitext(os.path.basename(apk_name))[0]
        apk_names.append(_name)
    for f_name in order_sequence:
        _name = os.path.splitext(os.path.basename(f_name))[0]
        if _name not in apk_names:
            _tmp_name_list.append(_name)
    return '\n'.join(_tmp_name_list)


def remove_interdependent_features(raw_features):
    if len(raw_features) == 0:
        return []

    rtn_features = []
    for features in raw_features:
        for s in DREBIN_FEAT_INFO['S6']:
            feat = list(filter(lambda e: e.split('_', 1)[0] != s, features))
            rtn_features.append(feat)
    return rtn_features


def get_vocab(drein_feature_set):
    c = collections.Counter()
    d = collections.defaultdict(set)
    clean_feature_set = []
    for features in drein_feature_set:
        clean_feature = []
        for feat in features:
            # here is dirty code. The symbol '::' is used
            # because the code line 111 and code line 116 in python file of BasicBlockAttrBuilder.py
            elements = feat.strip().split("::")
            if len(elements) == 0:
                raise ValueError("Null feature.")
            elif len(elements) == 1:
                c[elements[0]] = c[elements[0]] + 1
                d[elements[0]].add(elements[0].split('_', 1)[1]) # '_' is used because of line 208 of this file
                clean_feature.append(elements[0])
            elif len(elements) == 2:
                c[elements[0]] = c[elements[0]] + 1
                d[elements[0]].add(elements[1])
                clean_feature.append(elements[0])
            else:
                raise ValueError("Unexpected feature '{}'".format(feat))
        clean_feature_set.append(clean_feature)
    vocab, counter = zip(*c.items())
    return list(vocab), d, clean_feature_set


def get_word_category(vocabulary, vocabulary_info, defined_comp):
    """
    Get the category for each word in vocabulary, based on the COMP in conf file
    :rtype: object
    """

    def _api_check(dalvik_code_line_list):
        for code_line in dalvik_code_line_list:
            invoke_match = re.search(
                r'(?P<invokeType>invoke\-([^ ]*?)) (?P<invokeParam>([vp0-9,. ]*?)), (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$',
                code_line)
            if invoke_match is None:
                return defined_comp['Notdefined']
            if invoke_match.group('invokeType') == 'invoke-virtual' or invoke_match.group(
                    'invokeType') == 'invoke-virtual/range' or \
                    invoke_match.group('invokeType') == 'invoke-static' or \
                    invoke_match.group('invokeType') == 'invoke-static/range':
                if invoke_match.group('invokeObject').startswith('Landroid'):
                    return defined_comp['Android_API']
                elif invoke_match.group('invokeObject').startswith('Ljava'):
                    return defined_comp['Java_API']
                else:
                    return defined_comp['Notdefined']
            else:
                return defined_comp['Notdefined']

    word_cat_dict = collections.defaultdict()
    for w in vocabulary:
        if 'ActivityList_' in w:
            word_cat_dict[w] = defined_comp['Activity']
        elif 'RequestedPermissionList_' in w:
            word_cat_dict[w] = defined_comp['Permission']
        elif 'ServiceList_' in w:
            word_cat_dict[w] = defined_comp['Service']
        elif 'ContentProviderList_' in w:
            word_cat_dict[w] = defined_comp['Provider']
        elif 'BroadcastReceiverList_' in w:
            word_cat_dict[w] = defined_comp['Receiver']
        elif 'HardwareComponentsList_' in w:
            word_cat_dict[w] = defined_comp['Hardware']
        elif 'IntentFilterList_' in w:
            word_cat_dict[w] = defined_comp['Intentfilter']
        elif 'UsedPermissionsList_' in w:
            word_cat_dict[w] = defined_comp['Notdefined']
        elif 'RestrictedApiList_' in w:
            word_cat_dict[w] = _api_check(vocabulary_info[w])
        elif 'SuspiciousApiList' in w:
            word_cat_dict[w] = _api_check(vocabulary_info[w])
        elif 'URLDomainList' in w:
            word_cat_dict[w] = defined_comp['User_String']
        else:
            word_cat_dict[w] = defined_comp['Notdefined']
    return word_cat_dict


def preprocess_feature(drein_feature_set):
    clean_feature_set = []
    for features in drein_feature_set:
        clean_feature = []
        for feat in features:
            elements = feat.strip().split("::")
            if len(elements) == 0:
                raise ValueError("Null feature.")
            elif len(elements) == 1:
                clean_feature.append(elements[0])
            elif len(elements) == 2:
                clean_feature.append(elements[0])
            else:
                raise ValueError("Unexpected feature '{}'".format(feat))
        clean_feature_set.append(clean_feature)
    return clean_feature_set


def get_api_ingredient(api_dalvik_code):
    """get class name, method name, parameters from dalvik code line by line"""
    invoke_match = re.search(
        r'(?P<invokeType>invoke\-([^ ]*?)) (?P<invokeParam>([vp0-9,. ]*?)), (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$',
        api_dalvik_code)

    if invoke_match is None:
        return None, None, None
    else:
        invoke_object = invoke_match.group('invokeObject')
        invoke_method = invoke_match.group('invokeMethod')
        invoke_argument = invoke_match.group('invokeArgument')
        return invoke_object, invoke_method, invoke_argument
