import time

def GetBasicBlockDalvikCode(BasicBlock):
    '''
    Get the list of dalvik code of the instrcutions contained in the BasicBlock
    
    :param DVMBasicBlock BasicBlock
    :return DalvikCodeList
    :rtype List<String>
    '''

    DalvikCodeList = []
    for Instruction in BasicBlock.get_instructions():
        CodeLine = Instruction.get_name() + " " + Instruction.get_output()
        DalvikCodeList.append(CodeLine)
    return DalvikCodeList


def GetInvokedAndroidApis(DalvikCodeList):
    '''
    Get the android APIs invoked by a list of instrcutions and return these APIs and Suspicious API set.
    :param List<String> DalvikCodeList
    :return ApiList
    :rtype List
    :return SuspiciousApiSet
    :rtype Set([String])
    '''

    ApiList = []
    SuspiciousApiSet = []
    AndroidSuspiciousApiNameList = ["getExternalStorageDirectory", "getSimCountryIso", "execHttpRequest", 
                "sendTextMessage", "getPackageInfo", "getSystemService",
                 "setWifiDisabled", "Cipher"]
    OtherSuspiciousApiNameList = ["Ljava/net/HttpURLconnection;->setRequestMethod(Ljava/lang/String;)", "Landroid/telephony/SmsMessage;->getMessageBody",
                                  "Ljava/io/IOException;->printStackTrace", "Ljava/lang/Runtime;->exec"]
    NotLikeApiNameList = ["system/bin/su"]
    for DalvikCode in DalvikCodeList:
        if "invoke-" in DalvikCode:
            Parts = DalvikCode.split(",")
            for Part in Parts:
                if ";->" in Part:
                    Part = Part.strip()
                    if Part.startswith('Landroid'):
                        FullApi = Part
                        ApiParts = FullApi.split(";->")
                        ApiClass = ApiParts[0].strip()
                        ApiName = ApiParts[1].split("(")[0].strip()
                        ApiDetails = {}
                        ApiDetails['FullApi'] = DalvikCode.strip()
                        ApiDetails['ApiClass'] = ApiClass
                        ApiDetails['ApiName'] = ApiName
                        ApiList.append(ApiDetails)
                        if(ApiName in AndroidSuspiciousApiNameList):
                            #ApiClass = Api['ApiClass'].replace("/", ".").replace("Landroid", "android").strip()
                            SuspiciousApiSet.append(ApiClass+"."+ApiName + '::' + DalvikCode.strip())
                for Element in OtherSuspiciousApiNameList:
                    if(Element in Part):
                        SuspiciousApiSet.append(Element + '::' + DalvikCode.strip())
        for Element in NotLikeApiNameList:
            if Element in DalvikCode:
                SuspiciousApiSet.append(Element + '::' + DalvikCode.strip())
    return ApiList, SuspiciousApiSet

def GetPermissions(ApiList, PMap):
    '''
    Get Android Permissions used by a list of android APIs

    :param List ApiList
    :param PScoutMapping.PScoutMapping PMap
    :return PermissionSet
    :rtype Set<String>
    '''

    PermissionSet = set()
    for Api in ApiList:
        ApiClass = Api['ApiClass'].replace("/", ".").replace("Landroid", "android").strip()
        Permission = PMap.GetPermFromApi(ApiClass, Api['ApiName'])
        if(not Permission == None):
            PermissionSet.add(Permission)

    return PermissionSet


def GetPermissionsAndApis(ApiList, PMap, RequestedPermissionList, Suspicious_api_list = None):
    '''
    Get Android Permissions used by a list of android APIs
    and meanwhile Get RestrictedApiSet and SuspiciousApiSet

    :param List ApiList
    :param PScoutMapping.PScoutMapping PMap
    :param RequestedPermissionList List([String])
    :return PermissionSet
    :rtype Set<String>
    :return RestrictedApiSet
    :rtype Set([String])
    '''

    PermissionList = []
    RestrictedApiList=[]
    if Suspicious_api_list is None:
        Suspicious_api_list = []
    #SuspiciousApiSet=set()
    for Api in ApiList:
        ApiClass = Api['ApiClass'].replace("/", ".").replace("Landroid", "android").strip()
        Permission = PMap.GetPermFromApi(ApiClass, Api['ApiName'])
        if Permission is not None:
            #if Api['ApiName'] in ["getDeviceId", "getSubscriberId", "setWifiEnabled", "execHttpRequest", "sendTextMessage"]:
            #    SuspiciousApiSet.add(ApiClass+"."+Api["ApiName"])
            if(Permission in RequestedPermissionList):
                PermissionList.append(Permission)
                api_info = ApiClass + "." + Api["ApiName"] + '::' + Api['FullApi']
                if api_info not in Suspicious_api_list:
                    RestrictedApiList.append(api_info)
            else:
                api_info = ApiClass + "." + Api["ApiName"] + '::' + Api['FullApi']
                if api_info not in Suspicious_api_list:
                    RestrictedApiList.append(api_info)
    return PermissionList,RestrictedApiList#,SuspiciousApiSet

def GetSusiSrcsSinks(ApiList, SusiMap):
    '''
    Get sources and sinks used in a list of android APIs

    :param List ApiList
    :param Susi.SusiDictMaker SusiMap
    :return SourceSet: Set of SUSI src
    :rtype Set<String>
    :return SinkSet: Set of SUSI sink
    :rtype Set<String>
    '''

    SourceSet = set()
    SinkSet = set()

    for Api in ApiList:
        ApiClass = Api['ApiClass'].replace("/", ".").replace("Landroid", "android").strip()
        Source = SusiMap.GetSusiCategoryFromApi(ApiClass, Api['ApiName'], "src")
        if(not Source == -1):
            SourceSet.add(Source)
        Sink = SusiMap.GetSusiCategoryFromApi(ApiClass, Api['ApiName'], "sink")
        if(not Sink == -1):
            SinkSet.add(Sink)

    return SourceSet, SinkSet
