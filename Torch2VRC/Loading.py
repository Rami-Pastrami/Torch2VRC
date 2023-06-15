import os
import glob
import platform


def LoadLatestVRCLog(VRCLogDict: str = None, isFormatCSV: bool = False, isFormatFloat: bool = False, isFormatInt: bool = False,
               logTag='RAMI_EXPORT:', convertToNP: bool = False) -> dict:

    if VRCLogDict is None:
        VRCLogDict = _GetVRCLogDirectory()

    fullPath = _GetLatestVRCLogPath(VRCLogDict)

    return LoadLogFileRaw(fullPath, logTag, isFormatCSV, isFormatFloat,isFormatInt)

def LoadLogFileRaw(LogPath: str, identifier: str = 'RAMI_EXPORT:', isCSV: bool = True, isFloat: bool = True, isInt: bool = False) -> dict:
    '''
    Responsible for loading VRC logs into a friendly format
    :param LogPath: Path to the log file
    :param identifier: Optional: The tag in the log to denote an export
    :param isCSV: Optional: if each export should be formatted as a CSV
    :param isFloat: is each line (or CSV element) a float
    :param isInt: ditto but int
    :return: dictionary where each key is a given key from the export, and the value is a list of every instance a value has been exported under that key (convert as a CSV / etc. as specified)
    '''
    seenVariableNames: list[str] = []
    c: str
    d: dict = {}
    f = open(LogPath, 'r', encoding='utf8')
    for line in f:
        c = line.strip()

        # ignore the irrelevant log lines without the identifier
        if identifier in c:
            # Isolate the variable name and the data
            _, c = c.split(identifier)
            variableName, rawData = c.split('|')

            # process and add the data
            # if the variable name is unseen before, create a new dict key
            # if it has been seen before, add it to the same key but append it to the relevant array
            if variableName not in seenVariableNames:
                seenVariableNames.append(variableName)
                d[variableName] = [_ProcessRawData(rawData, isCSV, isFloat, isInt)]
            else:
                d[variableName].append(_ProcessRawData(rawData, isCSV, isFloat, isInt))
    f.close()  # We are good netizens
    return d

def _GetVRCLogDirectory() -> str:
    # Get Log File Directory
    logsPath: str
    if platform.system() == 'Windows':
        # why isn't there a proper system var for this?
        logsPath = os.getenv('APPDATA')[0:-7] + 'LocalLow\\VrChat\\VrChat\\'
    else:
        raise Exception("Other platforms currently not supported")
    return logsPath

def _GetLatestVRCLogPath(VRCLogDir: str) -> str:

    # Get log file path
    listOfFiles = glob.glob(VRCLogDir + 'output_log*.txt')
    if len(listOfFiles) == 0:
        raise Exception("No Log Files Found!")
    # we are assuming the latest log is the first one since VRC itself names the files by date and time
    return listOfFiles[0]



def _ProcessRawData(rawData: str, isCSV: bool, isFloat: bool, isInt: bool) -> str or int or float or \
    list[str] or list[int] or list[float]:  # lol
    '''
    Processes Raw Data to a more usable form (converting to a CSV, and to numbers if enabled in the object)
    :param rawData: raw data from a single log line
    :param isCSV: if we should format the incoming data as a CSV
    :param isFloat: if we should convert the string input to a float
    :param isInt: ditto but to Int
    :return: type as defined by the object settings
    '''

    if isCSV:
        CSVData = rawData.split(', ')
        for count, element in enumerate(CSVData):
            if isFloat:
                CSVData[count] = float(element)
            if isInt:
                CSVData[count] = int(element)
        return CSVData

    if isFloat:
        rawData = float(rawData)
    if isInt:
        rawData = int(rawData)

    return rawData