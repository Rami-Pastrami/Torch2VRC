from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd

# Following Settings affect parsing of log data.
# Ensure this matches the export settings Unity side
EXPORT_IDENTIFIER: str = "!DATA!"
EXPORT_SEPERATOR: str = "|!"
HEADER_STRING: str = "STR"
HEADER_CSV: str = "CSV"
HEADER_2DCSV: str = "2D_CSV"
HEADER_JSON: str = "JSON"

class DATA_PARTITION(Enum):
    FULL: int = 0
    START: int = 1
    MIDDLE: int = 2
    END: int = 3
    DYNAMIC: int = 4

class HEADER_TYPE(Enum):
    HEADER_STRING: int = 0
    HEADER_CSV: int = 1
    HEADER_2DCSV: int = 2
    HEADER_JSON: int = 3

class RawLogLine:
    def __init__(self, line: str):
        if not RawLogLine.is_line_export_line(line):
            raise Exception("This line does not seem to be a log exported line!")
        isolated_line: str = line.split( EXPORT_IDENTIFIER)[1]
        components: list[str] = isolated_line.split(EXPORT_SEPERATOR)
        self.tag: str = components[0]
        self.partition_type: DATA_PARTITION = DATA_PARTITION(int(components[1]))
        self.data_type: HEADER_TYPE = HEADER_TYPE(int(components[2]))
        self.raw_string: str = components[3]
        self.processed_value: list = []
        match self.data_type:
            case HEADER_TYPE.HEADER_CSV:
                self.processed_value = self._process_string_as_csv(self.raw_string)
            case _:
                raise NotImplementedError

    @staticmethod
    def is_line_export_line(line: str) -> bool:
        return EXPORT_IDENTIFIER in line


    def _process_string_as_csv(self, raw_string: str) -> list:
        elements: list = raw_string.split(",")
        ## Try converting to float, then int, then leave as string
        for i in range(len(elements)):
            try:
                cache = float(elements[i])
                elements[i] = cache
                continue
            except:
                try:
                    cache = int(elements[i])
                    elements[i] = cache
                    continue
                except:
                    continue
        return elements



def read_log_file_without_further_formatting(log_file_path: Path) -> list[RawLogLine]:
    f = open(log_file_path, 'r', encoding='utf8')
    output: list[RawLogLine] = []
    for line in f:
        line_stripped: str = line.strip()
        if not RawLogLine.is_line_export_line(line_stripped):
            continue
        output.append(RawLogLine(line_stripped))
    return output

def get_tags_in_order(imported_raw_lines: list[RawLogLine]) -> list[str]:
    output: list[str] = [""] * len(imported_raw_lines)
    for i in range(len(imported_raw_lines)):
        output[i] = imported_raw_lines[i].tag
    return output

def get_formatted_arrays_in_order(imported_raw_lines: list[RawLogLine]) -> list[list]:
    output: list[str] = [[]] * len(imported_raw_lines)
    for i in range(len(imported_raw_lines)):
        output[i] = imported_raw_lines[i].processed_value
    return output

def get_raw_strings_in_order(imported_raw_lines: list[RawLogLine]) -> list[str]:
    output: list[str] = [""] * len(imported_raw_lines)
    for i in range(len(imported_raw_lines)):
        output[i] = imported_raw_lines[i].raw_string
    return output


def segregate_by_tag_to_dict(imported_raw_lines: list[RawLogLine]) -> dict[list[RawLogLine]]:
    output: dict[list[RawLogLine]] = {}
    for raw_line in imported_raw_lines:
        if raw_line.tag not in output:
            output[raw_line.tag] = []
        output[raw_line.tag].append(raw_line)
    return output

def segregate_by_tag_to_dataframe_raw_string(imported_raw_lines: list[RawLogLine], tag_name: str, data_name: str) -> pd.DataFrame:
    tags: list[str] = get_tags_in_order(imported_raw_lines)
    values_as_strings: list[str] = get_raw_strings_in_order(imported_raw_lines)
    data: list[dict] = [
        {tag_name: tag,
         data_name: values_as_string} for tag, values_as_string in zip(tags, values_as_strings)]
    return pd.DataFrame(data)

def segregate_by_tag_to_dataframe_arrays(imported_raw_lines: list[RawLogLine], tag_name: str, array_element_labels: list[str]) -> pd.DataFrame:
    tags: list[str] = get_tags_in_order(imported_raw_lines)
    values_as_lists: list[list] = get_formatted_arrays_in_order(imported_raw_lines)
    data: list[dict] = [
        {tag_name: tag,
         "data": values_as_string} for tag, values_as_string in zip(tags, values_as_lists)]

    unlabled_frame: pd.DataFrame = pd.DataFrame(data)
    unlabled_frame[array_element_labels] = pd.DataFrame(unlabled_frame["data"].tolist(), index=unlabled_frame.index)
    unlabled_frame.drop(columns="data", inplace=True)

    return unlabled_frame

