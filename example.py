from pathlib import Path
import pandas as pd
import numpy as np
from  sklearn.preprocessing import LabelEncoder

from VRCDataImporter.LogFileImporters import RawLogLine, read_log_file_without_further_formatting, segregate_by_tag_to_dataframe_arrays


# This just gets the example file path
current_path: Path = Path(__file__)
current_folder: Path = current_path.parent
file_name: Path = Path("example_log.txt")
example_log_file_path: Path = current_folder.joinpath(file_name)

# load in the log file into a Pandas DataFrame
log_line_objects: list[RawLogLine] = read_log_file_without_further_formatting(example_log_file_path)
data_frame: pd.DataFrame = segregate_by_tag_to_dataframe_arrays(log_line_objects, "color", ["X", "Y", "Z"] )

# Get training / answer sets from the DataFrame
label_encoder: LabelEncoder = LabelEncoder()
data_frame["color_category"] = label_encoder.fit_transform(data_frame["color"])
XYZ_training_data: np.ndarray = data_frame[["X", "Y", "Z"]].to_numpy()
color_answer_data: np.ndarray = data_frame["color_category"].to_numpy()



print("Program End") # easy breakpoint