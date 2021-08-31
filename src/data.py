import pandas as pd
from pathlib import Path


class DataPreparation:

    def __init__(self, file_path: str, label_file_path: str):
        self.df_text: pd.DataFrame = self.load_and_prepare_book_data(file_path)
        self.df_label: pd.DataFrame = self.load_label_data(label_file_path)

    @staticmethod
    def read_and_prepare_text(file_path: str) -> str:
        file = Path(file_path)
        text_data = file.read_text(encoding="utf-8")
        text_data = text_data.replace('&', '&amp;')
        text_data = r'<books>\n' + text_data + r'\n</books>'
        return text_data

    @staticmethod
    def get_columns(file_path: str) -> list:
        file_delimiter = '\t'

        # The max column count a line in the file could have
        largest_column_count = 0

        # Loop the data lines
        with open(file_path, 'r', encoding='utf-8') as temp_f:
            # Read the lines
            lines = temp_f.readlines()

            for l in lines:
                # Count the column count for the current line
                column_count = len(l.split(file_delimiter)) + 1

                # Set the new most column count
                largest_column_count = column_count if largest_column_count < column_count else largest_column_count
        # Generate column names
        topic_columns = [f"topic{i}" for i in range(0, largest_column_count - 1)]  #
        columns = ['isbn'] + topic_columns
        return columns

    def load_and_prepare_book_data(self, file_path: str) -> pd.DataFrame:
        text_data = self.read_and_prepare_text(file_path)
        return pd.read_xml(text_data)

    def load_label_data(self, label_file_path: str) -> pd.DataFrame:
        cols = self.get_columns(label_file_path)
        return pd.read_table(label_file_path, sep="\t", names=cols)


