import pandas as pd
from IPython.display import display 

class StatisticsDF:
    # Function to format floats as integers if they are whole numbers, otherwise as floats
    @staticmethod
    def format_as_int_if_whole(val):
        if type(val) not in [float, int]:
            return val
        if pd.isna(val):
            return '' # Handle NaN values as empty strings for display
        if float(val).is_integer():
            return f'{int(val)}'
        else:
            return f'{val:.3f}' # Format non-whole numbers to one decimal place


    def __init__(self, dict_data: dict):
        self.df = pd.DataFrame(dict_data)
        self.df = self.df.reset_index(drop=True)
        for column in self.df:
          self.df[column] = self.df[column].apply(self.format_as_int_if_whole)

    def sort_values(self, by: list[str], ascending: bool = False):
        self.df = self.df.sort_values(by=by, ascending=ascending)
        return self

    def concat(self, *args: 'StatisticsDF'):
        self.df = pd.concat([self.df, *[arg.df for arg in args]])
        return self

    def display(self):
        display(self.df)

    def save(self, filename: str):
        self.df.to_csv(filename, sep=',', index=False)