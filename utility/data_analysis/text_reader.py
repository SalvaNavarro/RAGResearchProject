from .words import Words
import matplotlib.pyplot as plt
import os
import pandas as pd
from config import *


class TextReader:
    """
    Reads text from .txt files.

    Attributes: 
        words (Words): Analysis object.
        labels (List[str]): Most frequent 50 words.
        df (pd.DataFrame): Table of word counts per each file.
        values (np.ndarray): Array of integer values from df.
    """

    def __init__(self):
        """
        Constructor.
        """
        self.words = Words(self.get_full_text())
        self.labels = self.words.labels[:50]
        self.df = pd.DataFrame(columns=self.labels) # the main dataframe (for text for now)
        self.get_individual_text()
        self.values = self.df.values.astype(int)


    @staticmethod
    def get_full_text() -> str:
        """
        Reads full text from file.

        Returns:
            str: Contents of full text file.
        """
        with open (str(DA_TEXT / "full_text_output.txt"), "r") as file:
            full_text = file.read()
        return full_text
        
    
    def get_individual_text(self) -> None:
        """
        Reads full text from individual files.

        Returns:
            None
        """
        for file in os.listdir(str(DA_TEXT_IND)):
            filename = str(DA_TEXT_IND / file)
            with open(filename, "r") as f:
                indv_text = f.read()
            words = Words(indv_text)

            count = []

            for label in self.labels:
                if label in words.labels:
                    idx = words.labels.index(label)
                    c = words.count[idx]
                    count.append(int(c))
                else:
                    count.append(0)
            

            w_series = pd.Series(count, index=self.labels)
            row = w_series.to_frame().T
            self.df = pd.concat([self.df, row], ignore_index=True)