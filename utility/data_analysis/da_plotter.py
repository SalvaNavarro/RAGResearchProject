from .words import Words
from .text_reader import TextReader
from pandas import DataFrame
import matplotlib.pyplot as plt
import random
from config import *


class DAPlotter():
    """
    Plots the data.

    Attributes:
        wordcloud_out (str): Output path for word cloud chart.
        wordcount_bar_out (str): Output path for word count bar chart.
        heatmap_out (str): Output path for heat map chart.
    """
    
    def __init__(self) -> None:
        """
        Constructor.
        """
        self.wordcloud_out = str(DA_OUTPUT / "wordcloud.png")
        self.wordcount_bar_out = str(DA_OUTPUT / "wordcount_bar.png")
        self.heatmap_out = str(DA_OUTPUT / "heatmap.png")


    @staticmethod
    def grey_color_func(word, font_size, position, orientation, random_state=None,
                        **kwargs) -> str:
        """
        Sets colour scheme for heat map chart.

        Returns:
            str: The colour scheme.
        """
        return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)


    def plot_cloud(self, words: Words) -> None:
        """
        Saves the word cloud chart to specified location.

        Parameters:
            words (Words): A Words object.
        
        Returns:
            None
        """
        cloud = words.cloud
        plt.imshow(cloud.recolor(color_func=self.grey_color_func, random_state=3), 
                   interpolation='bilinear')
        cloud.to_file(self.wordcloud_out)
        plt.axis("off")
        # plt.show()
        print(f"Figure saved at {self.wordcloud_out}")


    def plot_bar_chart(self, words: Words) -> None:
        """
        Saves the word count bar chart to specified location.

        Parameters:
            words (Words): A Words object.

        Returns:
            None
        """    
        plt.figure(figsize=(12,8))
        plt.bar(words.labels[:50], words.count[:50], color='black')
        plt.title("Keyword distribution")
        plt.xlabel("Word")
        plt.ylabel("Count")
        plt.xticks(rotation=90, ha="right")
        plt.tight_layout()
        # plt.show()

        plt.savefig(self.wordcount_bar_out)
        print(f"Figure saved at {self.wordcount_bar_out}")


    def plot_heatmap(self, df: DataFrame, overwrite=True, filename="heatmap_additional.png") -> None:
        """
        Saves the heat map chart to specified location.

        Parameters:
            df (DataFrame): A DataFrame object.
            overwrite (bool): Sets the switch between original heatmap and clustered version.
            filename (str): The file name for the clustered version.

        Returns:
            None
        """    
        data = df.values.astype(int)
        plt.figure(figsize=(8, 8))
        plt.imshow(data, aspect='auto')
        plt.colorbar()
        plt.xticks(range(len(df.columns)), df.columns, rotation=90)
        plt.xlabel("Word")
        plt.yticks(range(len(df.index)), df.index)
        plt.ylabel("File index")
        plt.tight_layout()
        if overwrite:
            plt.savefig(self.heatmap_out)
            print(f"Figure saved at {self.heatmap_out}")
        else:
            f_name = str(DA_OUTPUT / f"{filename}")
            plt.savefig(f_name)
            print(f"Figure saved at {f_name}")
        plt.show()