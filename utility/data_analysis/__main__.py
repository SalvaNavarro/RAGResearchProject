from .text_reader import TextReader
from .words import Words
from .da_plotter import DAPlotter
from .semantic import SemanticGrouper
from config import *


def main():

    full_text = str(DA_TEXT / "full_text_output.txt")
    with open (full_text, "r") as file:
        text = file.read()

    text_reader = TextReader()
    words = Words(text)
    plotter = DAPlotter()
    grouper = SemanticGrouper()
    # plotter_clustered = DAPlotter()

    plotter.plot_cloud(words)
    plotter.plot_bar_chart(words)
    plotter.plot_heatmap(text_reader.df)
    semantic_df = grouper.get_groups(text_reader.df)
    filename = "heatmap_additional.png"
    plotter.plot_heatmap(semantic_df, overwrite=False, filename=filename)

    # show which word-cluster correspondence
    mapping = grouper.get_mapping()
    print(mapping)

    # show text reader data frame
    print(text_reader.df)


if __name__ == "__main__":
    main()