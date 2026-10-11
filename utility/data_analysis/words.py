from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re
from config import *


class Words:
    """
    Creates the bag of words, sorted by frequency.

    Attributes:
        stopwords (set[str]): Generic words to be ignored.
        additional_stopwords (set[str]): Additional words to be ignored.
        text (str): The text to get the bag of words from.
        cloud (WordCloud): The generated wordcloud.
        labels (list[str]): The words as list.
        count (list[int]): The word count per word.
    """

    def __init__(self, text) -> None:
        """
        Constructor.
        """
        self.stopwords = STOPWORDS
        self.additional_stopwords = {
            "Foster",
            "Partners",
            "will",
            "building",
            "design",
            "studio",
            "designed",
            "said",
            "Dezeen",
            "project",
            "British",
            "unveiled",
            "completed",
            "new",
            "made",
            "according",
            "year",
            "designs",
            "the",
            "+",
            "s",
            "skyscrapers",
            "set",
            "first",
            "two",
            "around",
            "contain",
            "created",
            "added",
            "include",
            "create",
            "located",
            "part",
            "world",
            "team",
            "every",
            "000",
            "week"
        }
        self.text = text
        self.add_stopwords()
        self.cloud = self.generate_cloud()

        self.labels = []
        self.count = []
        self.count_words()


    def add_stopwords(self) -> None:
        """
        Adds additional custom stopwords.

        Returns:
            None
        """
        for word in self.additional_stopwords:
            self.stopwords.add(word)
            self.stopwords.add(word.lower())


    def count_words(self) -> None:
        """
        Counts the occurrences of each word.

        Returns:
            None
        """
        tokens = re.findall(r'\w+', self.text.lower())
        filtered = []

        for token in tokens:
            if token not in self.stopwords:
                filtered.append(token)
        frequencies = Counter(filtered)
        shortlist = frequencies.most_common(500)

        for pair in shortlist:
            self.labels.append(pair[0])
            self.count.append(pair[1])


    def generate_cloud(self) -> WordCloud:
        """
        Generates the WordCloud.

        Returns:
            WordCloud
        """
        init_cloud = WordCloud(stopwords=self.stopwords)
        cloud = init_cloud.generate(self.text)
        return cloud