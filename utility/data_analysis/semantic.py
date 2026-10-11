from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

class SemanticGrouper:
    """
    clusters an initial list of words into a clustered semantic set of words.

    Attributes:
        clusters (int): The number of clusters for the words to be grouped in.
    """
    def __init__(self, clusters: int=8) -> None:
        """
        Constructor.
        """
        self.clusters = clusters
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.kmeans = None
        self.word_to_cluster = {}
        self.cluster_labels = []
        self.cluster_labels_renamed = [] # the list of manually renamed cluster labels


    def _embed(self, words: list[str]) -> list[str]:
        """
        Creates embeddings from a list of words.
        
        Args:
            words: List of words.

        Returns:
            list[str]
        """
        return self.model.encode(words)
    

    def _fit(self, words: list[str]) -> None:
        """
        Creates clusters.

        Args:
            words: List of original words.

        Returns:
            None
        """
        embeddings = self._embed(words)
        km = KMeans(n_clusters=self.clusters, random_state=1)
        cluster_indices = km.fit_predict(embeddings)

        self.kmeans = km
        self.word_to_cluster = dict(zip(words, cluster_indices))
        for i in range(self.clusters):
            self.cluster_labels.append(f"Cluster_{i+1}")


    def _transform(self, df: DataFrame) -> DataFrame:
        """
        Creates clusters.

        Args:
            df: A fitted dataframe.

        Returns:
            DataFrame
        """
        grouped_df = DataFrame(0, index=df.index, columns=self.cluster_labels)
        for word, cluster_index in self.word_to_cluster.items():
            g = self.cluster_labels[cluster_index]
            grouped_df[g] = grouped_df[g].add(df[word])

        return grouped_df
    

    def get_groups(self, df: DataFrame) -> DataFrame:
        """
        Create groups.

        Args:
            df: Dataframe.

        Returns:
            DataFrame
        """
        self._fit(df.columns)
        t_df = self._transform(df)
        self.rename_labels()
        t_df.columns = self.labels
        return t_df
    

    def get_mapping(self) -> dict[str,str]:
        """
        Shows dictionary with words and cluster assignation.

        Returns:
            dict
        """
        mapping = {}
        for word, cluster_index in self.word_to_cluster.items():
            label = self.cluster_labels[cluster_index]
            mapping[word] = label
        return mapping
    
    def rename_labels(self) -> None:
        new_labels = [
            "Country info",
            "Corporate terms",
            "Location description",
            "Planning terms",
            "Urban terms",
            "Miscellaneous",
            "Architectural terms",
            "Environments"
        ]
        self.labels = new_labels