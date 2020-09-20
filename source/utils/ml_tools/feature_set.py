from __future__ import annotations

from typing import List
from src.utils.reader_writers.reader_writers import write_object


class FeatureTypes:
    numerical = 'numerical'
    categorical = 'categorical'
    one_hot_encoded = 'one_hot_encoded'


class FeatureMetadata:

    def __init__(self, feature_name: str, category: str = None, type=None, path: str = None, comments: str = None):
        '''
        A classification of features used to forming the base of a feature library

        :param feature_name: The name of the feature
        :param category: User defined category for the feature
        :param type: The type of the feature, either: numerical, one-hot-encoded, categorical
        '''
        self.feature_name = feature_name
        self.category = category
        self.type = type
        self.path = path
        self.comments = comments

    def __str__(self):
        return f"{self.feature_name} -- {self.category} -- {self.type}"


class FeatureInformationSet:

    def __init__(self, feature_information_set: List[FeatureMetadata]):
        self.feature_information_set = feature_information_set

    def get_feature_by_type(self, feature_types: List[str]) -> FeatureInformationSet:
        return type(self)([
            feature
            for feature
            in self.feature_information_set
            if feature.type in feature_types
        ])

    def get_feature_by_category(self, categories: List[str]) -> FeatureInformationSet:
        return type(self)([
            feature
            for feature
            in self.feature_information_set
            if feature.category in categories
        ])

    def store(self, path):
        write_object(
            path=path,
            py_object=self.feature_information_set
        )


fi = FeatureMetadata(
    feature_name='hello',
    category='bad',
    type=FeatureTypes.numerical
)

str(fi)
print(fi)
