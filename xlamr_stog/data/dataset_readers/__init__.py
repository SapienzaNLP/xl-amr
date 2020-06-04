"""
A :class:`~xl-amr.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~xl-amr.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""

# pylint: disable=line-too-long
from xlamr_stog.data.dataset_readers.dataset_reader import DatasetReader
from xlamr_stog.data.dataset_readers.abstract_meaning_representation import AbstractMeaningRepresentationDatasetReader
