from pathlib import Path
from itertools import chain
from typing import Any, Literal

import pandas as pd


OutputType = Literal['series', 'str', 'stdout', 'print']


class KeyExtractor:
    """
    Base accessor to extract keys from a pandas object.
    """

    def __init__(self, pandas_obj: pd.DataFrame | pd.Series) -> None:
        self._obj = pandas_obj

    def __call__(
        self,
        key: str|None = None,
        *,
        unique: bool = True,
        sample: int|None = None,
        groupby: list[str]|str|None = None,
        batch_size: int|None = None,
        batch_name: str = 'batch',
        to: OutputType = 'str',
        sep: str = ';',
    ) -> pd.Series|str|None:
        """
        Process and return keys in the specified format.

        Parameters
        ----------
        key : str | None, optional
            Column name to extract (not needed for Series)
        unique : bool, default True
            If True, ensures keys are unique within groups
        sample : int | None, optional
            Number of random samples to take
        groupby : list[str] | str | None, optional
            Columns to group by
        batch_size : int | None, optional
            Size of batches to create
        batch_name : str, default 'batch'
            Name for batch groups
        to : {'series', 'str', 'stdout', 'print'}, default 'str'
            Output format
        sep : str, default ';'
            Separator for string output

        Returns
        -------
        pd.Series | str | None
            Processed data in requested format. None if output is to stdout/print.

        Raises
        ------
        ValueError
            If output type is invalid
        """
        if to not in ('series', 'str', 'stdout', 'print'):
            raise ValueError("Output type must be one of: 'series', 'str', 'stdout', 'print'")

        if isinstance(self, KeyExtractorSeries):
            processed_data = self._preprocess(
                unique=unique,
                sample=sample,
                groupby=groupby,
                batch_size=batch_size,
                batch_name=batch_name,
            )
        else:
            processed_data = self._preprocess(
                key=key,
                unique=unique,
                sample=sample,
                groupby=groupby,
                batch_size=batch_size,
                batch_name=batch_name,
            )

        return self._output_data(processed_data, to, sep, groupby, batch_name, batch_size)

    def _output_data(
        self,
        data: pd.Series,
        output_type: OutputType,
        sep: str,
        groupby: list[str]|str|None,
        batch_name: str,
        batch_size: int|None,
    ) -> pd.Series|str|None:
        """
        Route data to appropriate output format.

        Parameters
        ----------
        data : pd.Series
            Processed data to output
        output_type : {'series', 'str', 'stdout', 'print'}
            Desired output format
        sep : str
            Separator for string output
        groupby : list[str] | str | None
            Grouping columns
        batch_name : str
            Name for batch groups
        batch_size : int | None
            Size of batches

        Returns
        -------
        pd.Series | str | None
            Data in requested format
        """
        collected_groups = self._collect_groups(
            self._get_grouper(groupby),
            self._get_grouper(batch_name if batch_size else None)
        )

        if output_type == 'series':
            return data
        elif output_type == 'str':
            return self._stringify(data, collected_groups, sep=sep)
        elif output_type in ('stdout', 'print'):
            output = self._stringify(data, collected_groups, sep=sep)
            print(output)
            return None

    def _collect_groups(self, *args: list[str]) -> list[str]:
        """
        Combine all grouping arguments into a single list.

        Parameters
        ----------
        *args : list[str]
            Lists of group names to combine

        Returns
        -------
        list[str]
            Combined list of group names
        """
        return list(chain.from_iterable(args))

    def _get_grouper(self, groupby: Any) -> list[str]:
        """
        Convert groupby parameter to list format.

        Parameters
        ----------
        groupby : Any
            Grouping specification

        Returns
        -------
        list[str]
            List of group names
        """
        if groupby is None:
            return []
        if pd.api.types.is_scalar(groupby):
            return [groupby]
        return list(groupby)

    def _preprocess(
        self,
        key: str|None = None,
        unique: bool = True,
        sample: int|None = None,
        groupby: list[str]|str|None = None,
        batch_size: int|None = None,
        batch_name: str = 'batch',
    ) -> pd.Series:
        """
        Preprocess the data according to parameters.

        Parameters
        ----------
        key : str | None, optional
            Column name to extract
        unique : bool, default True
            If True, ensures keys are unique within groups
        sample : int | None, optional
            Number of random samples to take
        groupby : list[str] | str | None, optional
            Columns to group by
        batch_size : int | None, optional
            Size of batches to create
        batch_name : str, default 'batch'
            Name for batch groups

        Returns
        -------
        pd.Series
            Processed data
        """
        cols = [*self._get_grouper(groupby)]
        if hasattr(self, 'key'):
            cols.append(self.key)
        elif key:
            cols.append(key)

        data = self._obj.reset_index().filter(cols)

        if unique:
            data = data.drop_duplicates(subset=cols)

        if groupby:
            data = data.set_index(self._get_grouper(groupby), append=True)

        if batch_size:
            data = self._add_batches(data, batch_size, batch_name, groupby)

        if sample:
            data = data.sample(n=sample)

        return data.squeeze()

    def _add_batches(
        self,
        data: pd.Series,
        batch_size: int,
        batch_name: str,
        groupby: list[str]|str|None,
    ) -> pd.Series:
        """
        Add batch numbers to the index.

        Parameters
        ----------
        data : pd.Series
            Data to batch
        batch_size : int
            Size of each batch
        batch_name : str
            Name for batch groups
        groupby : list[str] | str | None
            Grouping columns

        Returns
        -------
        pd.Series
            Data with batch numbers in index
        """
        def get_batch_numbers(n: int) -> pd.RangeIndex:
            return (pd.RangeIndex(n) // batch_size + 1).rename(batch_name)

        if groupby:
            return data.groupby(groupby, group_keys=False).apply(
                lambda x: x.set_index(get_batch_numbers(len(x)), append=True)
            )
        return data.set_index(get_batch_numbers(len(data)), append=True)

    def _stringify(
        self,
        s: pd.Series,
        groupby: list[str]|None = None,
        sep: str = ';'
    ) -> str:
        """
        Convert series to string representation.

        Parameters
        ----------
        s : pd.Series
            Series to convert
        groupby : list[str] | None, optional
            Grouping columns
        sep : str, default ';'
            Separator for values

        Returns
        -------
        str
            String representation of data
        """
        def format_group(group_series: pd.Series, group_name: str = '') -> str:
            return (
                f"[{group_name}] ({len(group_series)})\n"
                f"{sep.join(map(str, group_series))}\n\n"
            )

        if not groupby:
            return sep.join(map(str, s))

        output = []
        for name, group in s.groupby(groupby):
            if isinstance(name, tuple):
                group_name = ' | '.join(f"{level}: {val}" for level, val in zip(groupby, name))
            else:
                group_name = f"{groupby[0]}: {name}"
            output.append(format_group(group, group_name))

        return ''.join(output)

    def to_file(
        self,
        path: Path|str,
        key: str|None = None,
        *,
        unique: bool = True,
        sample: int|None = None,
        groupby: list[str]|str|None = None,
        batch_size: int|None = None,
        batch_name: str = 'batch',
    ) -> None:
        """
        Save processed data to file(s).

        Parameters
        ----------
        path : Path | str
            Directory path for output files
        key : str | None, optional
            Column name to extract (not needed for Series)
        unique : bool, default True
            If True, ensures keys are unique within groups
        sample : int | None, optional
            Number of random samples to take
        groupby : list[str] | str | None, optional
            Columns to group by
        batch_size : int | None, optional
            Size of batches to create
        batch_name : str, default 'batch'
            Name for batch groups
        """
        path = Path(path)
        processed_data = self._preprocess(
            key=key,
            unique=unique,
            sample=sample,
            groupby=groupby,
            batch_size=batch_size,
            batch_name=batch_name,
        )

        collected_groups = self._collect_groups(
            self._get_grouper(groupby),
            self._get_grouper(batch_name if batch_size else None)
        )
        ymd = pd.Timestamp.today().strftime('%Y%m%d')
        if collected_groups:
            for group, data in processed_data.groupby(collected_groups):
                key_parts = group if isinstance(group, tuple) else [group]
                key_str = '_'.join(map(str, key_parts))
                filename = f"{ymd}.{key_str}.{len(data)}.txt"
                data.to_csv(path / filename, index=False)
        else:
            current_key = getattr(self, 'key', key)
            filename = f"{ymd}.{current_key}.{len(processed_data)}.txt"
            processed_data.to_csv(path / filename, index=False)


@pd.api.extensions.register_dataframe_accessor("askeys")
class KeyExtractorDataFrame(KeyExtractor):
    """DataFrame-specific key extractor."""
    pass


@pd.api.extensions.register_series_accessor("askeys")
class KeyExtractorSeries(KeyExtractor):
    """Series-specific key extractor."""
    @property
    def key(self) -> str:
        """
        Series name used as key.

        Returns
        -------
        str
            Name of the series
        """
        return self._obj.name
