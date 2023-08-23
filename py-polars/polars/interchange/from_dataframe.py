from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars._reexport as pl
import polars.functions as F
from polars.datatypes import Int64
from polars.interchange.dataframe import PolarsDataFrame
from polars.interchange.protocol import ColumnNullType, DtypeKind
from polars.interchange.utils import (
    dtype_to_polars_dtype,
    dtype_to_polars_physical_dtype,
)
from polars.polars import PySeries
from polars.utils._wrap import wrap_s

if TYPE_CHECKING:
    from polars import DataFrame, Series
    from polars.interchange.protocol import Buffer, Column, Dtype
    from polars.interchange.protocol import DataFrame as InterchangeDataFrame


def from_dataframe(df: Any, *, allow_copy: bool = True) -> DataFrame:
    """
    Build a Polars DataFrame from any dataframe supporting the interchange protocol.

    Parameters
    ----------
    df
        Object supporting the dataframe interchange protocol, i.e. must have implemented
        the ``__dataframe__`` method.
    allow_copy
        Allow memory to be copied to perform the conversion. If set to False, causes
        conversions that are not zero-copy to fail.

    Notes
    -----
    Details on the Python dataframe interchange protocol:
    https://data-apis.org/dataframe-protocol/latest/index.html

    Using a dedicated function like :func:`from_pandas` or :func:`from_arrow` is a more
    efficient method of conversion.

    Examples
    --------
    Convert a pandas dataframe to Polars through the interchange protocol.

    >>> import pandas as pd
    >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["x", "y"]})
    >>> dfi = df_pd.__dataframe__()
    >>> pl.from_dataframe(dfi)
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ a   ┆ b   ┆ c   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ f64 ┆ str │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3.0 ┆ x   │
    │ 2   ┆ 4.0 ┆ y   │
    └─────┴─────┴─────┘

    """
    if isinstance(df, pl.DataFrame):
        return df
    elif isinstance(df, PolarsDataFrame):
        return df._df

    if not hasattr(df, "__dataframe__"):
        raise TypeError(
            f"`df` of type {type(df).__name__!r} does not support the dataframe interchange protocol"
        )

    return _from_dataframe(df.__dataframe__(allow_copy=allow_copy))


def _from_dataframe(df: InterchangeDataFrame, *, allow_copy: bool = True) -> DataFrame:
    chunks = [
        _protocol_df_chunk_to_polars(chunk, allow_copy=allow_copy)
        for chunk in df.get_chunks()
    ]
    # If copy is allowed, rechunk as it will speed up subsequent computation
    return F.concat(chunks, rechunk=allow_copy)


def _protocol_df_chunk_to_polars(
    df: InterchangeDataFrame, *, allow_copy: bool = True
) -> DataFrame:
    columns = [
        _column_to_series(column, allow_copy=allow_copy) for column in df.get_columns()
    ]
    return pl.DataFrame(columns)


def _column_to_series(column: Column, *, allow_copy: bool = True) -> Series:
    polars_dtype = dtype_to_polars_dtype(column.dtype)

    buffers = column.get_buffers()

    data_buffer = _construct_data_buffer(*buffers["data"])
    validity_buffer = _construct_validity_buffer(
        column.describe_null,
        buffers["validity"],
        data_buffer,
        column.size(),
        column.offset,
        allow_copy=allow_copy,
    )
    offsets_buffer = _construct_offsets_buffer(buffers["offsets"])

    pyseries = wrap_s(
        PySeries.from_buffers(
            polars_dtype, data_buffer, validity_buffer, offsets_buffer
        )
    )
    return pyseries


def _construct_data_buffer(buffer_info: tuple[Buffer, Dtype]) -> Series:
    buffer, dtype = buffer_info
    polars_physical_dtype = dtype_to_polars_physical_dtype(dtype)

    s = wrap_s(PySeries.from_buffer(buffer.ptr, buffer.bufsize, polars_physical_dtype))
    s.__buffer = buffer  # Keep memory alive

    return s


def _construct_validity_buffer(
    describe_null: tuple[ColumnNullType, Any],
    validity_buffer_info: tuple[Buffer, Dtype] | None,
    data_buffer: Series,
    length: int,
    offset: int,
    *,
    allow_copy: bool = True,
) -> Series | None:
    null_type, value = describe_null
    if null_type == ColumnNullType.NON_NULLABLE:
        return None
    elif null_type in (ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        if validity_buffer_info is None:
            return None

        buffer, dtype = validity_buffer_info
        if null_type == ColumnNullType.USE_BITMASK:
            return _construct_validity_buffer_from_bitmask(
                buffer, dtype, value, length, offset, allow_copy=allow_copy
            )
        else:
            return _construct_validity_buffer_from_bytemask(
                buffer, dtype, value, length, offset, allow_copy=allow_copy
            )
    elif null_type in (ColumnNullType.USE_NAN, ColumnNullType.USE_SENTINEL):
        if not allow_copy:
            raise RuntimeError("bitmask must be constructed, which is not zero-copy")

        if null_type == ColumnNullType.USE_NAN:
            return data_buffer.is_not_nan()
        else:
            return data_buffer != value
    else:
        raise NotImplementedError(f"unsupported null type: {null_type!r}")


def _construct_validity_buffer_from_bitmask(
    buffer_info: tuple[Buffer, Dtype] | None,
    value: int,
    length: int,
    offset: int = 0,
    *,
    allow_copy: bool = True,
) -> Series:
    null_kind, sentinel_val = describe_null
    validity_kind, _, _, _ = validity_dtype
    assert validity_kind == DtypeKind.BOOL


    elif null_kind == ColumnNullType.USE_BYTEMASK or (
        null_kind == ColumnNullType.USE_BITMASK and sentinel_val == 1
    ):
        buff = pa.foreign_buffer(
            validity_buff.ptr, validity_buff.bufsize, base=validity_buff
        )

        if null_kind == ColumnNullType.USE_BYTEMASK:
            if not allow_copy:
                raise RuntimeError(
                    "To create a bitmask a copy of the data is "
                    "required which is forbidden by allow_copy=False"
                )
            mask = pa.Array.from_buffers(pa.int8(), length, [None, buff], offset=offset)
            mask_bool = pc.cast(mask, pa.bool_())
        else:
            mask_bool = pa.Array.from_buffers(
                pa.bool_(), length, [None, buff], offset=offset
            )

        if sentinel_val == 1:
            mask_bool = pc.invert(mask_bool)

        return mask_bool.buffers()[1]

    elif null_kind == ColumnNullType.USE_BITMASK and sentinel_val == 0:
        return pa.foreign_buffer(
            validity_buff.ptr, validity_buff.bufsize, base=validity_buff
        )
    else:
        raise NotImplementedError(
            f"{describe_null} null representation is not yet supported."
        )


def _construct_validity_buffer_from_bytemask(
    validity_buff: Any,
    validity_dtype: Dtype,
    describe_null: ColumnNullType,
    length: int,
    offset: int = 0,
    allow_copy: bool = True,
) -> Series:
    null_kind, sentinel_val = describe_null
    validity_kind, _, _, _ = validity_dtype
    assert validity_kind == DtypeKind.BOOL


    elif null_kind == ColumnNullType.USE_BYTEMASK or (
        null_kind == ColumnNullType.USE_BITMASK and sentinel_val == 1
    ):
        buff = pa.foreign_buffer(
            validity_buff.ptr, validity_buff.bufsize, base=validity_buff
        )

        if null_kind == ColumnNullType.USE_BYTEMASK:
            if not allow_copy:
                raise RuntimeError(
                    "To create a bitmask a copy of the data is "
                    "required which is forbidden by allow_copy=False"
                )
            mask = pa.Array.from_buffers(pa.int8(), length, [None, buff], offset=offset)
            mask_bool = pc.cast(mask, pa.bool_())
        else:
            mask_bool = pa.Array.from_buffers(
                pa.bool_(), length, [None, buff], offset=offset
            )

        if sentinel_val == 1:
            mask_bool = pc.invert(mask_bool)

        return mask_bool.buffers()[1]

    elif null_kind == ColumnNullType.USE_BITMASK and sentinel_val == 0:
        return pa.foreign_buffer(
            validity_buff.ptr, validity_buff.bufsize, base=validity_buff
        )
    else:
        raise NotImplementedError(
            f"{describe_null} null representation is not yet supported."
        )


def _construct_validity_buffer_from_nan(
    data_buffer: Series, sentinel_value: Any
) -> Series:
    return data_buffer.is_not_nan()


def _construct_validity_buffer_from_sentinel(
    data_buffer: Series, sentinel_value: Any
) -> Series:
    return data_buffer != sentinel_value


def _construct_offsets_buffer(
    buffer_info: tuple[Buffer, Dtype] | None, *, allow_copy: bool = True
) -> Series | None:
    if buffer_info is None:
        return None

    buffer, dtype = buffer_info
    polars_physical_dtype = dtype_to_polars_physical_dtype(dtype)

    s = wrap_s(PySeries.from_buffer(buffer.ptr, buffer.bufsize, polars_physical_dtype))
    s.__buffer = buffer  # Keep memory alive

    # Polars only supports 'large' types, which have Int64 offsets
    if polars_physical_dtype != Int64:
        if not allow_copy:
            raise RuntimeError()
        s = s.cast(Int64)

    return s
