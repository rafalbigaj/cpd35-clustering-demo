import cupy as cp
import pandas as pd
import cudf
import dask_cudf

def good_neigbour(df):
    """
    Computation of a big correlation matrix. Should be done on a GPU. We could not test this, as there is no gpu support
    on power. We expect that dask_cudf works on x86.
    :param df:
    :return:
    """
    cuda = cudf.DataFrame(df)
    df = dask_cudf.from_cudf(cuda, npartitions=2)

    df = df.groupby(['account', 'date'])['volume'].sum()

    unique_market_parties = df.index.get_level_values('account').unique()
    timepoints = df.index.get_level_values('date').unique()
    index = pd.MultiIndex.from_product([unique_market_parties, timepoints], names=['account', 'date'])
    corss_account_owners_timepoints = pd.DataFrame(index=index)
    corss_account_owners_timepoints = corss_account_owners_timepoints.sort_values(['account', 'date'])

    df = pd.merge(df, corss_account_owners_timepoints, on=['account', 'date'], how="outer")
    df['volume'] = df['volume'].fillna(0)
    df = df['volume']

    cor = df.unstack(level='account').corr()

    cor.index = cor.index.rename('center')
    cor.columns = cor.columns.rename('Peripherie')
    cor = cor.stack()
    cor.name = 'correlation'
    cor = cor.to_frame()

    buddy = cor.groupby('center')['correlation'].nsmallest(1)

    return buddy