import numpy as np
import pandas as pd
import random
import copy

from queries import *


def sample_race(q):
    """
    """
    race_counts = q.race_hisp.sum(axis=1)
    race_prob = race_counts/ race_counts.sum()
    rrace = np.random.choice(range(q.num_race), p=race_prob)
    return rrace

def sample_hisp(q, rrace):
    """
    """
    hisp_prob = q.race_hisp[rrace] / q.race_hisp[rrace].sum()
    rhisp = np.random.choice(range(q.num_hisp), p=hisp_prob)
    return rhisp

def sample_age(q, rrace, rhisp):
    """
    """
    race_coarse = q.race_coarsened(rrace)

    if race_coarse == 0:
        age_prob = q.white_hisp_age_sex[rhisp].sum(axis=1) / q.white_hisp_age_sex[rhisp].sum(axis=1).sum()
    else:
        age_prob = q.race_age_sex[race_coarse].sum(axis=1) / q.race_age_sex[race_coarse].sum(axis=1).sum()


    rage = np.random.choice(range(q.num_age), p=age_prob)
    return rage

def sample_sex(q, rrace, rhisp, rage):
    """
    """
    race_coarse = q.race_coarsened(rrace)

    if race_coarse == 0:
        sex_prob = q.white_hisp_age_sex[rhisp, rage] / q.white_hisp_age_sex[rhisp, rage].sum()
    else:
        sex_prob = q.race_age_sex[race_coarse, rage] / q.race_age_sex[race_coarse, rage].sum()

    rsex = np.random.choice(range(q.num_sex), p=sex_prob)
    return rsex

def sample_row(q, verbose=False):
    """
    """
    if verbose:
        print("Printing state:")
        q.print_state()

    rrace = sample_race(q)
    rhisp = sample_hisp(q, rrace)
    rage = sample_age(q, rrace, rhisp)
    rsex = sample_sex(q, rrace, rhisp, rage)

    return (rrace, rhisp, rage, rsex)


def add_row_to_reconstruction(recon_df, row, blk_id):
    """
    """
    rrace, rhisp, rage, rsex = row
    recon_df.loc[len(recon_df)] = [blk_id, rrace, rhisp, rage, rsex]

def reconstruct_block(blk_id, queries, verbose=False):
    """
    """
    q = copy.deepcopy(queries)

    recon_df = pd.DataFrame(columns=["TABBLK", "CENRACE", "CENHISP", "QAGE", "QSEX"])
    for _ in range(q.total):
        row = sample_row(q, verbose=verbose)
        add_row_to_reconstruction(recon_df, row, blk_id)
        remove_row_from_queries(row, q)

    assert(np.sum(q.race_hisp) == 0)
    assert(np.sum(q.white_hisp_age_sex) == 0)
    assert(np.sum(q.race_age_sex) == 0)

    return recon_df

def remove_row_from_queries(row, q):
    rrace, rhisp, rage, rsex = row

    q.total -= 1
    q.race_hisp[rrace, rhisp] -= 1

    if rrace == 0:
        q.white_hisp_age_sex[rhisp, rage, rsex] -= 1
    else:
        race_coarse = q.race_coarsened(rrace)
        q.race_age_sex[race_coarse, rage, rsex] -= 1

def reconstruct_at_block_level(queries):
    """
    """
    main_df = pd.DataFrame(columns=["TABBLK", "CENRACE", "CENHISP", "QAGE", "QSEX"])

    for (block_id, block_queries) in queries.items():
        recon_df = reconstruct_block(block_id, block_queries)
        main_df = pd.concat([main_df, recon_df])

    return main_df

def reconstruct_at_tract_level(tract_queries, blk_ids, blk_counts):
    """
    """
    blk_counts = blk_counts.copy()
    recon_df = reconstruct_at_block_level(tract_queries)
    recon_df = populate_blocks(recon_df, blk_ids, blk_counts)
    return recon_df

def tests_passed(real_queries, recon_df):
    """
    """
    reconq = get_block_queries(recon_df)

    for (block_id, realq) in real_queries.items():
        assert(realq.total == reconq[block_id].total)
        assert(np.sum(realq.race_hisp == reconq[block_id].race_hisp)
               == (realq.num_race * realq.num_hisp))

        assert(np.sum(realq.white_hisp_age_sex == reconq[block_id].white_hisp_age_sex)
               == (realq.num_hisp * realq.num_age * realq.num_sex))

        assert(np.sum(realq.race_age_sex == reconq[block_id].race_age_sex)
               == (realq.num_race_coarse * realq.num_age * realq.num_sex))

    print("Tests passed!")
    return True

def get_block_queries(df):
    """
    """
    block_queries = {}

    for block_id in sorted(df["TABBLK"].unique()):
        block = df[df["TABBLK"] == block_id]
        block_queries[block_id] = query_manager(block)

    return block_queries

def get_num_reconstructed(recon, real):
    """
        `recon' and `real' are both pandas DataFrames.
    """
    real_unique = real.groupby(real.columns.tolist(), as_index=False).size()
    recon_unique = recon.groupby(recon.columns.tolist(), as_index=False).size()

    num_reconstructed = 0

    for idx, recon_row in recon_unique.iterrows():
        (blk, race, hisp, age, sex, size) = recon_row
        real_row = real_unique.query(f'TABBLK == {blk} and ' +
                                     f'CENRACE == {race} and ' +
                                     f'CENHISP == {hisp} and ' +
                                     f'QAGE == {age} and ' +
                                     f'QSEX == {sex}')

        if len(real_row) > 0:
            num_reconstructed += min(real_row["size"].item(), size)

    return num_reconstructed

def get_n_reconstructions(block_queries, n, columns, block_level=True,
                          verbose=True, blk_ids=None, blk_counts=None):
    """
    """
    if block_level:
        if verbose:
            print("Print Getting Block-Level Reconstructions")
        assert(blk_ids is None)
        assert(blk_counts is None)

    all_recons = pd.DataFrame(columns=columns)
    counter = 0
    while counter < n:
        counter += 1
        if verbose:
            print(f'Reconstructing dataset {counter}/{n}')
        if block_level:
            recon = reconstruct_at_block_level(block_queries)
        else:
            recon = reconstruct_at_tract_level(block_queries, blk_ids, blk_counts)
        recon["run"] = counter
        all_recons = pd.concat([all_recons, recon])
    return all_recons

def get_blk_ids_and_counts(df):
    """ Returns 1. an array of block ids.
                2. an array of counts associated with those block ids.
    """
    blk_ids = np.array(df.groupby(["TABBLK"]).count().index.array)
    counts = np.array(df.groupby(["TABBLK"]).count().iloc[:, 0])

    return blk_ids, counts

def sample_block(blk_ids, blk_counts):
    probs = blk_counts / np.sum(blk_counts)
    return np.random.choice(blk_ids, p=probs)

def remove_block(blk, blk_ids, blk_counts):
    pos = np.where(blk_ids==blk)
    assert(len(pos) == 1)
    pos = pos[0][0]
    blk_counts[pos] -= 1

def populate_blocks(recon, blk_ids, blk_counts):
    """
    """
    for index, _ in recon.iterrows():
        blk = sample_block(blk_ids, blk_counts)
        recon.iloc[index]["TABBLK"] = blk
        remove_block(blk, blk_ids, blk_counts)

    assert(np.sum(blk_counts) == 0)
    return recon
