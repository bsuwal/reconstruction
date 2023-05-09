import matplotlib.pyplot as plt



def is_row_reconstructed(recon_row, real):
    """ if yes then returns (Boolean, index)
        where index is the first index of a matching row in real
    """
    if len(real) == 0:
        return (False, None)

    (blk, race, hisp, age, sex, _) = recon_row
    matched_rows = real.query(f'TABBLK == {blk} and ' +
                              f'CENRACE == {race} and ' +
                              f'CENHISP == {hisp} and ' +
                              f'QAGE == {age} and ' +
                              f'QSEX == {sex}')

    if len(matched_rows) > 0:
        return (True, matched_rows.index[0])
    else:
        return (False, None)

def plot_match_rate(all_recons_unique, real, title, num_runs, stable=False, save=False):
    u = len(all_recons_unique)

    if stable:
        real = real.copy()
    xs = []
    ys = []

    successful_recon = {}
    for k in range(u):
        status, idx = is_row_reconstructed(all_recons_unique.iloc[k], real)
        successful_recon[k] = status

        if status and stable:
            real = real.drop([idx])
        mr = sum(successful_recon.values()) / (k + 1)
        xs.append(k/u)
        ys.append(mr)

    plt.plot(xs, ys)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel("Match Rate")
    plt.xlabel("k/u")
    title_str = "imgs/" + title + f' num_runs={num_runs}, (u = {u})'
    plt.title(title_str)
    if save:
        plt.savefig(title_str)
    plt.show()

    return successful_recon
