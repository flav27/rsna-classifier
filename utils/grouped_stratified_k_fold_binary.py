import numpy as np
from time import time
from numpy.random import choice
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


class GroupStratifiedShuffleSplitBinary:
    """ A cross-validator (CV) object akin to sklearn's native GroupShuffleSplit or
        StratifiedShuffleSplit where both groups AND stratification (currently only
        with binary classification) are taken into account.

        target_var : the dataframe column which is the target to stratify by,
        group_var  : the dataframe column which specifies the groups kept seperate
                     in training and test splits,
        frac_train : fraction to put as CV train set (frac_test = 1 - frac_train)
        verbose    : provide textual feedback to user
        beta       : (see below)
        max_evals  : (see below)
        n_splits   : number of splits to perform

        GroupStratifiedShuffleSplitBinary is initialized with group and target info,
        using this we create a 'group_table' which has the format:

                                n_target | n_entries | ratio <-------- i.e. imbalance of group
                             ===================================
            group_id (index)|            |           |
                            |            |           |
                            |            |           |

        The subsequent .split method then creates a training and test set
        by randomly sampling groups from group_table with the goal that:

                (i)  the number of entries in train/test are approx equal to
                     those defined by the specified train fraction 'frac_train'
                (ii) the ratio n_target/n_entries in train (and therefore test)
                     is approx equal to that ratio for the entire dataset

        Sampling from group_table continues until frac_train has been approx reached
        (from below) OR until it has performed max_evals samples.

        Specifically, the train subset is formed by sampling from the group_table in
        such a way that the probability of choosing a given group is related to the
        difference Delta, where

            Delta := -([ratio-curr_train_imbalance] x sgn(curr_train_imbalance-data_set_imbalance))

        that is, when the current (partially filled) training set has a group
        imbalance greater than that of the actual dataset we're sampling from,
        groups with imbalance less than the current train imbalance are
        preferentially selected (and vice versa). Concretely, the probabilty
        of next selecting group i to add to our training set is:

            P(i) = softmax(Delta(i) x beta)

        By sampling from the group_table we are guaranteed to have the desired
        property that groups are never mixed between train and test. However, we
        can't simply draw groups randomly, because we want a train set with the
        same imbalance as the overall data set. This probabalistic method of
        choosing means that the partial training set's imbalance oscillates,
        hopefully converging on the desired (i.e. data set) imbalance.

        This can have the downside that groups with extreme values of imbalance
        (high or low) are preferentially chosen. It is therefore necessary to
        modify the parameter beta to find a satisfactory balance between groups
        being uniformly selected, and the training set being stratified (e.g.
        by using the method test_make_one_group_stratified_shuffle_split).

        The fact that GroupStratifiedShuffleSplitBinary is initialized with
        all required info makes it a bit different from the usual sklearn
        cross-validators. The result is that the split method returns train-test
        indeces even without input. But of course when called through an sklearn
        method e.g. RandomSearchCV().fit(X,y), the X and y must match to the data
        passed in to initialize GroupStratifiedShuffleSplitBinary.

    """

    def __init__(self, target_var, group_var, frac_train=0.8, verbose=False,
                 beta=100, max_evals=1e6, n_splits=3):
        """ target_var is the dataframe coloum which is the target, group_var is the
            dataframe column which specifies the group, n_splits is number of splits to perform
        """

        self.target_var = target_var
        self.group_var = group_var
        self.frac_train = frac_train
        self.n_splits = n_splits
        self.verbose = verbose
        self.beta = beta
        self.max_evals = max_evals
        # we'll often want to know the data set imbalance
        self.data_set_imbalance = np.sum(self.target_var == 1) / len(self.target_var)
        if self.verbose:
            print(
                'Created GroupStratifiedShuffleSplit object for stratified splitting of binary class data with groups.')
            print('Total df entries = {}, total groups = {}, N(y=0)/(N(y=0)+N(y=1)) = {:.4f}'.format(
                len(self.target_var),
                len(self.group_var.unique()),
                self.data_set_imbalance
            ))
        # make the group table we'll use for sampling
        self.make_group_table()
        # train_groups and test groups are lists of group labels for train and test
        self.train_groups = []
        self.test_groups = []

    def __iter__(self):
        return self

    def __next__(self):
        result = self.split()
        return result

    def make_group_table(self):
        """ make a pandas table with info on each group which will be used later
        """

        self.group_table = pd.DataFrame({'group_id': self.group_var,
                                         'flag': self.target_var}).groupby('group_id').agg(['sum', 'count'])
        self.group_table.columns = self.group_table.columns.droplevel()
        self.group_table.rename(columns={'sum': 'n_target', 'count': 'n_entries'}, inplace=True)
        self.group_table['ratio'] = self.group_table['n_target'] / self.group_table['n_entries']
        # make a numpy version with columns [ group_id | n_target | n_entries | ratio ]
        self.group_numpyarray = self.group_table.reset_index().values
        if self.verbose:
            print('Generated group_table')

    def make_one_group_stratified_shuffle_split(self):
        """ Given a numpy 'group_numpyarray' which has columns:
            [ group_id | n_target | n_entries | ratio ], make a
            train test split in the manner described in the docstring

        """

        # choose group at random to kick things off
        train_groups = [choice(self.group_numpyarray[:, 0])]
        curr_train_frac = np.sum(self.group_numpyarray[np.isin(self.group_numpyarray[:, 0], train_groups), 2]) / np.sum(
            self.group_numpyarray[:, 2])
        curr_train_imbalance = np.sum(
            self.group_numpyarray[np.isin(self.group_numpyarray[:, 0], train_groups), 1]) / np.sum(
            self.group_numpyarray[np.isin(self.group_numpyarray[:, 0], train_groups), 2])

        iterator = 0
        while (curr_train_frac < self.frac_train) & (iterator <= self.max_evals):  #

            if self.verbose:
                print('(Step {}) Train fraction = {:.4f}, Train imbalance = {:.4f} (full set = {:.4f})'.format(iterator,
                                                                                                               curr_train_frac,
                                                                                                               curr_train_imbalance,
                                                                                                               self.data_set_imbalance))
            # calculate preference for choosing next customer (boltz dist)
            next_choice_vec = -1.0 * (self.group_numpyarray[~np.isin(self.group_numpyarray[:, 0],
                                                                     train_groups), 3] - curr_train_imbalance) * np.sign(
                curr_train_imbalance - self.data_set_imbalance)
            next_choice_vec = np.array(next_choice_vec, dtype=np.float64)
            boltz_prob = np.exp(next_choice_vec * self.beta) / np.sum(np.exp(next_choice_vec * self.beta))
            # now make choice
            train_groups = np.append(train_groups, choice(
                self.group_numpyarray[~np.isin(self.group_numpyarray[:, 0], train_groups), 0], p=boltz_prob))
            # calculate updated properties
            curr_train_frac = np.sum(
                self.group_numpyarray[np.isin(self.group_numpyarray[:, 0], train_groups), 2]) / np.sum(
                self.group_numpyarray[:, 2])
            curr_train_imbalance = np.sum(
                self.group_numpyarray[np.isin(self.group_numpyarray[:, 0], train_groups), 1]) / np.sum(
                self.group_numpyarray[np.isin(self.group_numpyarray[:, 0], train_groups), 2])
            iterator = iterator + 1

            # test set is then anything not in train set
        test_groups = self.group_numpyarray[~np.isin(self.group_numpyarray[:, 0], train_groups), 0]

        if self.verbose:
            if iterator == self.max_evals:
                print('\nExited having reached max number of evaluations ({})'.format(self.max_evals))
            else:
                print('\nCompleted succesfully with:')
                print('Train fraction = {:.4f}, Train imbalance = {:.4f} (full set = {:.4f})'.format(curr_train_frac,
                                                                                                     curr_train_imbalance,
                                                                                                     self.data_set_imbalance))

                curr_test_frac = np.sum(
                    self.group_numpyarray[~np.isin(self.group_numpyarray[:, 0], train_groups), 2]) / np.sum(
                    self.group_numpyarray[:, 2])
                curr_test_imbalance = np.sum(
                    self.group_numpyarray[~np.isin(self.group_numpyarray[:, 0], train_groups), 1]) / np.sum(
                    self.group_numpyarray[~np.isin(self.group_numpyarray[:, 0], train_groups), 2])

                print('Test fraction = {:.4f}, Test imbalance = {:.4f} (full set = {:.4f})'.format(curr_test_frac,
                                                                                                   curr_test_imbalance,
                                                                                                   self.data_set_imbalance))
                print('|Train_groups| = {}, |Test_groups| = {}'.format(len(train_groups), len(test_groups)))

        self.train_groups = train_groups
        self.test_groups = test_groups
        self.curr_train_frac = curr_train_frac
        self.curr_train_imbalance = curr_train_imbalance

    def split(self, X=[], y=[], groups=None):
        """ Return indices of train and test splits
        """

        for isplit in range(self.n_splits):
            # make a train test split
            self.make_one_group_stratified_shuffle_split()
            # get the appropriate indices

            train_indx = self.group_var[self.group_var.isin(self.train_groups)].index
            test_indx = self.group_var[self.group_var.isin(self.test_groups)].index
            # N.B. sklearn wants positional locations (iloc)! Not indices (loc)!
            # they also must be numoy arrays so that they have a "flag" property
            train_indx = np.array([self.group_var.index.get_loc(idx) for idx in train_indx])
            test_indx = np.array([self.group_var.index.get_loc(idx) for idx in test_indx])

            yield train_indx, test_indx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

    def test_make_one_group_stratified_shuffle_split(self, iterations=100, save_loc=''):
        """ make a plot to see how well make_one_group_stratified_shuffle_split is shuffling
            the groups
        """

        t1 = time()
        self.make_one_group_stratified_shuffle_split()
        train_group_multiple = self.train_groups.copy()
        stratification_arr = [self.curr_train_imbalance]
        for l in range(iterations - 1):
            self.make_one_group_stratified_shuffle_split()
            train_group_multiple = np.append(train_group_multiple, self.train_groups)
            stratification_arr = np.append(stratification_arr, self.curr_train_imbalance)

        train_group_multiple.sort()
        count_groups = Counter(train_group_multiple)
        fig, ax = plt.subplots(figsize=[32, 16])
        ax.bar(x=np.arange(len(count_groups)), height=[count_groups[count] for count in count_groups],
               color=np.array([95, 75, 139]) / 255)
        ax.set_xticks(np.arange(len(count_groups)))
        ax.set_xticklabels([group for group in count_groups], rotation=90)

        ax_r = ax.twinx()
        ax_r.scatter(np.arange(len(count_groups)), self.group_table.loc[
            self.group_table.index.isin([group for group in count_groups])].sort_index().ratio,
                     color=np.array([228, 93, 191]) / 255)
        ax_r.set_ylabel('N(y=0)/(N(y=0)+N(y=1))')
        ax.set_title('n_iterations = {}, train_frac = {}, beta = {}'.format(iterations, self.frac_train, self.beta))
        ax.set_ylabel('n_times occurred in training set')
        ax.set_xlim(-1, len(count_groups))

        if len(save_loc) > 0:
            plt.savefig(save_loc, dpi=600, bbox_inches='tight')

        plt.show()

        print('Stratification = {:.4f} +/- {:.4f} (target = {:.4f})'.format(np.mean(stratification_arr),
                                                                            np.std(stratification_arr),
                                                                            self.data_set_imbalance))
        print('(Time taken for calcualtion = {:.2}s)'.format((time() - t1)))