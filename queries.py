import numpy as np
import pandas as pd
import random
import copy

class Queries:
    def __init__(self, num_race, num_hisp, num_age, num_sex,
                 block_col, race_col, hisp_col, age_col, sex_col):
        self.total = 0

        # save_params
        self.num_race = num_race
        self.num_hisp = num_hisp
        self.num_age = num_age
        self.num_sex = num_sex
        self.num_race_coarse = 7 # counting white only, the 5 other races only, and then "two or more races"

        self.block_col = block_col
        self.race_col = race_col
        self.hisp_col = hisp_col
        self.age_col = age_col
        self.sex_col = sex_col

        self.block_id = None

        # initialize arrays
        self.race_hisp = np.zeros([num_race, num_hisp])
        self.white_hisp_age_sex = np.zeros([num_hisp, num_age, num_sex])
        self.race_age_sex = np.zeros([self.num_race_coarse, num_age, num_sex])

    def add_total(self, block):
        self.total = len(block)

    def add_race_hisp(self, block):
        for race in range(self.num_race):
            for hisp in range(self.num_hisp):
                self.race_hisp[race, hisp] = len(block.query(f'{self.hisp_col} == {hisp} and {self.race_col} == {race}'))

    def add_white_hisp_age_sex(self, block):
        for hisp in range(self.num_hisp):
            for age in range(self.num_age):
                for sex in range(self.num_sex):
                    self.white_hisp_age_sex[hisp, age, sex] = len(block.query(f'{self.sex_col} == {sex} and ' +
                                                                              f'{self.age_col} == {age} and ' +
                                                                              f'{self.race_col} == 0 and ' +
                                                                              f'{self.hisp_col} == {hisp}'))
    def race_coarsened(self, race):
        """
        """
        return 6 if race > 5 else race

    def add_single_race_age_sex_except_white(self, block):
        """
        """
        for race in range(1, 6):
            for age in range(self.num_age):
                for sex in range(self.num_sex):
                    self.race_age_sex[race, age, sex] = len(block.query(f'{self.sex_col} == {sex} and ' +
                                                                        f'{self.age_col} == {age} and ' +
                                                                        f'{self.race_col} == {race}'))

    def add_multiracial_race_age(self, block):
        """
        """
        multirace_idx = 6

        for age in range(self.num_age):
            for sex in range(self.num_sex):
                self.race_age_sex[multirace_idx, age, sex] = len(block.query(f'{self.sex_col} == {sex} and ' +
                                                                             f'{self.age_col} == {age} and ' +
                                                                             f'{self.race_col} >= 6'))

    def add_race_age_sex(self, block):
        """
        """
        self.add_single_race_age_sex_except_white(block)
        self.add_multiracial_race_age(block)

    def add_block_id(self, block):
        """
        """
        self.block_id = block[self.block_col].iloc[0]

    def generate_queries(self, block):
        """
        """
        self.add_total(block)
        self.add_block_id(block)
        self.add_race_hisp(block)
        self.add_white_hisp_age_sex(block)
        self.add_race_age_sex(block)

    def print_state(self):
        print("Total: ", self.total)
        print("White_Hisp_Age_Sex: ", self.white_hisp_age_sex)
        print("Race_Age_Sex: ", self.race_age_sex)

def age_buckets(age):
    """ Given an age returns the bucket it fall in.
    """
    buckets = [(0, 5),
               (5, 10),
               (10, 15),
               (15, 18),
               (18, 20),
               (20, 21),
               (21, 22),
               (22, 25),
               (25, 30),
               (30, 35),
               (35, 40),
               (40, 45),
               (45, 50),
               (50, 55),
               (55, 60),
               (60, 62),
               (62, 65),
               (65, 67),
               (67, 70),
               (70, 75),
               (75, 80),
               (80, 85),
               (85, 126)
              ]

    for (idx, bucket) in enumerate(buckets):
        start, stop = bucket
        if age in range(start, stop):
            return idx

    raise Exception("Age out of bounds: ", age)

def query_manager(block):
    """
    """
    num_race = 63
    num_hisp = 2
    num_sex = 2
    num_age = 23

    block_col = "TABBLK"
    race_col = "CENRACE"
    hisp_col = "CENHISP"
    age_col = "QAGE"
    sex_col = "QSEX"

    queries = Queries(num_race, num_hisp, num_age, num_sex, block_col, race_col, hisp_col, age_col, sex_col)
    queries.generate_queries(block)

    return queries
