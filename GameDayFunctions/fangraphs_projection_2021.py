import pdb
import os
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

class Projection:
    ''' Read and Store Fangraphs Projection files in instance of class Projection.

    Parameters
    ----------
    path_data : string [optional]
        Where the data is stored (before date)

    model : string [optional]
        Choose from ZiPS (default), Steamer, Fans

    year: int [optional]
        year of projections

    Returns
    -------
    Instance of Projection, which contains:
        Objects:
        - self.statline
        - self.hitters_rank
        - self.pitchers_rank
        - self.hitter_stats
        - self.pitchers_stats

        Functions:
        - precompute_statlines

    '''

    def __init__(self, model = 'ZiPS', year = 2021, path_data = "projections/",
                 ranking_method = 'FantasyPros', ranking_file = False):
        self.statline = {}
        self.all_rank = {}
        self.hitters_rank = {}
        self.pitchers_rank= {}
        self.hitters_stats = pd.DataFrame()

        # Read in Batters by Position for Year and Position
        ranking_file = ranking_method + '_Roto_Ranking_' + str(year) + '.xlsx'

        # Use Rankings of choice; Yahoo, ESPN, FantasyPros
        xls = os.path.join(path_data+str(year)+'/PositionalRankings/'+ranking_method+'/', ranking_file)
        self.all_rank = pd.read_excel(xls, skiprows = 0, index_col = 'RANK')
        for irename in range(len(self.all_rank)):
            self.all_rank.iloc[irename]['PLAYER'] = remove_special_characters(self.all_rank.iloc[irename]['PLAYER'])

        #pdb.set_trace()
        # Loop through all files in path.
        for file in os.listdir(path_data+str(year)+'/'):
            # Skip files that don't end in a position.
            # print(file)
            #pdb.set_trace()

            if file.startswith(model) & file.endswith('Hitters.csv'):
                print(os.path.join(path_data + str(year) + '/', file))
                #pdb.set_trace()
                df = pd.read_csv(os.path.join(path_data + str(year) + '/', file), index_col='playerid')
                self.hitters_stats = df


            if file.startswith(model) & file.endswith('Pitchers.csv'):
                print(os.path.join(path_data + str(year) + '/', file))
                #pdb.set_trace()
                df = pd.read_csv(os.path.join(path_data + str(year) + '/', file), index_col='playerid')
                self.pitchers_stats = df

        self.add_position_column()

    def add_position_column(self):
        ''' Merge Positions from all_rank into hitters_stats and pitchers_stats dataframes'''

        hitter_positions = []
        for i,row in self.hitters_stats.iterrows():
            name = row.Name.split(' Jr.')[0]
            idx = self.all_rank.PLAYER.str.contains(name)
            if np.sum(idx) == 0:
                first_name = name.split(' ')[0]
                last_name = name.split(' ')[1]
                idx = self.all_rank.PLAYER.str.contains(last_name)
                if np.sum(idx) > 1:
                    idx = self.all_rank.PLAYER.str.contains(first_name[0] + '&' + last_name)
                    if np.sum(idx) > 1:
                        idx = self.all_rank.PLAYER.str.contains(first_name[:3] + '&' + last_name)

            if np.sum(idx) > 1:
                # If there are multiple players with same name (e.g., Will Smith), find position that is not a pitcher
                # print(np.sum(idx))
                pin = self.all_rank[idx][~self.all_rank[idx]['Elig. Pos.'].str.contains('P')]['Elig. Pos.']
            else:
                pin = self.all_rank[idx]['Elig. Pos.']

            if pin.empty:
                #print(row.Name)
                #pdb.set_trace()
                hitter_positions.extend(['NA'])
            else:
                hitter_positions.extend(pin)

        self.hitters_stats['EligiblePosition'] = hitter_positions

        pitcher_positions = []
        for i,row in self.pitchers_stats.iterrows():
            name = row.Name.split(' Jr.')[0]
            idx = self.all_rank.PLAYER.str.contains(name)
            if np.sum(idx) == 0:
                first_name = name.split(' ')[0]
                last_name = name.split(' ')[1]
                idx = self.all_rank.PLAYER.str.contains(last_name)
                if np.sum(idx) > 1:
                    idx = self.all_rank.PLAYER.str.contains(first_name[0] + '&' + last_name)
                    if np.sum(idx) > 1:
                        idx = self.all_rank.PLAYER.str.contains(first_name[:3] + '&' + last_name)

            if np.sum(idx) > 1:
                # If there are multiple players with same name (e.g., Will Smith), find position that is not a pitcher
                # print(np.sum(idx))
                pin = self.all_rank[idx][self.all_rank[idx]['Elig. Pos.'].str.contains('P')]['Elig. Pos.']
            else:
                pin = self.all_rank[idx]['Elig. Pos.']

            if pin.empty:
                #print(row.Name)
                pitcher_positions.extend(['NA'])
            else:
                pitcher_positions.extend(pin)

        self.pitchers_stats['EligiblePosition'] = pitcher_positions

        # Drop NA from dataframes
        #pdb.set_trace()

        # Add missing stats (like 1B, CG, SHO, SV, and BSV)
        self.define_missing_stats()

    def define_missing_stats(self):
        ''' Some Roto Stats are missing from Fangraphs Projections.  Include the trivial ones (singles).  Invent the non-obvious ones (blown saves)'''

        # Hitters_stats:
        # Add singles
        self.hitters_stats['1B'] = self.hitters_stats['H'] - self.hitters_stats['2B'] - self.hitters_stats['3B'] - \
                                   self.hitters_stats['HR']

        # Pitchers stats:
        # Starters complete games and shutouts
        ind_sp = self.pitchers_stats['EligiblePosition'].str.contains('SP')
        #pdb.set_trace()
        self.pitchers_stats['CG'] = 0
        self.pitchers_stats['CG'][ind_sp] = np.floor(
            self.pitchers_stats['IP'][ind_sp] * 0.01 * (1. / self.pitchers_stats['WHIP'][ind_sp]))

        self.pitchers_stats['SHO'] = 0
        self.pitchers_stats['SHO'][ind_sp] = np.ceil(self.pitchers_stats['CG'][ind_sp] * 0.55)

        # Relievers saves and blown saves
        ind_rp = self.pitchers_stats['EligiblePosition'].str.contains('RP')
        self.pitchers_stats['SV'] = 0
        self.pitchers_stats['SV'][ind_rp] = np.floor(
            self.pitchers_stats['IP'][ind_rp] * 0.5 * (1. / self.pitchers_stats['WHIP'][ind_rp]))

        self.pitchers_stats['BSV'] = 0
        self.pitchers_stats['BSV'][ind_rp] = np.floor(
            self.pitchers_stats['IP'][ind_rp] * 0.05 * (1. / self.pitchers_stats['WHIP'][ind_rp]))


def remove_special_characters(name_in):
    name_out = ((((((name_in.replace('ñ','n')).replace('í','i')).replace('é','e')).replace('á','a')).replace('ú','u')).replace('ó','o')).split(' Jr.')
    return name_out[0]

