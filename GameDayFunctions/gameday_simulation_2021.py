# Load packages.  Must have pandas and numpy.

import pdb
import os
import sys
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import copy
import pickle
from GameDayFunctions.fangraphs_projection_2021 import Projection
from GameDayFunctions.draft_2021 import Draft

class Simulation:

    def __init__(self,
                 silent = True,
                 overwrite_file = False,
                 projection_type = 'ZiPS',
                 ranking_method = 'FantasyPros',
                 path_projections = 'projections/',
                 year = 2020,
                 number_teams = 12,
                 number_sims = 50,
                 path_sims = 'simulations/',
                 roster_spots = {'C':1,'1B':1,'2B':1, '3B':1,'SS':1,'OF':3,'UTIL':1,'SP':2,'RP':2,'P':3,'BN':1},
                 batter_stats  = ['AB','R','1B','2B', '3B','HR','RBI','SB','BB','AVG','OPS'],
                 pitcher_stats = ['IP','W', 'L','CG','SHO','SV','BB','SO','ERA','WHIP','BSV'],
                 filter_injured_players = True,
                 naive_draft = False,
                 shuffle_picks = True,
                 search_depth = 2,
                 autodraft_depth = 'end',
                 sigmoid_cut = 1e-6):

        self.projection_type = projection_type
        self.ranking_method = ranking_method
        self.path_projections = path_projections
        self.year = year
        self.number_teams = number_teams
        self.number_sims = number_sims
        self.path_sims = path_sims
        self.roster_spots = roster_spots
        self.batter_stats = batter_stats
        self.pitcher_stats = pitcher_stats

        simulation_output = self.simulate_multiple_drafts(naive_draft = naive_draft, shuffle_picks = shuffle_picks, search_depth = search_depth, autodraft_depth = autodraft_depth, silent=silent)
        #pdb.set_trace()
        compiled_player_rankings = self.compile_simulation_results(simulation_output, self.number_sims, self.number_teams)
        average_rankings = self.rank_simulation_result_averages(compiled_player_rankings)
        self.write_simulation_results(simulation_output,average_rankings)

        return simulation_output,average_rankings
    
    def simulate_multiple_drafts(self, naive_draft = False, shuffle_picks = True, search_depth = 1, autodraft_depth = 'end', silent = True):

        player_projections = Projection(path_data=self.path_projections,year=self.year,model=self.projection_type,ranking_method = self.ranking_method)

        # Define Simulation Dictionary
        simulation_results = {}

        for isim in range(self.number_sims):
            print('Sim '+str(isim))

            draft_position_results = {}
            for idraft_position in (np.arange(self.number_teams) + 1):

                # Get an instance of the Draft Class with your league-specific details and projection preference.
                simulated_draft_position_i = Draft(player_projections,
                                     draft_position = idraft_position,
                                     number_teams = self.number_teams,
                                     roster_spots = self.roster_spots,
                                     batter_stats = self.batter_stats,
                                     pitcher_stats = self.pitcher_stats)
                simulated_draft_position_i.draft_all(naive_draft = naive_draft, search_depth = search_depth, shuffle_picks = shuffle_picks, silent = silent)
                draft_position_results[idraft_position] = simulated_draft_position_i.drafted_team

            simulation_results[isim] = draft_position_results

        return simulation_results #simulated_compiled_player_rankings, simulated_average_rankings

    def compile_simulation_results(self, simulation_results, number_sims, number_teams):

        # Give each position a number to add as first entry in simulated_player_rankings
        position_dict = {'C':1,'1B':2,'2B':3,'SS':4,'3B':5,'OF':6,'SP':7,'RP':8,'P':9,'UTIL':10,'BN':11}
        simulated_player_rankings = {}

        for isim in range(number_sims):

            for idraft_position in (np.arange(number_teams) + 1):

                # Define
                draft_position = np.arange(number_teams)[::-1]
                for iround in range(len(simulation_results[isim][idraft_position])):

                    # Reverse draft order every round
                    draft_position = draft_position[::-1]

                    # Translate round/draft_position to overall draft position
                    #overall_draft_position = number_teams * iround + draft_position[idraft_position-1]
                    overall_draft_position =  iround + 1

                    # Get player name as key to simulated_player_rankings
                    player_name = simulation_results[isim][idraft_position][iround][1]
                    player_position = simulation_results[isim][idraft_position][iround][2]

                    # Compile overall_position's for each player
                    if player_name in simulated_player_rankings:
                        simulated_player_rankings[player_name].append(overall_draft_position)
                    else:
                        simulated_player_rankings[player_name] = [position_dict[player_position],overall_draft_position]

        return simulated_player_rankings

    def rank_simulation_result_averages(self, simulated_player_rankings):

        # Give each position a number, in reverse, to back out position from first entry in simulated_player_rankings
        rev_position_dict = {1:'C',2:'1B',3:'2B',4:'SS',5:'3B',6:'OF',7:'SP',8:'RP',9:'P',10:'UTIL',11:'BN'}

        average_simulated_player_rankings = {}
        simulated_player_positions = {}
        for i in simulated_player_rankings:
            average_simulated_player_rankings[i] = np.mean(simulated_player_rankings[i][1:])
            simulated_player_positions[i] = rev_position_dict[simulated_player_rankings[i][0]]

        ranked_players = sorted(average_simulated_player_rankings, key=average_simulated_player_rankings.get)
        ranked_values = [average_simulated_player_rankings[i] for i in ranked_players]
        ranked_positions = [simulated_player_positions[i] for i in ranked_players]
        df_out = pd.DataFrame({'Players':ranked_players,'Positions':ranked_positions,'Ave. Pick Number':ranked_values})

        return df_out

    def write_simulation_results(self,simulation_results,simulated_average_rankings):
        # Pickle Filename
        file_pickle = self.path_sims + self.projection_type + '_' + self.ranking_method + '_' + str(self.number_teams) + '_teams_' + str(self.number_sims) + '_sims'
        pdb.set_trace()
        outfile = open(file_pickle,'wb')
        pickle.dump(simulation_results,outfile)
        outfile.close()

        file_rank = 'Player_Ranking_' + self.projection_type + '_' + self.ranking_method + '_' + str(self.number_teams) + '_teams_' + str(self.number_sims) + '_sims.csv'
        simulated_average_rankings.to_csv(self.path_sims+file_rank)
