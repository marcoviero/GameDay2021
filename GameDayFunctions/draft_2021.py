import pdb
import os
import copy
import numpy as np
import pandas as pd
from datetime import date

pd.options.mode.chained_assignment = None

class Draft:

    def __init__(self, projections_object,
                 draft_position = 2,
                 number_teams = 12,
                 roster_spots = {'C':1,'1B':1,'2B':1, '3B':1,'SS':1,'OF':3,'UTIL':1,'SP':2,'RP':2,'P':3,'BN':1},
                 batter_stats  = ['AB','R','1B','2B', '3B','HR','RBI','SB','BB','AVG','OPS'],
                 pitcher_stats = ['IP','W', 'L','CG','SHO','SV','BB','SO','ERA','WHIP','BSV'],
                 filter_injured_players = True,
                 sigmoid_cut = 1e-8):

        self.number_teams = number_teams
        self.number_rounds = sum(roster_spots.values())
        self.draft_position = draft_position - 1 # e.g., 1st pick is 0!
        self.draft_number = 1
        self.drafted_team = {}
        self.player_projections = projections_object
        self.remaining_ranked_players = projections_object.all_rank
        self.roto_stats_batting = pd.DataFrame(columns =  batter_stats[1:])
        self.roto_stats_pitching = pd.DataFrame(columns =  pitcher_stats[1:])
        self.sigmoid_cut = sigmoid_cut
        self.teams = {}
        # Eventually make this smarter, e.g.;
        # self.roster_spots{"fielders":{'C':1,'1B':1,'2B':1, '3B':1,'SS':1,'OF':3,'UTIL':1},
        #                   "pitchers":{'SP':2,'RP':2,'P':3}
        #                    "bench":{'BN':5}}
        self.roster_spots = roster_spots
        self.fielders = ['C','1B','2B','3B','SS','OF','UTIL']
        self.pitchers = ['SP', 'RP', 'P']
        for i in np.arange(number_teams):
            roto_stats = {}
            roto_stats['batting_stats'] = pd.DataFrame(columns =  batter_stats)
            roto_stats['pitching_stats'] = pd.DataFrame(columns =  pitcher_stats)
            roto_stats['roster_spots'] = roster_spots.copy()
            roto_stats['roster'] = {}
            self.teams[i] = roto_stats

        if filter_injured_players == True:
            self.filter_injured_list(path_list = "Injured_List_Spreadsheets/", injured_list_file = 'Injuries_2021.xlsx')

    # Find Resulting Standings
    def tabulate_roto(self, teams):
        # Determine which stats to include in roto scores
        batting_stat_names = self.roto_stats_batting.columns.values.tolist()
        pitching_stat_names = self.roto_stats_pitching.columns.values.tolist()

        # Estimate batting and pitching seperately and combine later
        roto_stats_batting = self.roto_stats_batting.copy()
        roto_stats_pitching = self.roto_stats_pitching.copy()

        # Estimate Statlines for each team and append to roto_stats_batting/pitching
        for iteam in np.arange(self.number_teams):
            raw_team_batting = teams[iteam]['batting_stats']
            raw_team_pitching = teams[iteam]['pitching_stats']
            roto_team_batting = raw_team_batting.sum()
            roto_team_pitching = raw_team_pitching.sum()

            # Weight Rate Stats by the number of AB or IP
            rate_stats = ['AVG','OPS','ERA','WHIP']
            for istats in batting_stat_names:
                if istats in rate_stats:
                    roto_team_batting[istats] = (raw_team_batting[istats]*raw_team_batting['AB']).sum()/raw_team_batting['AB'].sum()
            for istats in pitching_stat_names:
                if istats in rate_stats:
                    roto_team_pitching[istats] = (raw_team_pitching[istats]*raw_team_pitching['IP']).sum()/raw_team_pitching['IP'].sum()

            roto_stats_batting = roto_stats_batting.append(roto_team_batting[batting_stat_names],ignore_index = True)
            roto_stats_pitching = roto_stats_pitching.append(roto_team_pitching[pitching_stat_names],ignore_index = True)

        # Combine pitching and hitting into single DataFrame.
        roto_team_stats = pd.concat([roto_stats_batting, roto_stats_pitching], axis=1, sort=False)

        # Find Rank in ascending and descending order (suboptimal?)
        roto_standings_desc = pd.concat([roto_stats_batting.rank(ascending=False), roto_stats_pitching.rank(ascending=False).rename(columns={"BB": "BBP"})], axis=1, sort=False)
        roto_standings_ascn = pd.concat([roto_stats_batting.rank(), roto_stats_pitching.rank().rename(columns={"BB": "BBP"})], axis=1, sort=False)

        # Reverse rank of stats in which lower values are better.
        avg_stats = ['L','CS','BBP','ERA','WHIP','BSV']
        for avg_stat in avg_stats:
            if avg_stat in roto_standings_desc:
                roto_standings_ascn[avg_stat] = roto_standings_desc[avg_stat]
        roto_standings = roto_standings_ascn.sum(axis=1).sort_values(ascending=False)
        roto_placement = roto_standings.index.get_loc(self.draft_position) + 1 # standings starting from 1 (not 0)

        return roto_team_stats, roto_stats_batting, roto_stats_pitching, roto_standings, roto_placement, roto_standings_ascn

    def draft_into_teams(self, single_team, drafted_player, position = None, silent = False):
        # Put the drafted_player with specified position into the roster of single_team.

        name = drafted_player.iloc[0].PLAYER
        first_name = name.split(' ')[0]
        last_name = name.split(' ')[1]
        if position == None:
            #pdb.set_trace()
            eligible_positions = drafted_player['Elig. Pos.'].values[0]
            position = self.get_optimal_position(eligible_positions, single_team['roster_spots'])

        # Different Stats Entries for Pitchers and Batters
        #pdb.set_trace()
        if drafted_player['Elig. Pos.'].str.contains('P').bool() == True:
            idx_player = self.player_projections.pitchers_stats.Name.str.contains(first_name+' '+last_name)
            statline = self.player_projections.pitchers_stats[idx_player][single_team['pitching_stats'].keys()]
            single_team['pitching_stats'] = single_team['pitching_stats'].append(statline[0:1])

        else:
            idx_player = self.player_projections.hitters_stats.Name.str.contains(first_name+' '+last_name)
            statline = self.player_projections.hitters_stats[idx_player][single_team['batting_stats'].keys()]
            single_team['batting_stats'] = single_team['batting_stats'].append(statline[0:1])

        # Subtract position spot from roster_spots
        if single_team['roster_spots'][position] > 0:
            single_team['roster_spots'][position] -= 1
            recorded_position = position
        elif (drafted_player['Elig. Pos.'].str.contains('F').bool() == True) and (single_team['roster_spots']['OF'] > 0):
            single_team['roster_spots']['OF'] -= 1
            recorded_position = 'OF'
        elif (drafted_player['Elig. Pos.'].str.contains('P').bool() == True) and (single_team['roster_spots']['P'] > 0):
            single_team['roster_spots']['P'] -= 1
            recorded_position = 'P'
        elif (drafted_player['Elig. Pos.'].str.contains('P').bool() == False) and (single_team['roster_spots']['UTIL'] > 0):
            single_team['roster_spots']['UTIL'] -= 1
            recorded_position = 'UTIL'
        elif (single_team['roster_spots']['BN'] > 0):
            single_team['roster_spots']['BN'] -= 1
            recorded_position = 'BN'
        else:
            pdb.set_trace()
        #print(str(single_team['roster_spots'][recorded_position])+' '+recorded_position+' left')

        if silent == False:
            print('Picked '+ drafted_player.iloc[0].PLAYER + '('+ drafted_player.iloc[0].EligiblePosition +')' +' for ' + recorded_position)

        # Add Player to single_team Roster
        if recorded_position in single_team['roster']:
            single_team['roster'][recorded_position] = [single_team['roster'][recorded_position], drafted_player.PLAYER.values[0:1]]
        else:
            single_team['roster'][recorded_position] = drafted_player.PLAYER.values[0:1]

        return single_team

    def get_optimal_position(self, positions_in, roster_spots):
        # In the event that player is eligible for more than one position, return the optimal position to fill

        # Split players eligible for more that one position, e.g., 1B/OF
        single_positions = positions_in.split('/')

        # Check Pitchers
        P = False
        if ('RP' in single_positions):
            P = True
            if (roster_spots['RP'] > 0): return  'RP'
        if ('SP' in single_positions):
            P = True
            if (roster_spots['SP'] > 0): return  'SP'
        if P == True:
            if (roster_spots['P'] > 0):
                return  'P'
            elif (roster_spots['BN'] > 0):
                return 'BN'
            else:
                return 0

        # Check Hitters
        Util = False
        if ('C' in single_positions) & ('CF' not in single_positions):
            Util = True
            if (roster_spots['C'] > 0): return  'C'
        if ('1B' in single_positions):
            Util = True
            if (roster_spots['1B'] > 0): return  '1B'
        if ('2B' in single_positions):
            Util = True
            if (roster_spots['2B'] > 0): return  '2B'
        if ('OF' in single_positions) | ('LF' in single_positions) | ('CF' in single_positions) | ('RF' in single_positions):
            Util = True
            if (roster_spots['OF'] > 0): return  'OF'
        if ('SS' in single_positions):
            Util = True
            if (roster_spots['SS'] > 0): return  'SS'
        if ('3B' in single_positions):
            Util = True
            if (roster_spots['3B'] > 0): return  '3B'
        if ('Util' in single_positions):
            if (roster_spots['UTIL'] > 0): return  'UTIL'
        if Util == True:
            if (roster_spots['UTIL'] > 0):
                return  'UTIL'
            elif (roster_spots['BN'] > 0):
                return 'BN'
            else:
                print(positions_in)
                print(roster_spots)
                pdb.set_trace()
                return 0
        else:
            pdb.set_trace()
            return 0

    # Do the entire draft one round at a time
    def draft_all(self, naive_draft = False, search_depth = 1, shuffle_picks = False, silent = True):
        for iround in np.arange(self.number_rounds):
            self.teams, self.remaining_ranked_players = self.draft_round(iround, self.teams, self.remaining_ranked_players, naive_draft = naive_draft, shuffle_picks = shuffle_picks, search_depth = search_depth,  silent = silent)
        self.roto_team_stats,self.roto_stats_batting,self.roto_stats_pitching,self.roto_standings,self.roto_placement,self.roto_team_stats_rank = self.tabulate_roto(self.teams)

    # Draft each round one team at a time.  When reaching "draft_position", stop and to pseudo_drafts to figure out best choice.
    def draft_round(self, round_key, teams, df, naive_draft = False, shuffle_picks = False, search_depth = 1, silent = True):

        # Reverse draft order every other round
        draft_order = np.arange(self.number_teams)
        if round_key % 2 == 1:
            draft_order = draft_order[::-1]

        # Makde deep copies so that search for best position does not write to master
        teams_copy = copy.deepcopy(teams)
        df_copy = copy.deepcopy(df)

        # Draft each round one team at a time
        for iteam in draft_order:

            # Defaults to search for best picks at draft_position.  Skips when naive_draft == True
            if naive_draft == False:

                # When team is draft_position, search for best pick.
                if iteam == self.draft_position:
                    best_pick, best_position, best_placement, best_score = self.find_best_pick(iteam, teams_copy, df_copy, round_key, silent = silent, search_depth = search_depth)
                    self.drafted_team[round_key] = df_copy.index[best_pick-1], df_copy.iloc[best_pick-1]['PLAYER'], best_position, best_placement, best_score
                    teams_copy, df_copy = self.draft_next_best(iteam, teams_copy, df_copy, round_key, force_pick = best_pick, force_position = best_position, silent = silent)
                else:
                    teams_copy, df_copy = self.draft_next_best(iteam, teams_copy, df_copy, round_key, shuffle_picks = shuffle_picks, silent = silent)
            else:
                teams_copy, df_copy = self.draft_next_best(iteam, teams_copy, df_copy, round_key, shuffle_picks = shuffle_picks, silent = silent)

        return teams_copy, df_copy

    def draft_remaining(self, teams_copy, df_copy, draft_round,  autodraft_depth = 'end', shuffle_picks = False):
        # Complete draft in Naive mode (i.e., next best picks at available positions)

        if autodraft_depth == 'end':
            remaining_rounds = range(draft_round,self.number_rounds)
        else:
            pdb.set_trace()
            remaining_rounds = range(draft_round,min(self.number_rounds, draft_round + autodraft_depth + 1))
        # Draft all remaining players
        for iround in remaining_rounds:

            draft_order = np.arange(self.number_teams)

            # Reverse draft order every other round
            if iround % 2 == 1:
                draft_order = draft_order[::-1]

            # For the very first round begin at self.draft_position + 1
            if iround == draft_round:
                if iround % 2 == 1:
                    starting_position = self.number_teams - self.draft_position
                    #pdb.set_trace()
                else:
                    starting_position = self.draft_position + 1
                    #pdb.set_trace()
                draft_order = draft_order[starting_position:]

            # Finish the draft by picking the next best player in an open position
            for iteam in draft_order:
                #if sum(teams_copy[iteam]['roster_spots'].values()) == 0:
                #    pdb.set_trace()
                teams_copy, df_copy = self.draft_next_best(iteam, teams_copy, df_copy, iround, shuffle_picks = shuffle_picks)

        return teams_copy, df_copy

    def find_best_pick(self, team_key, teams_copy, df_copy, round_key, search_depth = 1, autodraft_depth = 'end', silent = True):
        # find_best_pick returns iloc, the index (of df) of the optimal pick, and the position being filled

        # Determine which roster_spots are still unfilled
        unfilled_positions = [k for (k,v) in teams_copy[team_key]['roster_spots'].items() if v > 0]
        idx_eligible, pos_eligible = self.idx_unfilled_positions(df_copy, unfilled_positions, search_depth = search_depth)

        #################################
        # START OF LOOP TO FIND BEST PLAYER
        player_based_drafted_outcomes = {}
        player_based_drafted_teams = {}

        #pdb.set_trace()
        # Loop over eligible players, then finish the draft
        n_eligible_positions = len(idx_eligible)
        for iposition, icounter in zip(idx_eligible, range(n_eligible_positions)):

            # make a copy of teams to finish drafting
            teams_loop = copy.deepcopy(teams_copy)
            df_loop = copy.deepcopy(df_copy) #df_copy.copy()

            # Get iplayer before dropping
            iplayer = df_loop.iloc[iposition].PLAYER

            # Prevent picking someone you could easily get in later round
            #pick_ok, pick_number = self.sigmoid_probability_fn(iposition,teams_copy,team_key,df_copy,round_key)
            pick_ok, pick_number = self.projected_pick_floor(iposition,teams_copy,team_key,df_copy,round_key)
            #pdb.set_trace()

            # Draft looping through idx_eligible
            df_loop,drafted_player=df_loop.drop(df_loop.iloc[iposition:iposition+1].index),df_loop.iloc[iposition:iposition+1]
            position = pos_eligible[icounter]

            teams_loop[team_key] = self.draft_into_teams(teams_loop[team_key], drafted_player, position, silent = True)

            # LOOP OVER WHOLE REST OF THE DRAFT HERE...
            teams_loop, df_loop = self.draft_remaining(teams_loop, df_loop, round_key, autodraft_depth = autodraft_depth)

            # Calculate the best pseudo-standings
            #pseudo_team_stats, pseudo_batting_stats, pseudo_pitching_stats, pseudo_standings, pseudo_placement = self.tabulate_roto(teams_loop)
            roto_stats = self.tabulate_roto(teams_loop)
            # [0] = roto_team_stats
            # [1] = roto_stats_batting
            # [2] = roto_stats_pitching
            # [3] = roto_standings
            # [4] = roto_placement
            # [5] = roto_standings_ascn

            # Store the result.
            #pdb.set_trace()
            keep_picking = True
            if (pick_ok == True) or (n_eligible_positions < 2):
                player_based_drafted_teams[iplayer] = teams_loop[self.draft_position]['roster']
                player_based_drafted_outcomes[iplayer] = [roto_stats[4],roto_stats[3][self.draft_position]] #[roto_stats[4],roto_stats[3][roto_stats[4]-1]]
                if silent == False:
                    print('Stored Result for Pick '+str(icounter)+' ['+str(pick_number)+'/'+str(drafted_player.index[0])+'] '+iplayer+' '+pos_eligible[icounter]+' whose placement/score is '+str(roto_stats[4])+'/'+str(roto_stats[3][self.draft_position]))
                    #pdb.set_trace()
            else:
                if silent == False:
                    print('Not Storing Result for Pick '+str(icounter)+' ['+str(pick_number)+'/'+str(drafted_player.index[0])+'] '+iplayer+' '+pos_eligible[icounter])
                    #pdb.set_trace()
                    keep_picking = False
            if keep_picking == False:
                break

            #pdb.set_trace()

        # Pick best of the bunch
        #pdb.set_trace()
        best_pick_plus_one, best_position, best_player, best_placement, best_score = self.decide_best_choice(df_copy, player_based_drafted_teams, player_based_drafted_outcomes,unfilled_positions, idx_eligible, pos_eligible, silent=silent)
        return best_pick_plus_one, best_position, best_placement, best_score
        # END OF LOOP TO FIND BEST PLAYER
        #################################

    def idx_unfilled_positions(self,df_copy, unfilled_positions0, search_depth = 1):
        # Identify positions that still need filling, taking into account that UTIL
        # can be filled by any batting position and so should be saved for last,
        # and that SP/RP should be filled before P.
        idx_eligible = []
        pos_eligible = []

        # First check unfilled_positions0 for 'BN' OR ('UTIL' AND 'P'); if true
        # unfilled_positions is all self.fielders + self.pitchers, minus 'UTIL' and 'P'
        if ('BN' in unfilled_positions0) or (('UTIL' in unfilled_positions0) and ('P' in unfilled_positions0)):
            unfilled_positions = self.fielders + self.pitchers
            unfilled_positions.remove('UTIL')
            unfilled_positions.remove('P')
        elif 'UTIL' in unfilled_positions0:
            unfilled_positions = np.unique(unfilled_positions0 + self.fielders).tolist()
            if unfilled_positions0 != 'UTIL':
                unfilled_positions.remove('UTIL')
        elif 'P' in unfilled_positions0:
            unfilled_positions = np.unique(unfilled_positions0 + self.pitchers).tolist()
            if unfilled_positions0 != 'P':
                unfilled_positions.remove('P')
        else:
            unfilled_positions = unfilled_positions0

        # Find index of best player at each remaining position
        filled_position_counter = np.ones(len(unfilled_positions)) * search_depth
        for iunfilled, icounter in zip(unfilled_positions, range(len(filled_position_counter))):
            if iunfilled == 'OF':
                punfilled = 'F'
            else:
                punfilled = iunfilled
            if iunfilled == 'C':
                #pdb.set_trace()
                #idx_position = [i for i, val in enumerate(df_copy['Elig. Pos.'].str.contains('C') & !df_copy['Elig. Pos.'].str.contains('CF')) if val]
                idx_position = [i for i, val in enumerate((df_copy['Elig. Pos.'].str.contains('C')==True) & (df_copy['Elig. Pos.'].str.contains('CF')==False)) if val]
            else:
                idx_position = [i for i, val in enumerate(df_copy['Elig. Pos.'].str.contains(punfilled)) if val]
            jdx = 0
            while filled_position_counter[icounter] > 0:
                if jdx == len(idx_position):
                    pdb.set_trace()
                if idx_position[jdx] in idx_eligible:
                    jdx+=1
                else:
                    idx_eligible.append(idx_position[jdx])
                    pos_eligible.append(iunfilled)
                    filled_position_counter[icounter] -=  1

        # Get rid of doubles (1B and OF is particularly prone)
        idx_eligible, idx_unique = np.unique(idx_eligible, return_index = True)
        pos_eligible = [pos_eligible[i] for i in idx_unique]
        #pdb.set_trace()
        return idx_eligible, pos_eligible

    def decide_best_choice(self, df_copy, player_based_drafted_teams, player_based_drafted_outcomes, unfilled_positions, idx_eligible, pos_eligible, rank_type = 'placement', silent = True):
        # End of Loop
        ranked_positions = ['C','1B','2B','OF','SS','3B','SP','RP','UTIL','P','BN']

        # Decide on best choice and return
        relative_ranking = [player_based_drafted_outcomes[i][0] for i in player_based_drafted_outcomes]
        relative_ranking_rank = np.argsort(relative_ranking)
        relative_scores = [player_based_drafted_outcomes[i][1] for i in player_based_drafted_outcomes]

        # If there is a tie for top relative_ranking, select by highest score, then optimal position
        n_max_ranking = sum(relative_ranking == np.min(relative_ranking))

        if n_max_ranking == 1:
            best_player = df_copy.iloc[idx_eligible[relative_ranking_rank[0]]:idx_eligible[relative_ranking_rank[0]]+1]
            best_pick_plus_one = idx_eligible[relative_ranking_rank[0]] + 1 # Avoid best_pick = 0
            best_position = pos_eligible[relative_ranking_rank[0]]
            best_placement = relative_ranking[relative_ranking_rank[0]]
            best_score = relative_scores[relative_ranking_rank[0]]
        else:
            # Of those tied for top rank, figure out had a highest score
            best_player_scores = [relative_scores[relative_ranking_rank[i]] for i in range(n_max_ranking)]
            best_players = [df_copy.iloc[idx_eligible[relative_ranking_rank[i]]] for i in range(n_max_ranking)]
            best_picks_plus_one = [idx_eligible[relative_ranking_rank[i]] + 1 for i in range(n_max_ranking)]
            best_player_positions = [pos_eligible[relative_ranking_rank[i]] for i in range(n_max_ranking)]
            best_player_placements = [relative_ranking[relative_ranking_rank[i]] for i in range(n_max_ranking)]
            idx_best_player_scores = np.argsort(best_player_scores)[::-1]
            #pdb.set_trace()
            # If still tied, take the optimal position (i.e., SS over OF)
            n_max_scores = sum(best_player_scores == np.max(best_player_scores))
            if n_max_scores == 1:
                best_player = df_copy.iloc[best_picks_plus_one[idx_best_player_scores[0]]-1:best_picks_plus_one[idx_best_player_scores[0]]-1+1]
                best_pick_plus_one = best_picks_plus_one[idx_best_player_scores[0]]
                best_position = best_player_positions[idx_best_player_scores[0]]
                best_placement = best_player_placements[idx_best_player_scores[0]]
                best_score = best_player_scores[idx_best_player_scores[0]]
            else:
                best_player_positions = [pos_eligible[idx_best_player_scores[i]] for i in range(n_max_scores)]
                for irank in range(len(ranked_positions)):
                    if any(ranked_positions[irank] in s for s in best_player_positions):
                        idx_best = best_player_positions.index(ranked_positions[irank])
                        best_player = df_copy.iloc[best_picks_plus_one[idx_best_player_scores[idx_best]]-1:best_picks_plus_one[idx_best_player_scores[idx_best]]-1+1]
                        best_pick_plus_one = best_picks_plus_one[idx_best_player_scores[idx_best]]
                        best_position = ranked_positions[irank]
                        best_placement = best_player_placements[idx_best_player_scores[idx_best]]
                        best_score = best_player_scores[idx_best_player_scores[idx_best]]
                        break

        return best_pick_plus_one, best_position, best_player, best_placement, best_score

    # Strategy is to take the best possible player, even if that means putting them in UTIL or BN (maybe BN should reconsidered...)
    def draft_next_best(self, team_key, teams, df, round_number, force_pick = False, force_position = False, shuffle_picks = False, search_depth = 1, silent = True):

        if (force_pick == False):

            # Get rid of the bench until the final rounds.
            teams_minus_bench = teams[team_key]['roster_spots'].copy()
            if 'BN' in teams_minus_bench:
                if sum(teams_minus_bench.values()) > 1:
                    del teams_minus_bench['BN']

            # Find unfilled positions and their indices
            unfilled_positions = [k for (k,v) in teams_minus_bench.items() if v > 0]
            idx_eligible, pos_eligible = self.idx_unfilled_positions(df, unfilled_positions, search_depth = search_depth)
            #pdb.set_trace()

            #idx_shuffle = np.arange(len(idx_eligible))
            if shuffle_picks == True:
                #pdb.set_trace()
                if round_number % 2 == 1:
                    pick_number = 0 + (round_number * self.number_teams) + (self.number_teams - team_key)
                else:
                    pick_number = 1 + (round_number * self.number_teams) + team_key
                #idx_eligible = idx_eligible[np.argsort((1 / (1 + np.exp(-(pick_number - np.random.normal(df.iloc[idx_eligible]['AVG'], df.iloc[idx_eligible]['STD DEV']))))))[::-1]]
                idx_eligible = idx_eligible[np.argsort((1 / (1 + np.exp(-(pick_number - np.random.normal(df.iloc[idx_eligible]['AVG'], df.iloc[idx_eligible]['STD DEV']))/df.iloc[idx_eligible]['STD DEV']))))[::-1]]
                #np.random.shuffle(idx_shuffle)
                #idx_eligible = idx_eligible[idx_shuffle]
                #pos_eligible = [pos_eligible[x] for x in idx_shuffle]
                #pdb.set_trace()

            # Draft next in list by making indices of unfilled_positions and taking first (or shuffle)
            #try:
            #    df,drafted_player=df.drop(df.iloc[idx_eligible[0]:idx_eligible[0]+1].index),df.iloc[idx_eligible[0]:idx_eligible[0]+1]

            #    if sum(teams[team_key]['roster_spots'].values()) > 0:
            #        teams[team_key] = self.draft_into_teams(teams[team_key], drafted_player, silent = True)
            #        if silent == False:
            #            print('Team '+ str(team_key+1) +' Drafting '+drafted_player.iloc[0].PLAYER)
            #except:
            #    pdb.set_trace()

            if sum(teams[team_key]['roster_spots'].values()) > 0:
                df,drafted_player=df.drop(df.iloc[idx_eligible[0]:idx_eligible[0]+1].index),df.iloc[idx_eligible[0]:idx_eligible[0]+1]

                teams[team_key] = self.draft_into_teams(teams[team_key], drafted_player, silent = True)
                if silent == False:
                    print('Team '+ str(team_key+1) +' Drafting '+drafted_player.iloc[0].PLAYER)

        else:

            pick = force_pick - 1
            df,drafted_player=df.drop(df.iloc[pick:pick+1].index),df.iloc[pick:pick+1]

            teams[team_key] = self.draft_into_teams(teams[team_key], drafted_player, position = force_position, silent = True)
            if silent == False:
                print('Team '+ str(team_key+1) +' picking '+drafted_player.iloc[0].PLAYER+' for '+force_position)

        return teams, df

    def print_draft_positions(self, draft_position, number_players = None):
        nrounds = sum(self.roster_spots.values())
        if number_players == None:
            number_players = nrounds * number_teams

        cnt = 0
        draft_positions = []
        while cnt < number_players:
            for iround in np.arange(nrounds):
                # Reverse draft order every other round
                draft_order = np.arange(1,self.number_teams+1)
                if iround % 2 == 1:
                    draft_order = draft_order[::-1]

                for i in np.arange(self.number_teams):
                    draft_positions.append(draft_order[i])
                    cnt = cnt + 1
                    if cnt == number_players:
                        break
                if cnt == number_players:
                        break

        return draft_positions

    def start_new_draft(self,path_draft='Draft_Pick_Spreadsheets/',
                        path_model='projections/2021/PositionalRankings/',
                        model='FantasyPros', file_date=None,year=2021, overwrite=False):
        file = model + '_Roto_Ranking_' + str(year) + '.xlsx'
        df = pd.read_excel(os.path.join(path_model, model, file))
        dp = self.print_draft_positions(self.draft_position, number_players=len(df))
        df['Team'] = dp
        df['Picks'] = ''
        #df = df[['RANK', 'Team', 'Picks', 'PLAYER', 'AVG', 'STD DEV', 'Elig. Pos.']]
        #df = df[['Team','RANK', 'AVG', 'STD DEV', 'Picks', 'PLAYER', 'Elig. Pos.' ]]
        df = df[['Picks', 'Team', 'PLAYER', 'Elig. Pos.','AVG', 'STD DEV' ]]

        if file_date == None:
            today = date.today()
            suf = '_'+model+'_'+today.strftime("%m%d%Y")
        else:
            suf = '_'+model+file_date
        ending = '.xlsx'
        fileout = 'LiveDraft_' + suf + '.xlsx'
        if os.path.isfile(os.path.join(path_draft, fileout)) == False or overwrite == True:
            df.to_excel(os.path.join(path_draft, fileout), index=False)

        return fileout

    def draft_from_list_and_find_best_pick(self, search_depth = 1, autodraft_depth = 'end', path_list = 'Draft_Pick_Spreadsheets/', draft_pick_file = 'TestPicks.xlsx', shuffle_picks = False, silent = False):
        # Read in Excel Sheet and draft picks before moving on to finishing script

        # Read current draft results
        xls = pd.ExcelFile(os.path.join(path_list,draft_pick_file))
        #pdb.set_trace()
        #complete_player_list = pd.read_excel(xls, skiprows = 0, names = ['Pick','PLAYER','EligiblePosition'], index_col = 'Pick')
        #complete_player_list = pd.read_excel(xls, skiprows = 0, names = ['RANK','Picks','PLAYER','AVG','STD DEV','Elig. Pos.'], index_col = 'Picks')
        complete_player_list = pd.read_excel(xls,index_col='Pick')
        # Remove players from list who don't have numbers (i.e., NANs)
        #pdb.set_trace()
        player_list = complete_player_list.loc[complete_player_list.index.dropna().values].sort_index()

        # Replace non-latin characters so lists agree
        for irename in range(len(player_list)):
            player_list.iloc[irename]['PLAYER'] = standardize_name(player_list.iloc[irename]['PLAYER'])

        teams_copy = copy.deepcopy(self.teams)
        df_copy = copy.deepcopy(self.remaining_ranked_players)

        for iround in np.arange(self.number_rounds):

            # Reverse draft order every other round
            draft_order = np.arange(self.number_teams)
            iter_team = 1
            if iround % 2 == 1:
                draft_order = draft_order[::-1]
                iter_team = -1

            for iteam in draft_order:
                #print('Drafting Team '+str(iteam+1))
                # Find player matching df_copy by iloc
                idx_match = [i for i, x in enumerate(df_copy['PLAYER'].str.match(player_list.PLAYER.iloc[0])) if x]
                player_list, drafted_player = player_list.drop(player_list.iloc[0:1].index), player_list.iloc[0]
                #pdb.set_trace()
                #best_position = self.get_optimal_position(drafted_player['EligiblePosition'], teams_copy[iteam]['roster_spots'])
                best_position = self.get_optimal_position(drafted_player['Elig. Pos.'], teams_copy[iteam]['roster_spots'])
                teams_copy, df_copy = self.draft_next_best(iteam, teams_copy, df_copy, iround, force_pick = idx_match[0] + 1, force_position = best_position)
                if len(player_list) == 0:
                    break
            if len(player_list) == 0:
                break

        #pdb.set_trace()
        # Find best pick
        if silent == False:
            print('Finding Best Pick For Team '+str(iteam+1+iter_team))

        best_pick, best_position, best_placement, best_score = self.find_best_pick(iteam+iter_team,copy.deepcopy(teams_copy),copy.deepcopy(df_copy),iround,silent=False,autodraft_depth = autodraft_depth, search_depth = search_depth)
        best_player_this_round = df_copy.iloc[best_pick-1].PLAYER
        teams_copy, df_copy = self.draft_next_best(iteam+iter_team, teams_copy, df_copy, iround, force_pick = best_pick, force_position = best_position)

        # Finish the draft and Rank
        teams_copy, df_copy = self.draft_remaining(teams_copy, df_copy, iround, autodraft_depth = autodraft_depth, shuffle_picks = shuffle_picks)

        # Calculate the best pseudo-standings
        roto_stats = self.tabulate_roto(teams_copy)
        if silent == False:
            print('Best Pick is ' + best_player_this_round+ ' putting you in ' + str(roto_stats[4]) + ' place')

        # Return Player Name and Projected Roto Stats
        return best_player_this_round, roto_stats

    def filter_injured_list(self, path_list = "Injured_List_Spreadsheets/", injured_list_file = 'Injuries_2021.xlsx'):
        # Read in Excel Sheet of Players to Exclude.  Should this be moved to Projection?  Yes.


        xls = pd.ExcelFile(os.path.join(path_list,injured_list_file))
        #pdb.set_trace()
        #injured_list = pd.read_excel(xls, skiprows =0, names = ['PLAYER','Elig. Pos.'])#, index_col = 'PLAYER')
        injured_list = pd.read_excel(xls, skiprows =0)#, index_col = 'PLAYER')
        #xls = pd.ExcelFile(os.path.join(path_list,draft_pick_file))
        #complete_player_list = pd.read_excel(xls, skiprows =0, names = ['Pick','PLAYER','EligiblePosition'], index_col = 'Pick')
        player_list = injured_list.loc[injured_list.index.dropna().values]

        for irename in range(len(player_list)):
            # Standardize name
            name = standardize_name(player_list.iloc[irename]['Name'])
            first_name = name.split(' ')[0]
            last_name = name.split(' ')[1]

            # Find matching player
            idx_match = [i for i, x in enumerate(self.remaining_ranked_players['PLAYER'].str.contains(first_name+'&'+last_name)) if x]
            # Remove from self.remaining_ranked_players
            self.remaining_ranked_players = self.remaining_ranked_players.drop(index=self.remaining_ranked_players.index[idx_match])

    def projected_pick_floor(self,iposition,teams_in,team_key_in,df_in,round_number):

        if round_number % 2 == 1:
            pick_number = 0+(round_number * self.number_teams) + (self.number_teams - team_key_in)
            next_pick_number = 1 + ((round_number+1) * self.number_teams) + team_key_in
        else:
            pick_number = 1+(round_number * self.number_teams) + team_key_in
            next_pick_number = 0+((round_number+1) * self.number_teams) + (self.number_teams - team_key_in)

        possible_pick = df_in.iloc[iposition:iposition+1]
        # Is the next pick number larger than lowest possible?
        pick_ok = next_pick_number > (possible_pick['AVG'] - possible_pick['STD DEV']).values[0]

        return pick_ok, pick_number

    def sigmoid_probability_fn(self,iposition,teams_in,team_key_in,df_in,round_number):
        pick_ok = True

        if round_number % 2 == 1:
            pick_number = 0+(round_number * self.number_teams) + (self.number_teams - team_key_in)
        else:
            pick_number = 1+(round_number * self.number_teams) + team_key_in

        # Use a sigmoid to get probability of drafting.  E.g.; if pick is usually taken
        # 5 and this is the 2nd pick, probability is low, but if is 15th pick, then
        # probability is near 1.
        # 1÷(1+EXP(−(pick_number−avg_pick_number)))
        possible_pick = df_in.iloc[iposition:iposition+1]
        if 'AVE' in possible_pick:
            sigmoid_probability = (1/(1+np.exp(-(pick_number-possible_pick['AVE'])/np.sqrt(pick_number))))
            sigmoid_probability = sigmoid_probability.values[0]
        else:
            sigmoid_probability = (1/(1+np.exp(-(pick_number-possible_pick.index[0])/np.sqrt(pick_number))))

        if sigmoid_probability < self.sigmoid_cut:
            #print(pick_number)
            #print(possible_pick)
            pick_ok = False

        return pick_ok, pick_number

def standardize_name(name_in):
    name_out = ((((((name_in.replace('ñ','n')).replace('í','i')).replace('é','e')).replace('á','a')).replace('ú','u')).replace('ó','o')).split(' Jr.')
    return name_out[0]
