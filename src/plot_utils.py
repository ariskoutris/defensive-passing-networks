import pandas as pd
import matplotlib.pyplot as plt

class PassPlotter:

    def __init__(self, passes_df, frame_id, pitch_length, pitch_width) -> None:
        
        mask = passes_df['frame'] == frame_id
        self.pass_df = passes_df[mask]
        
        self.play_direction = self.pass_df['play_direction'].values[0]
        self.passer_id = self.pass_df['player.id.skillcorner'].values[0]
        self.passer_x, self.passer_y = self.pass_df[self.pass_df['tracking.object_id'] == self.passer_id][['tracking.x', 'tracking.y']].values[0]
        self.receiver_id = self.pass_df['pass.recipient.id.skillcorner'].values[0]
        self.receiver_x, self.receiver_y = self.pass_df[self.pass_df['tracking.object_id'] == self.receiver_id][['tracking.x', 'tracking.y']].values[0]

        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def get_player_location(self, player_id):
        defender_data = self.pass_df[self.pass_df['tracking.object_id'] == player_id]
        return defender_data[['tracking.x', 'tracking.y']].values[0]

    def get_attacker_locations(self, include_passer=False, include_ids=False):
        mask = self.pass_df['tracking.is_teammate']
        if not include_passer:
            mask &= ~self.pass_df['tracking.is_self']        
        attackers_data = self.pass_df[mask]
        if include_ids:
            return attackers_data[['tracking.x', 'tracking.y']].values, attackers_data['tracking.object_id'].values
        return attackers_data[['tracking.x', 'tracking.y']].values

    def get_defender_locations(self, include_ids=False):
        defenders_data = self.pass_df[self.pass_df['tracking.is_opponent']]
        if include_ids:
            return defenders_data[['tracking.x', 'tracking.y']].values, defenders_data['tracking.object_id'].values
        return defenders_data[['tracking.x', 'tracking.y']].values

    def plot_players(self):

        _, ax = plt.subplots()
        ax.set_xlim([-self.pitch_length/2, self.pitch_length/2])
        ax.set_ylim([-self.pitch_width/2, self.pitch_width/2])

        labels = {'Passer': None, 'Teammate': None, 'Opponent': None}
        for _, player in self.pass_df.iterrows():
            x, y = player['tracking.x'], player['tracking.y']
            if player['tracking.is_self']:
                color = 'green'
                marker = 'D'
                label = 'Passer'
            elif player['tracking.is_teammate']:
                color = 'blue'
                marker = 'D'
                label = 'Teammate'
            elif player['tracking.is_opponent']:
                color = 'red'
                marker = 'o'
                label = 'Opponent'
            else:
                continue
            scatter = ax.scatter(x, y, color=color, marker=marker)
            if player['tracking.is_opponent']:
                ax.annotate(player['tracking.object_id'], (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            if labels[label] is None:
                labels[label] = scatter

        if self.play_direction == 'TOP_TO_BOTTOM':
            ax.arrow(self.pitch_length/2, 0, -self.pitch_length/32, 0, head_width=2, head_length=4, fc='k', ec='k')
        else:
            ax.arrow(-self.pitch_length/2, 0, self.pitch_length/32, 0, head_width=2, head_length=4, fc='k', ec='k')

        ax.legend(labels.values(), labels.keys())
        plt.show()
        
    def plot_pass(self, ax, defender_id=None, max_threat_receiver=None, optimal_location=None): 

        # Plot true pass
        ax.plot([self.passer_x, self.receiver_x], [self.passer_y, self.receiver_y], color='black', linestyle='dashed', label='True Pass', linewidth=1)

        # Plot attackers
        attackers = self.get_attacker_locations(include_passer=False)
        ax.scatter(attackers[:,0], attackers[:,1], color='red', marker='D', label='Attackers')

        # Plot defenders
        defenders = self.get_defender_locations()
        ax.scatter(defenders[:,0], defenders[:,1], color='blue', label='Defenders')

        # Plot passer
        ax.scatter(self.passer_x, self.passer_y, color='red', marker='D', edgecolors='black', linewidth=2, s=50, label='Passer')

        # Plot target defender
        if defender_id:
            x_def, y_def = self.get_player_location(defender_id)
            ax.scatter(x_def, y_def, color='blue', marker='o', edgecolors='black', linewidth=2, s=50, label='Target Defender')
            
        # Plot best receiver option
        if max_threat_receiver:
            x_best, y_best = self.pass_df[self.pass_df['tracking.object_id'] == max_threat_receiver][['tracking.x', 'tracking.y']].values[0]
            ax.plot([self.passer_x, x_best], [self.passer_y, y_best], color='gray', linestyle='dashed', label='Best Pass', linewidth=1)

        # Plot optimal defender position
        if optimal_location:
            x_opt, y_opt = optimal_location
            ax.scatter(x_opt, y_opt, color='purple', marker='x', label='Optimal Location')
        
        return ax
