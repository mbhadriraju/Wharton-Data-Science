import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
'''
    Using a weighted average of all the previous game stats
    as the regression parameters
'''

df = pd.read_csv("Data/games_2022.csv")

df = df.sort_values(by="game_date")

# To feature engineer, we are going to initialize a dictionary with all the teams as keys and compute the weighted average of each entry in the dictionary by multiplying
# the current value by some proportion and the observed value by 1- that proportion.

labels=('notD1_incomplete', 'OT_length_min_tot', 'tz_dif_H_E', 'prev_game_dist', 'home_away', 'largest_lead', 'TOV_team', 'game_date', 'game_id')

for label in labels:
    df = df.drop(labels=label, axis=1)

df['FGM_2'] = df['FGM_2'].div(df['FGA_2']).fillna(0)
df['FGM_3'] = df['FGM_3'].div(df['FGA_3']).fillna(0)
df['FTM'] = df['FTM'].div(df['FTA']).fillna(0)

df['rest_days'] = df['rest_days'].fillna(0)
df['attendance'] = df['attendance'].fillna(0)
df['travel_dist'] = df['travel_dist'].fillna(0)

'''
correlation matrix to observe correlations between variables

correlation_df = df.drop(columns='team')

corr_matrix = np.corrcoef(np.array(correlation_df).transpose())


plt.imshow(corr_matrix, cmap='autumn')

plt.colorbar()
plt.show()
'''


y_vals = np.array([1 if df["team_score"][i] > df["opponent_team_score"][i] else 0 for i in range(len(df))])

x_vals = []

averages = np.array([np.average(df[df.columns[i]]) for i in range(1, 17)])

data_dict = {}

ma_weight = 0.5 # Weight for weighted average calculation

for i in range(len(df)):
    if df.iloc[i]['team'] not in data_dict.keys():
        data_dict[df.iloc[i]['team']] = np.array(df.iloc[i][1:17])
        temp = averages.copy()
        temp = list(np.append(temp, np.array(df.iloc[i][17:])))
        x_vals.append(temp)
    else:
        new_arr = np.array(df.iloc[i][1:17])
        temp = data_dict[df.iloc[i]['team']]
        temp = list(np.append(temp, np.array(df.iloc[i][17:])))
        x_vals.append(temp)
        data_dict[df.iloc[i]['team']] = (ma_weight * new_arr) + ((1 - ma_weight) * data_dict[df.iloc[i]['team']])
'''
The plan is to implement a Bayesian logistic regression model. 
Some resources: 
    https://towardsdatascience.com/using-bayesian-modeling-to-predict-the-champions-league-8ebb069006ba/
    https://medium.com/@adilsaid64/predicting-premier-league-match-wins-using-bayesian-modelling-32eec733472e
    https://bayesball.github.io/BOOK/bayesian-multiple-regression-and-logistic-models.html#example-u.s.-women-labor-participation
Bayes Theorem:
    P(B | y) = (P(y | B) * P(B)) / P(y)

    P(B | y) is the posterior 

    P(y | B) is the likelihood

    P(B) is the prior
Steps:
    1. Make prior for beta vectors
    2. Specify likelihood function
    3. Update posterior using Markov Chain Monte Carlo
    4. Use statistics from the posterior distribution for predictions

'''


