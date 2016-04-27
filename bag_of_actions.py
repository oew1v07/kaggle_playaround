import pandas as pd

session = pd.read_csv("sessions.csv")

grouped_type = session.groupby(by=["action_type", "user_id"])

# get a list of all the action_type, action_detail etc.
action_types = session.action_type.unique()[1:]

action_details = session.action_detail.unique()[1:]

action = session.action.unique()[1:]

users_uni = session.user_id.unique()

# For action_type

act_type = pd.DataFrame(index=session.user_id.unique(), columns=session.action_type.unique())

act_type.fillna(value = 0, inplace=True)
for name, group in grouped_type:
	n = len(group)

	act_type[name[0]].ix[name[1]] = n

act_type.to_csv("User_bag_of_action_type.csv")

# For action_detail

grouped_detail = session.groupby(by=["action_detail", "user_id"])

act_detail = pd.DataFrame(index=users_uni, columns=action_details)

act_detail.fillna(value = 0, inplace=True)

for name, group in grouped_detail:
	n = len(group)

	act_detail[name[0]].ix[name[1]] = n

act_detail.to_csv("User_bag_of_action_detail.csv")

# For action

grouped = session.groupby(by=["action", "user_id"])

actions = pd.DataFrame(index=users_uni, columns=action)

actions.fillna(value = 0, inplace=True)

for name, group in grouped:
	n = len(group)

	actions[name[0]].ix[name[1]] = n

actions.to_csv("User_bag_of_actions.csv")

