#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from fuzzywuzzy import fuzz
import pandas as pd
import sys
from tqdm import tqdm
import numpy as np
import seaborn as sns
import pickle


def make_histogram(data, column, lable, xlab, ylab, sort_index=False):

	fig, ax = plt.subplots(figsize=(14, 7))
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.set_title(lable, fontsize=24, pad=20)
	ax.set_xlabel(xlab, fontsize=16, labelpad=20)
	ax.set_ylabel(ylab, fontsize=16, labelpad=20)
	plt.hist(data[column], bins=25, color='#3498db', ec='#2980b9', linewidth=2)
	plt.xticks(rotation=45)

	return 0


def plot_pi_chart():
	col = "type"
	grouped = df[col].value_counts().reset_index()
	grouped = grouped.rename(columns = {col : "count", "index" : col})
	trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0], marker=dict(colors=["#6ad49b", "#a678de"]))
	layout = go.Layout(title="", height=400, legend=dict(x=0.1, y=1.1))
	fig = go.Figure(data = [trace], layout = layout)
	# iplot(fig)
	fig.show()
	return 0


def plot_heatmap(data):

	data = data["listed_in"].astype(str).apply(lambda s : s.replace('&',' ').replace(',', ' ').split())
	mlb = MultiLabelBinarizer()
	res = pd.DataFrame(mlb.fit_transform(data), columns=mlb.classes_)
	corr = res.corr()
	mask = np.zeros_like(corr, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True
	f, ax = plt.subplots(figsize=(35, 34))
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
	plt.show()

	return 0


def make_histogram(data, column, lable, xlab, ylab, sort_index=False):

	fig, ax = plt.subplots(figsize=(14, 7))
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.set_title(lable, fontsize=24, pad=20)
	ax.set_xlabel(xlab, fontsize=16, labelpad=20)
	ax.set_ylabel(ylab, fontsize=16, labelpad=20)
	plt.hist(data[column], bins=25, color='#3498db', ec='#2980b9', linewidth=2)
	plt.xticks(rotation=45)
	plt.show()

	return 0


def read_data():

	data = pd.read_csv("./res/netflix_titles.csv")
	# data.isnull().sum()
	# data.info()
	return data


def process_data(data, remove_base):

	data.dropna(inplace=True)
	data.reset_index(drop=True, inplace=True)
	data["key_words"] = None
	print("Creating Key Words from Discription...")
	for idx in tqdm(range(len(data))):
		# idx = 0
		r_obj = Rake()
		r_obj.extract_keywords_from_text(data.loc[idx, "description"])
		# data.loc[idx, "key_words"] = list(r_obj.get_word_degrees().keys())
		data.at[idx, "key_words"] = list(r_obj.get_word_degrees().keys())

	print("Processing columns...")
	data["cast"] = data["cast"].map(lambda x: x.lower().split(','))
	data["listed_in"] = data["listed_in"].map(lambda x: x.lower().split(','))
	data["director"] = data["director"].map(lambda x: x.lower().split(','))
	data["country"] = data["country"].map(lambda x: x.lower().split(','))

	for idx in tqdm(range(len(data))):
		# idx = 0
		data.at[idx, "cast"] = [x.lower().replace(' ','') for x in data.loc[idx, "cast"]]
		data.at[idx, "director"] = "".join(data.loc[idx, "director"]).lower()

	data["bag_of_words"] = None
	columns = data[["type", "title", "director", "cast", "country", "release_year", "rating", "duration", "listed_in", "key_words"]].columns

	print("Creating Bag of Words...")
	for idx in tqdm(range(len(data))):
		# idx = 0
		words = ""
		for col in columns:
			# col = columns[3]
			if type(data.loc[idx, col]) == type(""):
				words = words+data.loc[idx, col]+" "
				pass
			elif type(data.loc[idx, col]) == type([]):
				words = words+" ".join(data.loc[idx, col])+" "
			else:
				words = words+str(data.loc[idx, col])+" "
		data.at[idx, "bag_of_words"] = words

	data.set_index("title", inplace = True)
	if remove_base:
		data.drop(columns = [col for col in data.columns if col!="bag_of_words"], inplace = True)

	return data


def recommendations(movie_title, cosine_sim, indices):

	# movie_title = "the gun"
	# movie_title = "skeleton key"
	data = pd.DataFrame()
	recommended_movies = []
	idx = -1
	if len(indices[indices == movie_title])>0:
		idx = indices[indices == movie_title].index[0]
	else:
		data["title"] = indices
		data["ratio"] = None
		for idx in range(len(data)):
			data.at[idx,"ratio"] = fuzz.ratio(data.loc[idx,"title"], movie_title)
		# do string matching to find idx
		idx = data[data["ratio"]==data.ratio.max()].index.to_list()[0]

	score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
	top_10_indexes = list(score_series.iloc[1:11].index)
	for i in top_10_indexes:
		recommended_movies.append(list(data.title)[i])
	return recommended_movies


def extract_features(data):

	count = CountVectorizer()
	count_matrix = count.fit_transform(data["bag_of_words"])
	indices = pd.Series(data.index)
	cosine_sim = cosine_similarity(count_matrix, count_matrix)

	indices.to_pickle("./out/indices.pkl")
	pickle.dump(cosine_sim, open("./out/cosine_sim.pkl", "wb"))
	return indices, cosine_sim


def read_index_cosine():

	indices = pd.read_pickle("./out/indices.pkl")
	cosine_sim = pickle.load(open("./out/cosine_sim.pkl", "rb"))
	return indices, cosine_sim


def display(recommended_movies):

	for idx, movie in enumerate(recommended_movies):
		print(idx, "::", movie)

	return 0

def functionality():

	indices, cosine_sim = read_index_cosine()
	while True:
		try:
			text = input("Enter Any Movie You Liked: ")
			recommended_movies = recommendations(text, cosine_sim, indices)
			display(recommended_movies)
		except Exception as ex:
			print({"0".format(ex)})
			continue


def plot_data():
	data = read_data()
	# data.info()
	make_histogram(data, "release_year", "Movies Published Per Year", "Year", "Counts")
	plot_heatmap(data)
	data = process_data(data, True)
	indices, cosine_sim = extract_features(data)

	return 0


def main():

	if len(sys.argv)>1:
		if sys.argv[1]=="1":
			plot_data()
		else:
			functionality()
	else:
		print("Please Provide Run Mode by Passing 1 or 0")

	return 0


if __name__ == '__main__':
	main()
