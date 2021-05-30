import praw
import reticker
import statistics
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# creating reticker object to fetch mentions of stock symbols
extractTick = reticker.TickerExtractor()
type(extractTick.pattern)


def mentionTableCreate(reddit, url):
    redThread = reddit.submission(url=url)
    redThread.comments.replace_more(limit=40)

    mentionData = {'Symbols Mentioned': [], 'Sentiment':[], 'Comment': []}
    for index, comment in enumerate(redThread.comments):
        extractedSymbolList = extractTick.extract(str(comment.body))

        if(len(extractedSymbolList) > 0 and not (str(comment.body)).isupper()
                and sentimentAnalysis(str(comment.body)) != 0):
            mentionData['Symbols Mentioned'].append(extractTick.extract(str(comment.body)))
            mentionData['Sentiment'].append((sentimentAnalysis(str(comment.body))))
            mentionData['Comment'].append(str(comment.body))

    mentionTable = pd.DataFrame(mentionData)
    mentionTable.drop_duplicates('Comment', keep=False, inplace=True)

    return mentionTable

# uses vader to analyze sentiment of a comment and the symbol mentioned
def sentimentAnalysis(comment):
    wsbSentimentAnalyzer = SentimentIntensityAnalyzer()
    wsbSentiment ={'Score': float("{:.2f}".format(wsbSentimentAnalyzer.polarity_scores(comment)['compound']))}
    return wsbSentiment['Score']


def symbolSentiment(mentionTable):
    symbolSentimentDict = {'Symbol':[], 'Sentiment':[]}
    for ind in mentionTable.index:

        for val in mentionTable['Symbols Mentioned'][ind]:
            symbolSentimentDict['Symbol'].append(val)
            symbolSentimentDict['Sentiment'].append(mentionTable['Sentiment'][ind])

    return pd.DataFrame(symbolSentimentDict)


def symbolMatchData(symbolTimeStampDict):
    matchedData = {'Symbol': [], 'Avg.Sentiment':[], 'Weighted Sentiment':[] , 'Mentions': [], 'Sentiment': []}
    for index, x in enumerate(symbolTimeStampDict['Symbol']):
        if symbolTimeStampDict['Symbol'][index] not in matchedData['Symbol']:
            matchedData['Symbol'].append(symbolTimeStampDict['Symbol'][index])
            matchedData['Sentiment'].append([symbolTimeStampDict['Sentiment'][index]])
        else:
            matchedData['Sentiment'][matchedData['Symbol'].index(x)].append(symbolTimeStampDict['Sentiment'][index])

    for index, x in enumerate(matchedData['Symbol']):
        matchedData['Avg.Sentiment'].append(statistics.fmean(matchedData['Sentiment'][index]))

    index = 0
    while index < len(matchedData['Symbol']):
        if len(matchedData['Sentiment'][index]) < 10:
            matchedData['Symbol'].pop(index)
            matchedData['Sentiment'].pop(index)
            matchedData['Avg.Sentiment'].pop(index)
        else:
            matchedData['Mentions'].append(len(matchedData['Sentiment'][index]))
            index += 1

    index = 0
    while index < len(matchedData['Avg.Sentiment']):
        weight = 0.0
        count = 0
        while count < matchedData['Mentions'][index]:
            weight += 0.01 #would take 1,000 mentions with an avg sentiment 0.10 to get the weighted sentiment to 1.0
            count += 1
        weighted = float(matchedData['Avg.Sentiment'][index] * weight)
        matchedData['Weighted Sentiment'].append(weighted)
        index+=1

    return pd.DataFrame(matchedData).sort_values(by='Mentions',ascending=False).round(decimals=2)


def symbolSentimentVisualization(dataset):
    plotdata = dataset.explode('Sentiment')
    pos = [0.0, 1.0]
    colors = ['#FF5000', '#00C805']
    cmap = LinearSegmentedColormap.from_list("", list(zip(pos, colors)))
    matplotlib.cm.register_cmap("newmap", cmap)
    sns.set_style("darkgrid")
    sns.set(rc={'figure.figsize': (32, 14)})
    sns.set(font_scale=2.0)

    plt.scatter(x="Symbol", y="Avg.Sentiment", data=dataset, color='black', marker='x', s=150)
    plt.scatter(x="Symbol", y="Weighted Sentiment", data=dataset, color='r', marker='_', s=500)
    dplot = sns.swarmplot(x="Symbol", y="Sentiment", hue='Sentiment', palette="newmap", data=plotdata,
                          order=plotdata.Symbol.value_counts().iloc[:10].index)
    dplot.get_legend().remove()

    plt.show()

def main():
    reddit = praw.Reddit(
        # client_id = "",
        # client_secret = "",
        # user_agent = "",
        # username = "",
        # password = ""
    )
    url = input("Paste url of thread to parse: ")

    # Original table made out of all the comments parsed, contains every comment with a mention, sentiment
    # and the symbols in each sentence parsed
    mentionTable = mentionTableCreate(reddit, url)

    # Table that contains each instance of the symbol mentioned paired with its sentiment value
    # for the purpose of splitting the sentences that contained multiple values
    symbolSentimentTable = symbolSentiment(mentionTable)

    # Table of each symbol paired with all of their sentiment values, filters out to only symbols that are mentioned
    # 6 or more times
    matchedDataTable = symbolMatchData(symbolSentimentTable)

    print(matchedDataTable)
    print(symbolSentimentVisualization(matchedDataTable))


main()

