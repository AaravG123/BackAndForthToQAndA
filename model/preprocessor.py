import json
import re

with open('data_scraper/data_scraper/spiders/data.json', 'r') as rawData:
    data = json.load(rawData)

newData = {}

for i in range(0, len(data)):
    newData.update({f"pair{i+1}": {}})
    newData[f"pair{i+1}"].update({f"question{i+1}": data[i]['usercomment'][0].split('----')[1]})
    newData[f"pair{i+1}"].update({f"answer{i+1}": ""})

    questioner = data[i]['usercomment'][0].split('----')[0]

    for j in range(1, len(data[i]['usercomment'])):
        user = data[i]['usercomment'][j].split('----')[0]
        newComment = data[i]['usercomment'][j].split('----')[1]

        if user == questioner:
            newData[f"pair{i+1}"][f"question{i+1}"] += " " + newComment
        else:
            newData[f"pair{i+1}"][f"answer{i+1}"] += " " + newComment

preprocessedFile = open("model/preprocessed_data.json", "w")
json.dump(newData, preprocessedFile, indent=4)
preprocessedFile.close()