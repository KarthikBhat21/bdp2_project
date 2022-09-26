from webbrowser import get
import dvc.api
import pandas as pd
import urllib.request
import wget

#myUrl = dvc.api.get_url('data/Ukraine_Data.csv', repo='https://github.com/KarthikBhat21/bdp2_project', rev='878bb971a28d767dd3436c525688af61c2d1de82')

myUrl = dvc.api.get_url('data/Ukraine_Data.csv', repo='https://github.com/KarthikBhat21/bdp2_project', rev='878bb971a28d767dd3436c525688af61c2d1de82')

print(myUrl)



#wget.download(myUrl)



# data = urllib.request.urlopen(url)
# for line in data:
#     print(line)

# df = pd.read_csv(url, encoding='utf-8')

# print(df)

#with open()

# dvc.api.get(url, "C://BDBA/BDP2/Git_Repo/")

# with dvc.api.open(
#     "data/Ukraine_Data.csv.dvc",
#     repo="https://github.com/KarthikBhat21/bdp2_project") as fd:
#         for line in fd:
#             print(line)
#         #fd.open()





