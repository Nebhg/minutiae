import requests
from bs4 import BeautifulSoup
import csv


def fetch_box_score():
    hrefs = set()
    url = "https://www.basketball-reference.com/playoffs/NBA_2023.html"

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Steps #1 to #9
    table_container = soup.find("div", {"id": "div_all_playoffs"})
    tbody = table_container.find("tbody")
    tr_toggleable = tbody.find_all("tr", {"class": "toggleable"})
    for trs in tr_toggleable:
        a_s = trs.find_all("a")
        [hrefs.add(x["href"]) for x in a_s]


    print(hrefs)
    for href_link in list(hrefs):
        boxscore_link = f"https://www.basketball-reference.com{href_link}"
        boxscore_link = "https://www.basketball-reference.com/boxscores/202304160DEN.html"
        print(boxscore_link)
        response = requests.get(boxscore_link)
        soup = BeautifulSoup(response.content, 'html.parser')
        titles = set()
        tags_with_text = soup.find_all(lambda tag: " and Advanced Stats" in tag.text)
        lowest_level_tags = [tag for tag in tags_with_text if
                             not tag.find(lambda child: " and Advanced Stats" in child.text)]
        for x in lowest_level_tags:
            titles.add(x.text.replace("Table", "").strip())
        print(titles)

        rows = []
        for tbody in soup.find_all("tbody"):
            for trs in tbody.find_all("tr"):
                row = {}
                row["name"] = trs.find("th").text
                if trs.find("th").text == "Reserves":
                    continue
                for tds in trs.find_all("td"):
                    row[tds["data-stat"]] = tds.text
                rows.append(row)

        print(rows[12])


        break
fetch_box_score()
