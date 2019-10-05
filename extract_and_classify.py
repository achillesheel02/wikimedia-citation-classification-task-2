# import statements
import mwapi
import mwparserfromhell

title_query = input()
session = mwapi.Session('https://en.wikipedia.org')

response = session.get(
    action="query",
    list="search",
    format="json",
    srsearch=title_query

)
for item in response['query']['search']:
    print("{title}\n{pageid}\n......................................\n\n".format(**item))
    content = session.get(
        action="parse",
        pageid=int("{pageid}".format(**item)),
        format="json"
    )
    wikicode = mwparserfromhell.parse(content['parse']['text']['*'])
    print(wikicode.filter_text())
    break