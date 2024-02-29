def get_warnings():
    import feedparser

    url = "https://ims.gov.il/sites/default/files/ims_data/rss/alert/rssAlert_general_country_he.xml"
    feed = feedparser.parse(url)
    summaries = [{"summery": summary['summary'], "title":summary['title']} for summary in feed['entries']]
    return_data = {'encoding': feed['encoding'],
                   'summaries': summaries
                   }
    return return_data