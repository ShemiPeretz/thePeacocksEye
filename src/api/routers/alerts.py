import xml.etree.ElementTree as ET
import json
import re
import requests
from dotenv import load_dotenv
from fastapi import HTTPException, APIRouter
from starlette.responses import JSONResponse

load_dotenv()
router = APIRouter()

xml_url = 'https://ims.gov.il/sites/default/files/ims_data/rss/alert/rssAlert_general_country_en.xml'

def strip_html_tags(text):
    # Remove HTML tags using a regular expression
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_update_prefix(text):
    # Remove the "update:" prefix if it exists
    return text.replace('update:', '').strip()


def clean_title(text):
    # Remove the "Updated on:..." part from the title using regex
    return re.sub(r' Updated on:.*', '', text)


def parse_xml_to_json(xml_url):
    # Fetch the XML content from the URL
    response = requests.get(xml_url)
    response.raise_for_status()  # Raise an error if the request was unsuccessful

    # Parse the XML content
    root = ET.fromstring(response.content)

    items = []
    for item in root.findall('./channel/item'):
        title = item.find('title').text
        title_clean = clean_title(title)
        description = item.find('description').text
        date = item.find('pubDate').text

        # Clean the description text
        description_clean = strip_html_tags(description)
        description_clean = remove_update_prefix(description_clean)

        item_dict = {
            'title': title_clean,
            'description': description_clean,
            'date': date
        }
        items.append(item_dict)

    if not items:
        return json.dumps("no alerts", indent=4)

    return items


@router.get("/get-alerts/")
async def get_alerts():
    alerts = parse_xml_to_json(xml_url)
    return JSONResponse(content={"alerts": alerts})
