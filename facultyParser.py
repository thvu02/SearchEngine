from pymongo import MongoClient
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin  

# Connect to MongoDB
def connectDataBase():
    DB_NAME = 'CPP'
    DB_HOST = 'localhost'
    DB_PORT = 27017
    try:
        client = MongoClient(host=DB_HOST, port=DB_PORT)
        db = client[DB_NAME]
        return db
    except:
        print("Database not connected successfully")

# Need this method to fix this error: https://www.cpp.edu/faculty/alas, so we can store its image correctly
def check_base_url(url):
    if not url.endswith('/'):
        # Check if the URL looks like a file (Ex. doesn't have indexs.html at end of url)
        if '.' not in url.split('/')[-1]:
            url += '/' 
    return url   

# Goes through each faculty member and parses the information
def process_faculty_pages(pages_collection, faculty_collection):
    # get the 10 faculty pages from the pages collection
    faculty = list(pages_collection.find({"isTarget": True}))

    # loop through each person in the faculty list and parse 
    for member in faculty:
        # get the html and url from MongoDB
        html = member['html']
        bs = BeautifulSoup(html, 'html.parser')
        url = member['url']

        # get the area of search content 
        fac_info = bs.find('div', class_='fac-info')
        main_section = bs.find_all('div', class_='blurb')
        side_bar = bs.find_all('div', class_= 'accolades')
        main_section.extend(fac_info)
        main_section.extend(side_bar)
        main_section_html = "".join(str(section) for section in main_section)

        # Extracting faculty details from faculty info
        name = fac_info.h1.get_text(strip=True) if fac_info.h1 else "Missing"
        email = fac_info.find('a', {'href': re.compile(r'mailto:')}).get_text(strip=True) if fac_info.find('a', {'href': re.compile(r'mailto:')}) else "Missing"
        phone = fac_info.find('p', class_='phoneicon').get_text(strip=True) if fac_info.find('p', class_='phoneicon') else "Missing"    
        image_tag = fac_info.find('img')
        if image_tag and 'src' in image_tag.attrs:
            image_url = image_tag['src']
            # Ensure the image URL is absolute
            if not image_url.startswith('http'):
                url = check_base_url(url)
                image_url = urljoin(url, image_url)
        else:
            image_url = "no image"

        # print faculty information
        print(f"Name: {name}")
        print(f"Email: {email}")
        print(f"Phone: {phone}")
        print(f"Image URL: {image_url}")
        print()

        faculty_member_data = {
            'name': name,
            'email': email,
            'phone': phone,
            'url': url,
            'page_id': member['_id'],
            'image_url': image_url,
            'faculty_parse': main_section_html
        }

        # Store the faculty members in faculty collection
        faculty_collection.insert_one(faculty_member_data)
    

if __name__ == '__main__':
    db = connectDataBase()
    pages_collection = db['pages']
    faculty_collection = db['faculty']
    process_faculty_pages(pages_collection, faculty_collection)
    print("Faculty Pages parsing completed!")