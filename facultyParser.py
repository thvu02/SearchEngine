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

# Need this method to fix this error: https://www.cpp.edu/faculty/alas, so we can store its "image" correctly
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
        main_body_text = ''
        main_body = bs.find('div', id='main-body')
        if main_body:
            main_body_text = main_body.get_text(separator='\n', strip=True)
            print(f"Extracted Main Body Text from {url}")
        else:
            print(f"No main body found in {url}")

        aside_text = ''
        asides = bs.select('main aside')
        for aside in asides:
            aria_label = aside.get('aria-label', 'Unknown Section')
            section_text = aside.get_text(separator='\n', strip=True)
            aside_text += f"\n[Section: {aria_label}]\n{section_text}\n"
            print(f"Extracted text from aside section: {aria_label}")
        
        combined_text = f"{main_body_text}\n\n{aside_text}".strip()


        # Extracting faculty details
        fac_info = bs.find('div', class_='fac-info')
        title_dept = fac_info.find('span', class_='title-dept').get_text(strip=True) if fac_info.find('span', class_='title-dept') else "Not Available"   
        name = fac_info.h1.get_text(strip=True) if fac_info.h1 else "Not Available" 
        email = fac_info.find('a', {'href': re.compile(r'mailto:')}).get_text(strip=True) if fac_info.find('a', {'href': re.compile(r'mailto:')}) else "Not Available"
        phone = fac_info.find('p', class_='phoneicon').get_text(strip=True) if fac_info.find('p', class_='phoneicon') else "Not Available"    
        image_tag = fac_info.find('img')
        if image_tag and 'src' in image_tag.attrs:
            image_url = image_tag['src']
            # Ensure the image URL is absolute
            if not image_url.startswith('http'):
                url = check_base_url(url)
                image_url = urljoin(url, image_url)
        else:
            image_url = "default_image.jpg"  # Fallback image
        
        text_snippet = f"{name}. {title_dept}. Email. {email} Phone number. {phone}"

        # print faculty information
        print(f"Name: {name}")
        print(f"Email: {email}")
        print(f"Phone: {phone}")
        print(f"Image URL: {image_url}")
        print(f"Text Snippet: {text_snippet}")
        print()

        faculty_member_data = {
            'name': name,
            'email': email,
            'phone': phone,
            'url': url,
            'page_id': member['_id'],
            'image_url': image_url,
            'text': combined_text,
            'text_snippet': text_snippet
        }

        # Store the faculty members in faculty collection
        faculty_collection.insert_one(faculty_member_data)
    

if __name__ == '__main__':
    db = connectDataBase()
    pages_collection = db['pages']
    faculty_collection = db['faculty']
    process_faculty_pages(pages_collection, faculty_collection)
    print("Faculty Pages parsing completed!")