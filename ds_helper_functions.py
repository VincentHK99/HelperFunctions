#import relevent packages
from bs4 import BeautifulSoup
import requests


def get_soup(url,parser='html.parser',print_soup=False,timeout=10):
  """
    Sends a request to webpage and creates a soup object for a given url.
    If an error or a status code different to 200 occus the function will
    attemp to isolate the issue for the user of the function.
  """
  try:
    page = requests.get(url,timeout=timeout)
    
    # prove clarification as to what different status codes indicate
    if page.status_code == 200:
      print('Request successful!')
    elif page.status_code == 404:
      print('URL is not recognised')
    else:
      print(f'Uncommon response occured ({print(page.status_code)} refer to article below for quidance \n https://developer.mozilla.org/en-US/docs/Web/HTTP/Status')
    soup = BeautifulSoup(page.content,parser)
    if print_soup == True:
      print(soup.prettify())
  
  except:
    # the the user know if the requests package did not work
    print('An exception occurred. Perhaps the URL entered is incorrect')
  
  return soup
