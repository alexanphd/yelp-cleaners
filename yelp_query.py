import pandas as pd
from yelpapi import YelpAPI

# get yelp api key stored in txt file
f = open(r'data\yelp_api_key.txt','r')
api_key = f.read()
f.close()

df = pd.read_csv('data/processed data/reviews_nearest_score_no_limit_df.csv')
# business_df = pd.read_csv('data/processed data/business_df.csv')

def get_business_id(name="Shanghai Noodle House", address="10300 Anderson Mill Rd, Ste A"):
    yelp_api = YelpAPI(api_key, timeout_s=3.0)
    response = yelp_api.business_match_query(name=name,
                                         address1=address,
                                         city='Austin',
                                         state='TX',
                                         country='US')
    if not response['businesses']:
        print('Business not found in Yelp API query!')
        return
    response_name = response['businesses'][0]['name']
    bid = response['businesses'][0]['id']
    if bid not in df.business_id.unique():
        print('Business not found in Yelp Academic Dataset!')
        return
    print(f'Business found: {response_name}')
    return bid

def get_reviews(name="Shanghai Noodle House", address="10300 Anderson Mill Rd, Ste A"):
    bid = get_business_id(name=name, address=address)
    if bid:
        return df[df['business_id'] == bid].text.values