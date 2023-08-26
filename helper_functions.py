from geopy.geocoders import GoogleV3, Nominatim
from geopy.extra.rate_limiter import RateLimiter

def reverse_lookup(df, column, lookup_dict, new_column):
    new_array = []
    for entry in df[column]:
        for key, values in lookup_dict.items():
            if entry in values:
                new_array.append(key)
    df[new_column] = new_array
    return df

def geocoding_Nominatim(unique_address):
    locator = Nominatim(user_agent = 'edwin.goh@ncs.com.sg', timeout = 100000)
    geocode = RateLimiter(locator.geocode, max_retries = 5)
    print('Converting {} to GPS coordinates'.format(unique_address))
    location = geocode(unique_address, country_codes = 'SG')
    return location.latitude, location.longitude
   
def geocoding_GoogleMaps(unique_address):
    locator = GoogleV3(api_key = 'AIzaSyAVcsEuf2KiG512u15E4fVNdsMZKWJAtuk', timeout = 100000)
    geocode = RateLimiter(locator.geocode, min_delay_seconds = 0.01, max_retries = 5)
    print('Converting {} to GPS coordinates'.format(unique_address))
    location = geocode(unique_address, region = 'SG', bounds = [[1.203748, 103.585763], [1.481474, 104.041009]])
    return location.latitude, location.longitude