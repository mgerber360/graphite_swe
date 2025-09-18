import pandas as pd
import requests
import warnings

def get_density(pop_file_name, area_file_name, FIPS):
    """
    Calculate the population density for a given FIPS code.

    Args: pop_file_name (str): Path to the CSV file containing population estimates.
          area_file_name (str): Path to the CSV file containing county landmass data.
          FIPS (str): The 5-digit FIPS code (state + county).
    Returns: float: Population density (people per square mile).
    """
    area_data = pd.read_csv(area_file_name, dtype={'FIPS': str})
    area_data['FIPS'] = area_data['FIPS'].str.zfill(5) # Ensure FIPS codes are 5 digits
    area_data = area_data.set_index('FIPS')
    pop_data = pd.read_csv(pop_file_name, dtype={'FIPStxt': str})
    pop_data['FIPStxt'] = pop_data['FIPStxt'].str.zfill(5)
    pop_data = pop_data.set_index('FIPStxt')
    area = area_data.loc[FIPS, 'sq_mi']
    pop = pop_data.loc[FIPS]
    pop = pop[pop['Attribute'] == 'CENSUS_2020_POP']['Value'].values[0]    
    return float(pop) / area

def get_FIPS_code(lat, lon):
    """
    Retrieves the FIPS code for a given latitude and longitude using the US Census Geocoder API.

    Args: latitude (float): The latitude coordinate, longitude (float): The longitude coordinate.
    Returns: str or None: The 5-digit FIPS code (state + county) if found, otherwise None.
    """
    url = f"https://geocoding.geo.census.gov/geocoder/geographies/coordinates?x={lon}&y={lat}&benchmark=4&vintage=423&format=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract FIPS code from the response
        if "result" in data and "geographies" in data["result"] and "Counties" in data["result"]["geographies"]:
            county_data = data["result"]["geographies"]["Counties"][0]
            state_fips = str(county_data["STATE"])
            county_fips = str(county_data["COUNTY"])
            return state_fips + county_fips
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        return None
    except IndexError:
        print(f"No county data found for coordinates: ({lat}, {lon})")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None  
    
def is_dense_area(departure_coords, destination_coords, pop_file_name='population_estimates.csv', area_file_name='county_landmass.csv', threshold=1000):
    """
    Determine if a given latitude and longitude is in a dense area based on population density threshold (1000 is commonly used as the upper-limit for rural areas).

    Args: departure_coords (tuple): (latitude, longitude) for the departure location.
          destination_coords (tuple): (latitude, longitude) for the destination location.
          pop_file_name (str): Path to the CSV file containing population estimates.
          area_file_name (str): Path to the CSV file containing county landmass data.
          threshold (int): Population density threshold to classify as dense area.
    Returns: bool: True if both departure and destination are in dense areas, False otherwise.
    """
    departure_FIPS = get_FIPS_code(departure_coords[0], departure_coords[1])
    destination_FIPS = get_FIPS_code(destination_coords[0], destination_coords[1])
    if departure_FIPS is None or destination_FIPS is None:
        warnings.warn(f"Could not retrieve FIPS code for the departure location or destination.")
        return True # default to True if FIPS code cannot be determined
    departure_density = get_density(pop_file_name, area_file_name, departure_FIPS)
    destination_density = get_density(pop_file_name, area_file_name, destination_FIPS)
    if departure_density >= threshold and destination_density >= threshold:
        return True
    return False
