import requests
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_site_data(state_code: str, parameter_code: str, county: str = None, start_year: int = None, end_year: int = None) -> pd.DataFrame:
    """Get site data with optional county and year filtering."""
    base_url = f'https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd={state_code}&parameterCd={parameter_code}'
    
    if county:
        try:
            base_url += f'&countycd={state_code}{county}'
        except Exception as e:
            st.warning(f"County filtering failed: {str(e)}. Proceeding without county filter.")
    
    site_response = requests.get(base_url)
    all_data = []
    status = st.empty()

    for row in site_response.text.splitlines():
        if row.split('\t')[0] == 'agency_cd':
            headers = row.split('\t')
        elif row.split('\t')[0] == 'USGS':
            site_info = row.split('\t')
            site_code = site_info[headers.index('site_no')]
            site_name = site_info[headers.index('station_nm')]
            try:
                latitude = float(site_info[headers.index('dec_lat_va')])
                longitude = float(site_info[headers.index('dec_long_va')])
            except ValueError:
                continue

            try:
                df = get_site_timeseries(site_code, parameter_code, 
                                       start_date=f'{start_year}-01-01' if start_year else '1990-01-01',
                                       end_date=f'{end_year}-12-31' if end_year else '2023-12-31')
                if df is not None:
                    df['latitude'] = latitude
                    df['longitude'] = longitude
                    df['site_no'] = site_code
                    df['site_name'] = site_name
                    all_data.append(df)
                    status.write(f'Collecting data from {site_name}...')
            except Exception as e:
                continue

    status.empty()
    if all_data:
        return pd.concat(all_data, axis=0)
    return None

def get_sites_with_multiple_parameters(state_code: str, parameter_codes: List[str], min_values: int = 10000) -> Dict:
    """Find sites that have data for all specified parameters with minimum number of values."""
    sites_data = {}
    
    for param_code in parameter_codes:
        url = f'https://nwis.waterdata.usgs.gov/nwis/dv/?referred_module=sw&site_tp_cd=ST&format=rdb&freq=D&' + \
              f'parameterCd={param_code}&stateCd={state_code}&siteStatus=all'
        
        response = requests.get(url)
        for row in response.text.splitlines():
            if row.startswith('USGS'):
                parts = row.split('\t')
                site_code = parts[1]
                try:
                    value_count = int(parts[-1])
                    if value_count >= min_values:
                        if site_code not in sites_data:
                            sites_data[site_code] = {'parameters': [], 'value_counts': {}}
                        sites_data[site_code]['parameters'].append(param_code)
                        sites_data[site_code]['value_counts'][param_code] = value_count
                except (ValueError, IndexError):
                    continue
    
    # Filter sites that have all parameters
    return {k: v for k, v in sites_data.items() if len(v['parameters']) == len(parameter_codes)}

def get_site_timeseries(site_code: str, parameter_code: str, start_date: str = '1990-01-01', 
                       end_date: str = '2023-12-31') -> pd.DataFrame:
    """Get time series data for a specific site and parameter."""
    url = f'https://waterservices.usgs.gov/nwis/dv/?format=json&sites={site_code}&' + \
          f'startDT={start_date}&endDT={end_date}&parameterCd={parameter_code}'
    
    response = requests.get(url)
    try:
        data = response.json()['value']['timeSeries'][0]['values'][0]['value']
        df = pd.DataFrame(data)
        df['dateTime'] = pd.to_datetime(df['dateTime'])
        df.set_index('dateTime', inplace=True)
        df['value'] = pd.to_numeric(df['value'])
        df.drop(columns=['qualifiers'], inplace=True)
        return df
    except:
        return None

def parallel_site_data_collection(site_codes: List[str], parameter_code: str, 
                                max_workers: int = 4) -> pd.DataFrame:
    """Collect data from multiple sites in parallel."""
    all_data = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_site = {
            executor.submit(get_site_timeseries, site_code, parameter_code): site_code 
            for site_code in site_codes
        }
        
        with st.spinner('Collecting data from multiple sites...'):
            progress_bar = st.progress(0)
            completed = 0
            
            for future in as_completed(future_to_site):
                site_code = future_to_site[future]
                try:
                    df = future.result()
                    if df is not None:
                        df['site_no'] = site_code
                        all_data.append(df)
                except Exception as e:
                    st.error(f"Error collecting data for site {site_code}: {str(e)}")
                
                completed += 1
                progress_bar.progress(completed / len(site_codes))
    
    if all_data:
        return pd.concat(all_data)
    return None

def get_parameter_name(parameter_code: str) -> str:
    """Get parameter name from code."""
    parameter_names = {
        '00060': 'Discharge (cfs)',
        '00010': 'Temperature (°C)',
        '00095': 'Conductivity (µS/cm)',
        '00400': 'pH',
        '00300': 'Dissolved Oxygen (mg/L)',
        '00665': 'Phosphorus (mg/L)',
        '00631': 'Nitrate + Nitrite (mg/L)',
        '00608': 'Ammonia (mg/L)',
        '00945': 'Sulfate (mg/L)',
        '00940': 'Chloride (mg/L)'
    }
    return parameter_names.get(parameter_code, f'Parameter {parameter_code}')

def collect_single_parameter_data(state_code: str, selected_params: List[str], 
                                parameter_codes: Dict[str, str], county: str = None,
                                start_year: int = None, end_year: int = None) -> pd.DataFrame:
    """Collect data for single parameters with optional filtering."""
    data_frames = []
    status = st.empty()
    
    for param in selected_params:
        param_code = parameter_codes[param]
        status.write(f"Fetching {param} data...")
        df = get_site_data(state_code, param_code, county, start_year, end_year)
        if df is not None:
            df['parameter'] = get_parameter_name(param_code)
            data_frames.append(df)
    
    status.empty()
    if data_frames:
        return pd.concat(data_frames, axis=0)
    return None

def collect_multiple_parameter_data(state_code: str, selected_params: List[str], 
                                  parameter_codes: Dict[str, str], min_values: int,
                                  county: str = None, start_year: int = None, 
                                  end_year: int = None) -> pd.DataFrame:
    """Collect data for sites that have all selected parameters."""
    param_code_list = [parameter_codes[param] for param in selected_params]
    sites = get_sites_with_multiple_parameters(state_code, param_code_list, min_values)
    
    if not sites:
        st.warning("No sites found with all selected parameters.")
        return None
    
    st.write(f"Found {len(sites)} sites with all selected parameters.")
    data_frames = []
    status = st.empty()
    
    for site_code in sites:
        for param in selected_params:
            param_code = parameter_codes[param]
            status.write(f"Fetching {param} data for site {site_code}...")
            df = get_site_timeseries(site_code, param_code,
                                   start_date=f'{start_year}-01-01' if start_year else '1990-01-01',
                                   end_date=f'{end_year}-12-31' if end_year else '2023-12-31')
            if df is not None:
                df['site_no'] = site_code
                df['parameter'] = get_parameter_name(param_code)
                data_frames.append(df)
    
    status.empty()
    if data_frames:
        return pd.concat(data_frames, axis=0)
    return None

def collect_high_frequency_data(state_code: str, param: str, parameter_codes: Dict[str, str], 
                              min_values: int, county: str = None, 
                              start_year: int = None, end_year: int = None) -> pd.DataFrame:
    """Collect data from sites with high-frequency measurements."""
    param_code = parameter_codes[param]
    base_url = f'https://nwis.waterdata.usgs.gov/nwis/dv/?referred_module=sw&site_tp_cd=ST&format=rdb&freq=D&' + \
              f'parameterCd={param_code}&stateCd={state_code}&siteStatus=all'
    
    if county:
        try:
            base_url += f'&countycd={state_code}{county}'
        except Exception as e:
            st.warning(f"County filtering failed: {str(e)}. Proceeding without county filter.")
    
    response = requests.get(base_url)
    high_freq_sites = []
    
    for row in response.text.splitlines():
        if row.startswith('USGS'):
            parts = row.split('\t')
            site_code = parts[1]
            try:
                value_count = int(parts[-1])
                if value_count >= min_values:
                    high_freq_sites.append(site_code)
            except (ValueError, IndexError):
                continue
    
    if not high_freq_sites:
        st.warning(f"No sites found with {min_values}+ values for {param}")
        return None
    
    st.write(f"Found {len(high_freq_sites)} high-frequency sites.")
    data_frames = []
    status = st.empty()
    
    for site_code in high_freq_sites:
        status.write(f"Fetching data for site {site_code}...")
        df = get_site_timeseries(site_code, param_code,
                               start_date=f'{start_year}-01-01' if start_year else '1990-01-01',
                               end_date=f'{end_year}-12-31' if end_year else '2023-12-31')
        if df is not None:
            df['site_no'] = site_code
            df['parameter'] = get_parameter_name(param_code)
            data_frames.append(df)
    
    status.empty()
    if data_frames:
        return pd.concat(data_frames, axis=0)
    return None 