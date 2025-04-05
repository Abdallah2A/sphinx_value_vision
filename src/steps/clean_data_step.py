import os.path
import pandas as pd
import re
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from zenml import step


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file and remove the 'url' column.

    Parameters:
        dataset_path (str): File path to the CSV dataset.

    Returns:
        pd.DataFrame: DataFrame with the 'url' column dropped.
    """
    data_frame = pd.read_csv(dataset_path)
    data_frame.drop('url', axis=1, inplace=True)
    return data_frame


def clean_price_col(price_series: pd.Series) -> pd.Series:
    """
    Clean the price column by removing 'EGP' and commas.

    Parameters:
        price_series (pd.Series): Series containing price data.

    Returns:
        pd.Series: Cleaned price series.
    """
    price_series = price_series.replace('EGP', '', regex=True)
    price_series = price_series.replace(',', '', regex=True)
    return price_series.astype(int)


def clean_area_col(area_series: pd.Series) -> pd.Series:
    """
    Clean the area column by removing 'm²'.

    Parameters:
        area_series (pd.Series): Series containing area data.

    Returns:
        pd.Series: Cleaned area series.
    """
    area_series = area_series.replace('m²', '', regex=True)
    return area_series.astype(int)


def clean_rooms_col(rooms_series: pd.Series) -> pd.Series:
    """
    Clean the rooms column by removing the word 'rooms'.

    Parameters:
        rooms_series (pd.Series): Series containing room count data.

    Returns:
        pd.Series: Cleaned rooms series.
    """
    rooms_series = rooms_series.replace('rooms', '', regex=True)

    return rooms_series.astype(int)


def clean_bathrooms_col(bathrooms_series: pd.Series) -> pd.Series:
    """
    Clean the bathrooms column by removing the word 'bathroom'.

    Parameters:
        bathrooms_series (pd.Series): Series containing bathroom count data.

    Returns:
        pd.Series: Cleaned bathrooms series.
    """
    bathrooms_series = bathrooms_series.replace('bathroom', '', regex=True)
    bathrooms_series = bathrooms_series[pd.to_numeric(bathrooms_series, errors="coerce").notnull()]
    return bathrooms_series.astype(int)


def clean_floor_col(floor_series: pd.Series) -> pd.Series:
    """
    Clean the floor column by replacing 'Ground' with 0.

    Parameters:
        floor_series (pd.Series): Series containing floor data.

    Returns:
        pd.Series: Cleaned floor series.
    """
    floor_series = floor_series.replace('Ground', 0)
    floor_series = floor_series[pd.to_numeric(floor_series, errors="coerce").notnull()]
    return floor_series.astype(float).astype(int)


def clean_year_built_col(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Create 'year_built' and calculate age based on the year 2025.

    Parameters:
        data_frame (pd.DataFrame): Input DataFrame containing 'year_built' column.

    Returns:
        pd.DataFrame: DataFrame with 'apartment_age' column updated.
    """
    data_frame['apartment_age'] = 2025 - data_frame['year_built']
    data_frame['apartment_age'] = data_frame['apartment_age'].astype(int)
    return data_frame


def remove_keywords(text: str, keywords: list[str]) -> str:
    """
    Remove specified keywords from a text string.

    Parameters:
        text (str): Input text to process.
        keywords (list[str]): List of keywords to remove from the text.

    Returns:
        str: Text with all specified keywords removed.
    """
    for keyword in keywords:
        text = text.replace(keyword, '')
    return text


def bulk_replace(text: str, replacements: dict[str, str]) -> str:
    """
    Apply multiple string replacements to a text based on a dictionary.

    Parameters:
        text (str): Input text to process.
        replacements (dict[str, str]): Dictionary mapping old strings to new strings.

    Returns:
        str: Text with all replacements applied.
    """
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def regex_replace_chain(text: str) -> str:
    """
    Apply a series of regex-based replacements to simplify location text.

    Parameters:
        text (str): Input text to process.

    Returns:
        str: Text after applying regex replacements.
    """
    patterns = [
        (r'.*(mountain view icity).*', r'\1'),
        (r'.*(banafsag).*', r'\1'),
        (r'.*(narges).*', r'\1'),
        (r'.*(yasmeen).*', r'\1'),
        (r'.*(area).*', r'\1'),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def final_processing(text: str) -> str:
    """
    Perform final text cleaning, removing standalone letters and specific words.

    Parameters:
        text (str): Input text to clean.

    Returns:
        str: Fully cleaned text.
    """
    # Remove standalone letters, optionally in parentheses
    text = re.sub(r'\(?\b[a-zA-Z]\b\)?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Replace standalone 'al' with 'el'
    text = re.sub(r'\bal\b', 'el', text)

    # Remove specific standalone words
    words_to_remove = ['hay', 'st', 'south', 'east', 'qebly', 'bahri', 'north', 'of', 'old', 'greater', 'city']
    for word in words_to_remove:
        text = re.sub(r'\b' + re.escape(word) + r'\b', '', text)

    # Remove 'el' if at the start
    text = re.sub(r'^el\b\s*', '', text)

    # Final space cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_location(text: str, keywords_to_remove: list[str]) -> str:
    """
    Process location text through a series of cleaning steps.

    Steps include removing keywords, applying bulk replacements, regex replacements,
    and final processing.

    Parameters:
        text (str): Input location text.
        keywords_to_remove (list[str]): Keywords to remove from the text.

    Returns:
        str: Processed location text.
    """
    text = remove_keywords(text, keywords_to_remove)
    replacements = {
        'ain sokhna': 'ain el sokhna',
        '6 october': '6th october',
        'gameaya': 'gmaayat',
        'amreya': 'amirya',
        'katameya': 'kattameya',
        'eastown compound - sodic': 'sodic east compound',
        'matruh': 'matrouh',
        'hadabah': 'hadaba',
        'elshorouk': 'el shorouk',
        'neighborhoods': 'neighborhood',
        'matareya': 'mataria',
        'gesr': 'gisr',
        'garden lakes compound - hyde park': 'hyde park compound',
        'beshr': 'bishr',
        'shaikh': 'sheikh',
        'soyouf': 'seyouf',
        'zaytun': 'zaytoun',
        'mahalah': 'mahallah',
        'shibin': 'shebeen',
        'hai el kawsr': '4th neighborhood',
        'sodic compound': 'sodic',
        '2nd nozha': 'nozha'
    }
    text = bulk_replace(text, replacements)
    text = regex_replace_chain(text)
    text = final_processing(text)
    return text


def convert_to_ordinal(text: str, pattern: re.Pattern, number_map: dict[str, str]) -> str:
    """
    Convert matched numbers in text to their ordinal forms using a pattern and mapping.

    Note: The original code has a special case where '12', 'twelve', 'twelfth', '13',
    'thirteen', and 'thirteenth' are all mapped to '13th', which may be unintended.
    This behavior is preserved to maintain functionality.

    Parameters:
        text (str): Input text to process.
        pattern (re.Pattern): Compiled regex pattern to match numbers.
        number_map (dict[str, str]): Mapping from number strings to ordinal forms.

    Returns:
        str: Text with numbers converted to ordinals.
    """
    special_numbers = {'12', 'twelve', 'twelfth', '13', 'thirteen', 'thirteenth'}
    return pattern.sub(
        lambda match: '13th' if match.group(0).lower() in special_numbers else number_map[match.group(0).lower()],
        text
    )


def process_location_numbers(location_series: pd.Series) -> pd.Series:
    """
    Process location text by converting numbers to ordinals and rearranging if ending with an ordinal.

    Parameters:
        location_series (pd.Series): Series containing location data.

    Returns:
        pd.Series: Processed location series.
    """
    number_map = {
        '1': '1st', 'one': '1st', 'first': '1st',
        '2': '2nd', 'two': '2nd', 'second': '2nd',
        '3': '3rd', 'three': '3rd', 'third': '3rd',
        '4': '4th', 'four': '4th', 'fourth': '4th',
        '5': '5th', 'five': '5th', 'fifth': '5th',
        '6': '6th', 'six': '6th', 'sixth': '6th',
        '7': '7th', 'seven': '7th', 'seventh': '7th',
        '8': '8th', 'eight': '8th', 'eighth': '8th',
        '9': '9th', 'nine': '9th', 'ninth': '9th',
        '10': '10th', 'ten': '10th', 'tenth': '10th',
        '11': '11th', 'eleven': '11th', 'eleventh': '11th',
        '12': '12th', 'twelve': '12th', 'twelfth': '12th',
        '13': '13th', 'thirteen': '13th', 'thirteenth': '13th',
        '14': '14th', 'fourteen': '14th', 'fourteenth': '14th',
        '22': '22nd', 'twenty two': '22nd', 'twenty two nd': '22nd',
        '28': '28th', 'twenty eight': '28th', 'twenty eightth': '28th',
    }
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, number_map.keys())) + r')\b', re.IGNORECASE)
    location_series = location_series.apply(convert_to_ordinal, args=(pattern, number_map))
    location_series = location_series.str.replace(
        r'^(.*)\s+(\d+(?:st|nd|rd|th))\s*$', r'\2 \1', regex=True
    ).str.strip()
    return location_series


def clean_location_col(location_series: pd.Series) -> pd.Series:
    """
    Clean the location column through multiple processing steps.

    Steps include converting to lowercase, extracting text after ' in ',
    applying location processing, and handling ordinal numbers.

    Parameters:
        location_series (pd.Series): Series containing location data.

    Returns:
        pd.Series: Cleaned location series.
    """
    keywords_to_remove = [
        'al bahri', 'omarat', 'investors', 'villas', 'corniche', 'ganob el', '/ red sea',
        'resorts', 'buildings', 'south of', 'east of', 'el sharkeya',
        'el-bahareya', 'el gharbeyah', 'expansion of', 'extension of', 'el sharkia', '- masr el gedida',
        'sarai compound - ', 'taj compound - ', ' - el zohour', 'villette compound - ', ' east compound',
        '.', 'hai el ashgar - ', ' west - el fayrouz', ' east - marbella', 'hai el banafsg - ',
        ' - el lazurde', 'hai el yasmen - ', ' - el zomorod', ' west - el massa', ' - el berlant',
        ' west - el yaqoot', ' east - el andalus', ' - el remas', 'hai el zohour - ', 'hai el nozha - ',
        'hai el safwa - ', 'hai el kawsr - ', ' east - granada', ' compound - marakez', ' - el orchid',
        ' east - el andalus', 'mivida compound - ', 'belle vie compound - ', 'uptown cairo compound - ',
        'cairo gate compound - ', ' el gharbeyah', 'new cairo - ', ' el gharbeyah', 'hai el nozha - ',
        ' - el ashgar', ' - west el bald', ' - tulip', 'hai el kawsr - ', 'taj compound - '
    ]
    location_series = location_series.str.lower()
    location_series = location_series.str.split(' in ').str[-1]
    location_series = location_series.apply(process_location, args=(keywords_to_remove,))
    location_series = process_location_numbers(location_series)
    return location_series


def drop_location_rows(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where 'location' contains 'phase' or has a count <= 50.

    Parameters:
        data_frame (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with specified rows removed.
    """
    data_frame = data_frame[~data_frame['location'].str.contains('phase', case=False, na=False)]
    location_counts = data_frame["location"].value_counts()
    locations_to_drop = location_counts[location_counts <= 50].index
    data_frame = data_frame[~data_frame["location"].isin(locations_to_drop)]
    return data_frame


def encode_dataset(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns using LabelEncoder and save mappings to a JSON file.

    Parameters:
        data_frame (pd.DataFrame): Input DataFrame with categorical columns.

    Returns:
        pd.DataFrame: DataFrame with encoded columns.
    """
    columns_to_encode = ['style', 'seller_type', 'view', 'payment_method', 'location']
    mappings = {}
    for column in columns_to_encode:
        encoder = LabelEncoder()
        data_frame[column] = encoder.fit_transform(data_frame[column])
        mapping = {category: int(label) for category, label in
                   zip(encoder.classes_, encoder.transform(encoder.classes_))}
        mappings[column] = mapping
    with open('../data/encoded_mapping.json', 'w') as file:
        json.dump(mappings, file, indent=4)
    return data_frame


def normalize_dataset(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize specified columns using StandardScaler.

    Parameters:
        data_frame (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with normalized columns.
    """
    columns_to_normalize = [
        'price', 'area', 'rooms', 'bathrooms', 'style', 'floor', 'apartment_age',
        'seller_type', 'view', 'payment_method', 'location'
    ]
    scaler = StandardScaler()
    data_frame[columns_to_normalize] = scaler.fit_transform(data_frame[columns_to_normalize])
    return data_frame


def save_cleaned_dataset(data_frame: pd.DataFrame) -> None:
    """
    Save the cleaned DataFrame to a CSV file and print a confirmation message.

    Parameters:
        data_frame (pd.DataFrame): Cleaned DataFrame to save.
    Returns:
        None
    """
    data_frame.to_csv('../../data/dataset.csv', index=False)
    print("dataset saved")


def remove_outliers(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows from the DataFrame where any of the specified columns have values exceeding the upper bound
     calculated using the IQR method.

    For each column in ['bathrooms', 'rooms', 'floor', 'area', 'apartment_age', 'price'], calculate the upper bound as
    the third quartile (Q3) plus 1.5 times the interquartile range (IQR), then remove rows where the column value
     exceeds this bound.

    Parameters:
        data_frame (pd.DataFrame): The input DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with rows removed where any specified column exceeds its upper bound.
    """
    cols = ['bathrooms', 'rooms', 'floor', 'area', 'apartment_age', 'price']
    for col in cols:
        q1 = data_frame[col].quantile(0.25)
        q3 = data_frame[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        data_frame = data_frame[data_frame[col] <= upper_bound]
    return data_frame


@step
def clean_dataset(dataset_path: str = '../data/raw_dataset.csv') -> None:
    """
    Load the dataset and execute the full dataset cleaning pipeline and save the result.

    Steps include:
    - Cleaning individual columns (price, area, rooms, bathrooms, floor, year built).
    - Processing the location column.
    - Dropping rows based on location criteria.
    - Encoding categorical columns.
    - Dropping specific rows by index (61760, 166145, 183962).
    - Normalizing specified columns.
    - Saving the cleaned dataset.

    Parameters:
        dataset_path (str): path of the dataset,
    Returns:
        None
    """
    if not os.path.exists('../data/dataset.csv'):
        data_frame = load_dataset(dataset_path)
        data_frame['price'] = clean_price_col(data_frame['price'])
        data_frame['area'] = clean_area_col(data_frame['area'])
        data_frame['rooms'] = clean_rooms_col(data_frame['rooms'])
        data_frame = data_frame[data_frame['rooms'] != 0]
        data_frame['bathrooms'] = clean_bathrooms_col(data_frame['bathrooms'])
        data_frame['floor'] = clean_floor_col(data_frame['floor'])
        data_frame = clean_year_built_col(data_frame)
        data_frame['location'] = clean_location_col(data_frame['location'])
        data_frame = drop_location_rows(data_frame)
        data_frame = remove_outliers(data_frame)
        # data_frame = encode_dataset(data_frame)
        # data_frame = normalize_dataset(data_frame)
        save_cleaned_dataset(data_frame)


# if __name__ == '__main__':
#     raw_dataset_path = '../../data/raw_dataset.csv'
#     clean_dataset(raw_dataset_path)
