import requests

# Define the fields required by the API and their expected data types
fields = {
    "location": str,
    "area": float,
    "rooms": int,
    "bathrooms": int,
    "style": str,
    "floor": int,
    "year_built": int,
    "seller_type": str,
    "view": str,
    "payment_method": str
}


def get_user_input():
    """
    Prompt the user to enter values for each field and ensure they are of the correct type.
    Returns a dictionary with the user-provided data.
    """
    data = {}
    for field, dtype in fields.items():
        while True:
            try:
                value = input(f"Enter {field}: ")
                if dtype == int:
                    data[field] = int(value)
                elif dtype == float:
                    data[field] = float(value)
                else:
                    data[field] = value  # No conversion needed for strings
                break
            except ValueError:
                print(f"Invalid input for {field}. Please enter a valid {dtype.__name__}.")
    return data


def main():
    # API endpoint URL (assumes the API is running locally; adjust as needed)
    url = "http://localhost:8000/predict"

    # Get user input
    print("Please provide the following details for prediction:")
    data = get_user_input()

    try:
        # Send POST request to the API with the data as JSON
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes

        # Parse and display the successful response
        result = response.json()
        print(f"\nPrediction: {result['prediction']}")
        print(f"Apartment Age: {result['apartment_age']}")

    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors (e.g., validation errors or server issues)
        print(f"HTTP Error: {e}")
        try:
            error_detail = response.json()
            print(f"Error Detail: {error_detail}")
        except ValueError:
            print(f"Response Text: {response.text}")

    except requests.exceptions.RequestException as e:
        # Handle network-related errors (e.g., connection issues)
        print(f"Request failed: {e}")


if __name__ == "__main__":
    main()
