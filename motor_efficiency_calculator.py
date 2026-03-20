import requests
from bs4 import BeautifulSoup


def get_motor_efficiency(voltage, target_current):
    """
    Fetches motor performance data for a given voltage from lehner-motoren.com
    and returns the efficiency for the current closest to the target_current.
    """
    # Construct the URL with the given voltage input
    url = f"https://www.lehner-motoren.com/calc/graph/liste.php?t=2260&w=40&sp={voltage}"

    # Fetch the webpage
    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP errors

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the data table on the page
    table = soup.find('table')
    if not table:
        raise ValueError("Could not find the data table on the webpage.")

    min_diff = float('inf')
    closest_efficiency = None
    matched_current = None

    # Iterate through all table rows, skipping the header row
    rows = table.find_all('tr')[1:]

    for row in rows:
        cols = row.find_all('td')
        # Ensure the row has enough columns (Current is idx 0, Efficiency is idx 5)
        if len(cols) >= 6:
            try:
                # Extract and clean the text to convert to float
                row_current = float(cols[0].text.strip())
                row_efficiency = float(cols[5].text.strip())

                # Calculate the difference from the target current
                diff = abs(row_current - target_current)

                # Update closest match if this row is closer
                if diff < min_diff:
                    min_diff = diff
                    closest_efficiency = row_efficiency
                    matched_current = row_current

            except ValueError:
                # Ignore rows where the text can't be parsed into a float
                continue

    if closest_efficiency is not None:
        #print(f"Matched closest current: {matched_current} A (Target: {target_current} A), Efficiency: {closest_efficiency} %")
        return closest_efficiency/100
    else:
        raise ValueError("Could not extract valid data from the table.")


# --- Example Usage ---
if __name__ == "__main__":
    voltage_input = 40
    current_input = 3.1

    efficiency = get_motor_efficiency(voltage_input, current_input)
    print(f"Efficiency: {efficiency}")