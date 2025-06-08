import json
from scripts.persistence import recall_last_run, read_ml_data, write_ml_data
import math
import sys

def delete_ml_sales_for_recalled_release(points_to_delete_json):
    """
    Deletes sales entries from the ML data file (ml_data.pkl) for a recalled release
    if those sales match points marked for deletion.

    This is used when points are deleted from the chart, but no new Discogs data
    is provided, implying the deletions pertain to the last loaded release.

    Args:
        points_to_delete_json (str): A JSON string representing a list of points
                                     (dictionaries) to be deleted. Each point dict
                                     should have 'date', 'price', and 'quality' keys.
    """
    if not points_to_delete_json:
        print("Info (ML Delete): No points_to_delete_json provided. No ML sales to delete.", file=sys.stderr)
        return 0 # Return 0 deleted

    try:
        points_to_delete = json.loads(points_to_delete_json)
        if not isinstance(points_to_delete, list) or not points_to_delete:
            print("Info (ML Delete): Parsed points_to_delete is empty or not a list. No ML sales to delete.", file=sys.stderr)
            return 0
    except json.JSONDecodeError as e:
        print(f"Error (ML Delete): Could not decode points_to_delete_json: {e}. No ML sales will be deleted.", file=sys.stderr)
        return 0

    # 1. Retrieve the Last Processed Discogs ID
    recalled_info = recall_last_run()
    discogs_id_from_recall = recalled_info.get("discogs_id")

    if not discogs_id_from_recall:
        print("Info (ML Delete): No Discogs ID found in recalled data. Cannot determine which release's ML sales to modify.", file=sys.stderr)
        return 0

    release_key = str(discogs_id_from_recall)
    print(f"Info (ML Delete): Attempting to delete ML sales for recalled Release ID: {release_key}", file=sys.stderr)

    # 2. Load ML Data
    all_releases_data = read_ml_data()
    if not isinstance(all_releases_data, dict):
        print("Error (ML Delete): ML data file is not a dictionary. Cannot proceed.", file=sys.stderr)
        return 0

    # 3. Access Specific Release Entry
    release_entry = all_releases_data.get(release_key)
    if not release_entry:
        print(f"Info (ML Delete): No ML entry found for Release ID: {release_key}. Nothing to delete.", file=sys.stderr)
        return 0

    original_sales_history = release_entry.get('sales_history')
    if not isinstance(original_sales_history, list):
        print(f"Info (ML Delete): 'sales_history' for Release ID {release_key} is missing or not a list. Nothing to delete.", file=sys.stderr)
        return 0
    if not original_sales_history:
        print(f"Info (ML Delete): 'sales_history' for Release ID {release_key} is empty. Nothing to delete.", file=sys.stderr)
        return 0

    # 4. Filter sales_history
    new_sales_history = []
    ml_sales_deleted_count = 0

    for sale_dict in original_sales_history:
        # Prepare sale_dict in a format points_match expects for a 'grid_row'
        # points_match expects: [date, N, N, price_adj, native_price, quality_score, comment]
        temp_row_representation = [
            sale_dict.get('date'),
            None,  # Placeholder for quality1_str_raw
            None,  # Placeholder for quality2_str_raw
            sale_dict.get('price'),        # Inflation-adjusted price
            sale_dict.get('native_price', ''), # Native price
            sale_dict.get('quality'),      # Quality score
            sale_dict.get('sale_comment', '') # Sale comment
        ]

        matched_for_deletion = False
        for point_to_delete_item in points_to_delete:
            if points_match(temp_row_representation, point_to_delete_item):
                matched_for_deletion = True
                break

        if not matched_for_deletion:
            new_sales_history.append(sale_dict)
        else:
            ml_sales_deleted_count += 1

    # 5. Update and Save ML Data if changes were made
    if ml_sales_deleted_count > 0:
        release_entry['sales_history'] = new_sales_history
        # all_releases_data[release_key] = release_entry # Already modified in place
        write_ml_data(all_releases_data)
        artist_name_for_log = release_entry.get('api_artist', 'N/A')
        album_title_for_log = release_entry.get('api_title', 'N/A')
        print(f"Info (ML Delete): Deleted {ml_sales_deleted_count} sales from ML data for Release ID '{release_key}' ({artist_name_for_log} - {album_title_for_log}). ML data file updated.", file=sys.stderr)
    else:
        print(f"Info (ML Delete): No matching sales found in ML data for Release ID '{release_key}' to delete based on the provided points.", file=sys.stderr)

def points_match(grid_row, point_to_delete, tolerance=0.001):
    """
    Checks if a row from the processed grid matches a point selected for deletion.
    Compares date, quality score, and price. Comment matching is optional/flexible.
    """
    if len(grid_row) < 6:  # Needs at least date, price, score
        return False

    grid_score = grid_row[5]
    grid_price = grid_row[3]  # Inflation-adjusted price
    grid_date = grid_row[0]
    # grid_comment = grid_row[6] if len(grid_row) > 6 else "" # Comment matching can be tricky

    delete_score = point_to_delete.get('quality')
    delete_price = point_to_delete.get('price')  # This is the inflation-adjusted price from the chart
    delete_date = point_to_delete.get('date')
    # delete_comment = point_to_delete.get('comment', "")

    if delete_score is None or delete_price is None or delete_date is None:
        return False

    # Ensure types are compatible for comparison, especially for floats
    try:
        grid_score_float = float(grid_score)
        delete_score_float = float(delete_score)
        grid_price_float = float(grid_price) if grid_price is not None else -1  # Handle None grid_price
        delete_price_float = float(delete_price)
    except (ValueError, TypeError):
        return False  # Cannot compare if conversion fails

    score_match = math.isclose(grid_score_float, delete_score_float, rel_tol=tolerance)
    price_match = math.isclose(grid_price_float, delete_price_float,
                               rel_tol=tolerance) if grid_price is not None else False
    date_match = (grid_date == delete_date)
    # comment_match = ((grid_comment or "") == (delete_comment or "")) # Keep comment matching simple or remove

    return score_match and price_match and date_match  # and comment_match

def delete_points(points_to_delete_json, processed_grid):
    """
    Filters the processed grid, removing rows that match points marked for deletion.
    """
    points_to_delete = []
    if points_to_delete_json:
        try:
            points_to_delete = json.loads(points_to_delete_json)
        except json.JSONDecodeError:
            print("Warning: Could not decode points_to_delete JSON. Proceeding without deleting points.",
                  file=sys.stderr)
            points_to_delete = []

    deleted_count = 0

    if points_to_delete and processed_grid:
        initial_count = len(processed_grid)
        filtered_grid = []
        for row in processed_grid:
            should_delete = False
            for point in points_to_delete:
                if points_match(row, point):
                    should_delete = True
                    break
            if not should_delete:
                filtered_grid.append(row)

        deleted_count = initial_count - len(filtered_grid)
        if deleted_count > 0:
            print(f"Info: Deleted {deleted_count} point(s) based on selection.", file=sys.stderr)
        processed_grid = filtered_grid
    return processed_grid, deleted_count