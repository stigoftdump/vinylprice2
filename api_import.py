import sys
import discogs_client
from discogs_secrets import DISCOGS_USER_TOKEN

def fetch_api_data(release_id):
    """
    Fetches detailed information for a given release ID from the Discogs API,
    including the original year from its master release.

    Args:
        release_id (str or int): The Discogs release ID.

    Returns:
        dict: A dictionary containing extracted release information (e.g., api_year,
              api_artist, api_title, api_genres, api_styles, api_country, api_label,
              api_catno, api_format_descriptions, api_notes, api_community_have,
              api_community_want, api_community_rating_average, api_community_rating_count,
              api_master_id, api_original_year),
              or None if an error occurs or ID is not found.
    """
    if not release_id:
        return None

    try:
        d = discogs_client.Client('VinylPriceCalculator/0.1', user_token=DISCOGS_USER_TOKEN)
        release = d.release(int(release_id))
        api_data = {}

        api_data['api_year'] = getattr(release, 'year', None)
        api_data['api_title'] = getattr(release, 'title', None)
        api_data['api_artist'] = getattr(release.artists[0], 'name', None) if release.artists else None
        api_data['api_genres'] = getattr(release, 'genres', [])
        api_data['api_styles'] = getattr(release, 'styles', [])
        api_data['api_country'] = getattr(release, 'country', None)

        if release.labels:
            api_data['api_label'] = getattr(release.labels[0], 'name', None)
            api_data['api_catno'] = getattr(release.labels[0], 'catno', None)
        else:
            api_data['api_label'] = None
            api_data['api_catno'] = None

        format_descriptions = []
        if release.formats:
            for fmt in release.formats:
                if fmt.get('descriptions'):
                    format_descriptions.extend(fmt['descriptions'])
        api_data['api_format_descriptions'] = list(set(format_descriptions))  # Deduplicate

        api_data['api_notes'] = release.data.get('notes', None)  # Notes often in .data

        community_data = release.data.get('community', {})
        api_data['api_community_have'] = community_data.get('have', None)
        api_data['api_community_want'] = community_data.get('want', None)
        rating_data = community_data.get('rating', {})
        api_data['api_community_rating_average'] = rating_data.get('average', None)
        api_data['api_community_rating_count'] = rating_data.get('count', None)

        master_id = getattr(release, 'master_id', None)
        if not master_id or master_id == 0:  # Check .data if not on object or is 0
            master_id_from_data = release.data.get('master_id')
            if master_id_from_data and master_id_from_data != 0:
                master_id = master_id_from_data
            else:
                master_id = None

        api_data['api_master_id'] = master_id
        api_data['api_original_year'] = None

        if master_id:
            print(
                f"Info: Found Master ID: {master_id} for Release ID: {release_id}. Fetching master release details...")
            try:
                master_release = d.master(int(master_id))
                master_release.refresh()  # Crucial step to get all data for the master release
                print(f"Info: Refresh complete for Master ID: {master_id}.")

                original_year_from_attr = getattr(master_release, 'year', None)
                original_year_from_data = master_release.data.get('year')

                if original_year_from_data and original_year_from_data != 0:
                    api_data['api_original_year'] = original_year_from_data
                elif original_year_from_attr and original_year_from_attr != 0:  # Fallback to attribute if .data failed
                    api_data['api_original_year'] = original_year_from_attr

                if api_data['api_original_year']:
                    print(
                        f"Info: Successfully fetched original year: {api_data['api_original_year']} for Master ID: {master_id}")
                elif original_year_from_data == 0 or original_year_from_attr == 0:  # Explicitly 0
                    print(f"Warning: Master ID {master_id} (Release ID: {release_id}) has 'year' as 0.",
                          file=sys.stderr)
                else:  # Both were None
                    print(
                        f"Warning: Master ID {master_id} (Release ID: {release_id}) found, but 'year' is missing or None after refresh.",
                        file=sys.stderr)

            except discogs_client.exceptions.HTTPError as master_http_err:
                if master_http_err.status_code == 404:
                    print(
                        f"Warning: Discogs API Error - Master ID {master_id} (for Release ID: {release_id}) not found (404).",
                        file=sys.stderr)
                else:
                    print(
                        f"Warning: Discogs API HTTP Error for Master ID {master_id} (for Release ID: {release_id}): {master_http_err}",
                        file=sys.stderr)
            except Exception as master_e:
                print(
                    f"Warning: An unexpected error occurred while fetching Master ID {master_id} (for Release ID: {release_id}): {master_e}",
                    file=sys.stderr)
        else:
            print(f"Info: No valid Master ID found on Release ID: {release_id}.")

        print(f"Info: Successfully processed API data for release ID: {release_id}")
        return api_data

    except discogs_client.exceptions.HTTPError as http_err:
        if http_err.status_code == 404:
            print(f"Warning: Discogs API Error - Release ID {release_id} not found (404).", file=sys.stderr)
        elif http_err.status_code == 401:
            print(f"Error: Discogs API Error - Unauthorized (401). Check your User Token.", file=sys.stderr)
        else:
            print(f"Warning: Discogs API HTTP Error for Release ID {release_id}: {http_err}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while fetching API data for Release ID {release_id}: {e}",
              file=sys.stderr)
        return None
