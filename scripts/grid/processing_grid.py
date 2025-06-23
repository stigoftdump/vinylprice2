import json
import sys

from scripts.persistence import (
    recall_last_run,
    remember_last_run,
    machine_learning_save
)
from scripts.grid.grid_functions import (
    is_valid_discogs_input,
    make_processed_grid,
    delete_points,
    merge_and_deduplicate_grids
)
from scripts.grid.api_import import fetch_api_data, fake_api_data
from scripts.grid.point_deletion import delete_ml_sales_for_recalled_release

def _load_initial_data_state():
    """Loads the initial data state from the last run.
    Responsibility: Load and return persisted application state.
    """
    recalled_data = recall_last_run()
    initial_grid = recalled_data.get("processed_grid", [])
    initial_api_data = {
        "api_artist": recalled_data.get("artist"),
        "api_title": recalled_data.get("title"),
        "api_year": recalled_data.get("year"),
        "api_original_year": recalled_data.get("original_year")
    }
    recalled_discogs_id = recalled_data.get("discogs_id")
    return initial_grid, initial_api_data, recalled_discogs_id


def _check_full_info(discogs_data, discogs_release_id):
    """Checks if full Discogs information is provided.
    Responsibility: Validate Discogs data and ID presence.
    """
    return is_valid_discogs_input(discogs_data, discogs_release_id)


def _process_new_data_and_api(discogs_data_input, start_date_input, discogs_release_id_input):
    """Processes newly provided Discogs data and fetches API information.
    Responsibility: Parse new sales data and fetch corresponding API data.
    """
    messages = []
    newly_parsed_grid, parsing_status = make_processed_grid(discogs_data_input, start_date_input)
    if parsing_status:
        messages.append(f"Parsing new data: {parsing_status}")

    api_data = fetch_api_data(discogs_release_id_input)
    if not api_data:
        messages.append("API data could not be fetched for the new ID. Using fallback.")
        api_data = fake_api_data()
    else:
        messages.append(f"Successfully fetched API data for ID: {discogs_release_id_input}.")
    return newly_parsed_grid, api_data, messages


def _are_points_marked_for_deletion(points_to_delete_json_input):
    """Checks if any points are marked for deletion from the chart.
    Responsibility: Determine if chart point deletion is requested.
    """
    if not points_to_delete_json_input:
        return False
    try:
        points = json.loads(points_to_delete_json_input)
        return bool(points)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse points_to_delete_json: {points_to_delete_json_input}", file=sys.stderr)
        return False


def _should_add_new_data_flag(add_data_flag_input_str):
    """Determines if new data should be added to existing data.
    Responsibility: Interpret the 'add data' flag.
    """
    return add_data_flag_input_str == "True"


def _handle_point_deletion_branch(
        current_grid,
        current_api_data,
        current_id_context,
        points_to_delete_json,
        recalled_id_context_for_m1,  # Explicitly pass the ID for M1 deletion context
        status_messages_list):
    """
    Handles the logic for deleting points from chart and M1 data.
    Responsibility: Manage chart point deletion, M1 data deletion, and related state saving.
    Returns: (updated_grid, chart_points_deleted_count, early_exit_flag)
    """
    # First, delete points from the in-memory grid for immediate display and use.
    grid_after_chart_deletion, chart_points_deleted_count = delete_points(
        points_to_delete_json, current_grid
    )
    if chart_points_deleted_count > 0:
        status_messages_list.append(f"{chart_points_deleted_count} points deleted from chart data.")

    # Second, run the ML-specific deletion. This function reads the original ML file,
    # filters the sales history, and writes the updated data back. This is now the
    # single source of truth for updating the ML file in this deletion branch.
    m1_points_deleted_count = delete_ml_sales_for_recalled_release(recalled_id_context_for_m1, points_to_delete_json)

    # We have removed the call to `machine_learning_save` from this branch.
    # It is redundant because `delete_ml_sales_for_recalled_release` handles
    # the ML file update, and calling it would overwrite the changes.

    # Finally, save the modified grid for the next run's "recall_last_run"
    remember_last_run(grid_after_chart_deletion, current_id_context,
                      current_api_data.get("api_artist"),
                      current_api_data.get("api_title"),
                      current_api_data.get("api_year"),
                      current_api_data.get("api_original_year"))
    status_messages_list.append("Current data state saved after point deletions.")

    return grid_after_chart_deletion, chart_points_deleted_count, True  # True indicates early exit

def _determine_final_grid_for_display(
        grid_before_add_decision,
        add_data_flag,
        is_full_info_provided_flag,
        initial_grid_from_load,  # Pass the initially loaded grid for merging
        status_messages_list):
    """Determines the final grid based on the 'Add Data?' decision.
    Responsibility: Combine or select data based on the 'add data' flag.
    """
    if add_data_flag:
        status_messages_list.append("Add Data: Yes.")
        if is_full_info_provided_flag:  # Implies grid_before_add_decision is newly_parsed_grid
            final_grid = merge_and_deduplicate_grids(
                initial_grid_from_load,  # Use the initially loaded grid
                grid_before_add_decision
            )
            status_messages_list.append("Combined newly processed data with previously saved data.")
        else:  # Not full info, so grid_before_add_decision is already initial_old_grid.
            final_grid = grid_before_add_decision
            status_messages_list.append(
                "Using previously saved data (no new data to parse and combine for 'Add Data: Yes').")
    else:  # Add Data: No
        status_messages_list.append("Add Data: No.")
        final_grid = grid_before_add_decision
        if is_full_info_provided_flag:
            status_messages_list.append("Using only newly processed data (not adding to previous).")
        else:
            status_messages_list.append("Using only previously saved data (not adding, no new data).")
    return final_grid


def _save_final_state(
        final_grid,
        id_context,
        api_data_context,
        status_messages_list):
    """Saves the final state of the data.
    Responsibility: Persist final grid to ML store and for 'remember_last_run'.
    """
    machine_learning_save(final_grid, id_context, api_data_context)
    status_messages_list.append(f"Final sales data for context ID {id_context or 'N/A'} updated in ML store.")

    remember_last_run(final_grid, id_context,
                      api_data_context.get("api_artist"),
                      api_data_context.get("api_title"),
                      api_data_context.get("api_year"),
                      api_data_context.get("api_original_year"))
    status_messages_list.append("Final data state saved.")


# --- Main Orchestrating Function (SRP: Orchestration) ---

def execute_data_processing_flow(
        discogs_data_input,
        discogs_release_id_input,
        points_to_delete_json_input,
        add_data_flag_input_str,
        start_date_input
):
    """
    Orchestrates the data processing logic by delegating to SRP-focused functions.
    Returns:
        tuple: (final_grid_for_display, final_api_data_for_display, status_messages_list, chart_points_deleted_count)
    """
    status_messages = []
    chart_points_deleted_count = 0
    early_exit_from_flow = False

    initial_grid, initial_api_data, recalled_discogs_id = _load_initial_data_state()
    status_messages.append("Initial data state loaded from last run.")

    # Initialize working context variables
    current_grid_for_processing = initial_grid
    current_api_for_processing = initial_api_data
    current_id_for_processing = recalled_discogs_id

    is_full_info = _check_full_info(discogs_data_input, discogs_release_id_input)

    if is_full_info:
        status_messages.append(f"Full info provided (Discogs data and ID: {discogs_release_id_input}).")
        current_id_for_processing = discogs_release_id_input

        newly_parsed_grid, fetched_api_data, processing_msgs = _process_new_data_and_api(
            discogs_data_input, start_date_input, discogs_release_id_input
        )
        status_messages.extend(processing_msgs)
        current_api_for_processing = fetched_api_data
        current_grid_for_processing = newly_parsed_grid  # This is the data to potentially delete from or add to

        if _are_points_marked_for_deletion(points_to_delete_json_input):
            status_messages.append("Points marked for deletion from newly processed data.")
            current_grid_for_processing, chart_points_deleted_count, early_exit_from_flow = \
                _handle_point_deletion_branch(
                    current_grid_for_processing, current_api_for_processing, current_id_for_processing,
                    points_to_delete_json_input, recalled_discogs_id, status_messages
                )
        else:
            status_messages.append("No points marked for deletion from newly processed data.")
            # grid_before_add_decision will be current_grid_for_processing (newly_parsed_grid)
    else:  # Not Full Info
        status_messages.append("Partial info (Discogs data or ID missing). Using loaded old data as base.")
        # current_grid_for_processing, current_api_for_processing, current_id_for_processing
        # remain as initially loaded.

        if _are_points_marked_for_deletion(points_to_delete_json_input):
            status_messages.append("Points marked for deletion from loaded old data.")
            current_grid_for_processing, chart_points_deleted_count, early_exit_from_flow = \
                _handle_point_deletion_branch(
                    current_grid_for_processing, current_api_for_processing, current_id_for_processing,
                    points_to_delete_json_input, recalled_discogs_id, status_messages
                )
        else:
            status_messages.append("No points marked for deletion from loaded old data.")
            # grid_before_add_decision will be current_grid_for_processing (initial_grid)

    # If an early exit occurred (due to point deletion path completing), return now.
    if early_exit_from_flow:
        status_messages.append("Process finished after point deletion branch.")
        return current_grid_for_processing, current_api_for_processing, status_messages, chart_points_deleted_count

    # --- Common Path: "Decision: Add Data?" ---
    # current_grid_for_processing is now the grid_before_add_decision
    should_add_data = _should_add_new_data_flag(add_data_flag_input_str)
    final_grid = _determine_final_grid_for_display(
        current_grid_for_processing,  # This is grid_before_add_decision
        should_add_data,
        is_full_info,
        initial_grid,  # Pass the original loaded grid for merging if needed
        status_messages
    )

    _save_final_state(
        final_grid,
        current_id_for_processing,
        current_api_for_processing,
        status_messages
    )

    status_messages.append("Process Completed.")
    return final_grid, current_api_for_processing, status_messages, chart_points_deleted_count
