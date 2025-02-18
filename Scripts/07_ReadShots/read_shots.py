import os
import glob
import json
import numpy as np
import pickle

def extract_tracking_data_from_file(filename):
    """
    Extracts billiard shot tracking data from a single JSON file.
    
    For each shot (Entry) in every set, this function extracts the tracking data.
    It checks if tracking data exists, converts DeltaT_500us to seconds (500 µs = 0.0005 s),
    and converts the time, x, and y lists into NumPy arrays.
    
    The result for each shot is stored with:
        - "shotID": the shot ID.
        - "balls": a dict mapping each ball's color to its data arrays ("t", "x", and "y").
        - "filename": the name of the file from which the shot was extracted.
    
    Parameters:
        filename (str): The path to the JSON file.
        
    Returns:
        list: A list of dictionaries, one per shot.
    """
    with open(filename, "r") as f:
        data = json.load(f)
    
    shots = []
    file_name = os.path.basename(filename)
    
    # Loop over each set in the match.
    for set_item in data.get("Match", {}).get("Sets", []):
        # Each shot is stored in the "Entries" list.
        for shot in set_item.get("Entries", []):
            # Check if tracking data exists.
            path_tracking = shot.get("PathTracking")
            if not path_tracking or not path_tracking.get("DataSets"):
                continue  # Skip shots without tracking data
            
            shot_id = shot.get("PathTrackingId")
            ball_data = {}
            
            # Process each ball's dataset.
            for dataset in path_tracking.get("DataSets", []):
                ball_color = dataset.get("BallColor")
                times, xs, ys = [], [], []
                coords = dataset.get("Coords")
                if not coords:
                    continue  # Skip if no coordinates are present
                
                for coord in coords:
                    times.append(coord.get("DeltaT_500us", 0) * 0.0005)
                    xs.append(coord.get("X")*2840)
                    ys.append(coord.get("Y")*1420)
                
                ball_data[ball_color] = {
                    "t": np.array(times),
                    "x": np.array(xs),
                    "y": np.array(ys)
                }
            
            if ball_data:
                shots.append({
                    "shotID": shot_id,
                    "balls": ball_data,
                    "filename": file_name
                })
    
    return shots

def extract_all_shots_from_folder(folder):
    """
    Iterates through all JSON files in the given folder, extracts shot data from each,
    prints the number of shots extracted per file, and aggregates them into a single list.
    
    Parameters:
        folder (str): The path to the folder containing JSON files.
        
    Returns:
        list: A list containing the shot data dictionaries from all files.
    """
    all_shots = []
    json_files = glob.glob(os.path.join(folder, "*.json"))
    
    for json_file in json_files:
        shots = extract_tracking_data_from_file(json_file)
        print(f"File: {os.path.basename(json_file)} - Shots extracted: {len(shots)}")
        all_shots.extend(shots)
    
    return all_shots

if __name__ == '__main__':
    folder = "BilliardGamesData"
    all_shots = extract_all_shots_from_folder(folder)
    
        # Save to a file
    with open("all_shots.pkl", "wb") as f:
        pickle.dump(all_shots, f)

    # Optionally, print a summary of the total number of shots extracted
    print(f"\nTotal shots extracted from all files: {len(all_shots)}")

    # # Load the shots from the file
    # with open("all_shots.pkl", "rb") as f:
    #     shots = pickle.load(f)

    # # Example: Access the first shot
    # print("Shot ID:", shots[0]["shotID"])
    # print("From File:", shots[0]["filename"])

    # # Example: Access ball tracking data
    # for ball_color, data in shots[0]["balls"].items():
    #     print(f"Ball {ball_color}:")
    #     print("  Time (s):", data["t"])
    #     print("  X (m):", data["x"])
    #     print("  Y (m):", data["y"])
