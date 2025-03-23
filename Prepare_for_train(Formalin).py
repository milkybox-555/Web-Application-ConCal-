import cv2
import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split, GridSearchCV,LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor 
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

#Noted that all vaarible name are define by "ton"
start_time = time.time() #Track run time 
#Define main_dir and na of each folder #Note: follow instruction 1 in msteam 
main_dir = os.path.join(".", "newdata") #main folder ***
#Created output folder
INPUT_DIR = os.path.join(main_dir, "Formalin") #Input subfolder raw image ***

model_main_dir = os.path.join(".", "static")
folders = ["Air_Real_Crop", "Air_Real_Mark", "Air_Crop_Mark", "Air_Crop_Warp", 
    "Air_Crop_Warp_add", "Anal_real_mark", "Anal_real_warp", "Anal_real_warp_add"]  
for folder in folders:
    folder_path = os.path.join(model_main_dir, folder)
    os.makedirs(folder_path, exist_ok=True)
OUTPUT_CROP_IMAGE = os.path.join(model_main_dir, "Air_Real_Crop") #Output crop area of air image
OUTPUT_REAL_MARK = os.path.join(model_main_dir, "Air_Real_Mark") # Output full image with mark Air and analysis image
OUTPUT_Crop_MARKED = os.path.join(model_main_dir, "Air_Crop_Mark") 
OUTPUT_Crop_WARP = os.path.join(model_main_dir, "Air_Crop_Warp") # Output Air with warp perspective
output_crop_Warp_add = os.path.join(model_main_dir, "Air_Crop_Warp_add") # Output Warp air image with add condition
OUTPUT_Anal_real = os.path.join(model_main_dir, "Anal_real_mark") # Output full image with mark analysis image
OUTPUT_ANAL_WARP = os.path.join(model_main_dir, "Anal_real_warp") # Output Analysis with warp perspective
OUTPUT_ANAL_WARP_add = os.path.join(model_main_dir, "Anal_real_warp_add") # Output Warp Analysis image with add conditio  
MODEL = os.path.join(model_main_dir, "model")
OUTPUT_L = os.path.join(model_main_dir, 'data_L.xlsx')
#Define value for image pre-processing #Note: follow instruction 2 in msteam
height_solution_ratio = 1.2 #height of crop box for solution if not use put on 1
width_solution_ratio = 1.4 #width of crop box for solution if not use put on 1
distance_x = 475 #distance in x axis from air center to solution center neglect elevation
more_red =-6 #9.5 #adjust to give more red in solution image if not use put on 0
rotate_counter_clockwise = 1 #1.2 #adjust number for image rotate cc if not use put on 1
rotate_clockwise = 0 #0.025 # adjust number for image rotate c if not use put on 0
angle_adjustment = 0 #2 #adjust output solution image orientation if not use put on 0
perfect_morph = 0 #0.01 #for image that have rotation but after morph give perfect no rotate rectangular if not use put on 0
new_angle = 0.00
#Delete all image in last running code
directories = [OUTPUT_CROP_IMAGE, OUTPUT_REAL_MARK, OUTPUT_Crop_MARKED, OUTPUT_Crop_WARP, output_crop_Warp_add, 
                OUTPUT_Anal_real, OUTPUT_ANAL_WARP, OUTPUT_ANAL_WARP_add] #Define subfolder to delete image inside
for directory in directories: #loop through each data in directories
    for filename in os.listdir(directory): #Loop each image
        file_path = os.path.join(directory, filename) # Define path for each image
        try: # Define condition for delete image
            if os.path.isfile(file_path): 
                os.remove(file_path)
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}")
#Objective: Crop Reference spectral band image. Input: Raw Image Output: Reference Spectral band crop areaa           
def Crop_by_avg(Raw_Image): #input: Full raw image from experiment
    lab_image = cv2.cvtColor(Raw_Image, cv2.COLOR_BGR2LAB) # Read image with LAB system
    L_channel = lab_image[:,:,0] # Pick only axis of L value
    avg_L_values_horizontal = L_channel.mean(axis=0).flatten() # Average L value in axis = 0 (row)
    avg_L_values_vertical = L_channel.mean(axis=1).flatten() # Average L value in axis = 1 (Columns)
    #horizontal pixels process
    pixel_columns_horizontal_df = pd.DataFrame() # Crate Blank Dataframe to store average L value 
    pixel_columns_horizontal_df['avg_L_values_horizontal'] = avg_L_values_horizontal # Store value in Blank DataFrame
    pixel_columns_horizontal_df = pixel_columns_horizontal_df.apply(pd.to_numeric).T # Transpose for loop find pixel number
    #vertical pixels process 
    pixel_columns_vertical_df = pd.DataFrame() # Crate Blank Dataframe to store average L value 
    pixel_columns_vertical_df['avg_L_values_vertical'] = avg_L_values_vertical # Store value in Blank DataFrame
    pixel_columns_vertical_df = pixel_columns_vertical_df.apply(pd.to_numeric).T # Transpose for loop find pixel number
    column_max_df_vertical = pd.DataFrame() # Define Blank DataFrame for contain max vertical for each image
    for index, row in pixel_columns_horizontal_df.iterrows(): # loop through the DataFraame
        max_column_number = row.idxmax() # find pixel with heightest average L value
    for row_index, Column in pixel_columns_vertical_df.iterrows(): # loop through the DataFraame
        max_column_number_ver = Column.idxmax() # find pixel with heightest average L value 
        max_column_number_ver = int(max_column_number_ver) #int
        max_column_number_new = max_column_number_ver - 100 # DeFine heightest for only left side of image
        column_max_df_vertical.at[row_index, 'y'] = max_column_number_new # get Heightest DataFrame pixel for each image
    for row_index, column in pixel_columns_vertical_df.iterrows(): # Same Horizontral
        max_column_number_new = int(column_max_df_vertical.at[row_index, 'y'])  # Same Horizontral
        new_data_slice = column[:max_column_number_new + 1]  # Same Horizontral
        new_max_column_number_vertical = new_data_slice.idxmax()  # Same Horizontral

    x,y = max_column_number, new_max_column_number_vertical # get position for center point area of crop image
    width = 600 # Define area for crop
    height = 336 # Define area for crop
    half_width= width // 2 # Define area for crop
    half_height = height // 2 # Define area for crop
    top_left_x = x - half_width # Define area for crop
    top_left_y = y - half_height # Define area for crop
    bottom_right_x = x + half_width # Define area for crop
    bottom_right_y = y + half_height # Define area for crop
    crop_box = (top_left_x, top_left_y, bottom_right_x, bottom_right_y) # Define area for crop
    cropped_img = Raw_Image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] # Define area for crop
    return cropped_img, crop_box # Two output is Crop image and crop position respectively
# Objective: Find contour of the Desired image
def preprocess_image(img): # #input: Full raw image from experiment
    cropimage,Crop_box = Crop_by_avg(img) #Sent full raw image to Crop_by_avg function for cropimage
    blurred = cv2.GaussianBlur(cropimage, (5, 5), 0) #Blur
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY) # turn to gray scale system
    _, thresh = cv2.threshold(gray,12, 255, cv2.THRESH_BINARY) # Defind Threshold for seperate forground and blackground 
    kernel = np.ones((5,5), np.uint8) # Define kernal for mophyology
    OPEN = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3) #use opening method 
    closing = cv2.morphologyEx(OPEN, cv2.MORPH_CLOSE, kernel, iterations=3) #use Colseing method  
    return closing # Output: contour or desired crop image
# Objective: Find corners of the Desired square image
def find_corners(img): # input: Full raw image from experiment
    processed_img = preprocess_image(img) # Sent image to processed_img for contour of dessired image
    contours, _ = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Use find contour to find corners
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contour = contours[0]
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    return box # Output array of position in each corners of square contour
# Objective: Find distance between each corners 
def Linedistance(corners): # input: array of possition in each corners of square contour
# Find distance 1 #Noted: Value change everytime the image have angle. Check varible in corners to know each distance.
    distance1 = np.linalg.norm(corners[1] - corners[0]) 
    distance2 = np.linalg.norm(corners[2] - corners[1])
    distance3 = np.linalg.norm(corners[3] - corners[2])
    distance4 = np.linalg.norm(corners[3] - corners[0])
    distances = np.array([distance1, distance2, distance3, distance4]) # Combine all distance to array
    return distances # Output: Array of distance
# Objective: Reorder the corners of square contour
def corner_reorder(corners): # input: array of corners
    corners = find_corners(img) # Find corners of the square contour
    distances = Linedistance(corners) # find distance between corners
    if distances[0] < distances[1]: # Define condition based on the square contour
        recorners = np.array([corners[1], corners[2], corners[3], corners[0]])
    else:
        recorners = np.array([corners[2], corners[3], corners[0], corners[1]])
    return recorners # OutputL array of reordered corners
# Objective: turn reordered_corners(8 parameters) in to 5 parameter
def FiveParam(reordered_corners): # input: array of reordered_corners
    distanceheight = np.linalg.norm(reordered_corners[0] - reordered_corners[1]) # Find height of square image
    distancewidth = np.linalg.norm(reordered_corners[0] - reordered_corners[3]) # Find Width of square image
    center = (reordered_corners[3] + ([[distancewidth/2, distanceheight/2]])) # Find Center point of square
    ref_distance = reordered_corners[1][0] - reordered_corners[2][0] # Find Horizontal reference line for angle finding
    angledistance = np.linalg.norm(reordered_corners[1] - reordered_corners[2]) # Find distance of angle finding
    if reordered_corners[0][1] > reordered_corners[3][1]: # Define when angle are negative
        anglevalue = -np.arccos(ref_distance/angledistance) #*180/np.pi
    else :
        anglevalue = np.arccos(ref_distance/angledistance) # Define angle when positive
    return center,distancewidth,distanceheight,anglevalue # get all 5 parameter
# Objective: Find 5 parameter in analysis image with center point of analysis image
def CenterToFiveParamFormRef(new_center_point, old_reordered_corners, old_angle, height_solution_ratio, width_solution_ratio, more_red): 
    #Input: center point of analysis image, and all 3 parameter from Air image with adjusted values
    distanceheight = np.linalg.norm(old_reordered_corners[0] - old_reordered_corners[1]) # Define height
    distanceheight = distanceheight*height_solution_ratio + more_red # Adjusted for ratio of height
    distancewidth = np.linalg.norm(old_reordered_corners[0] - old_reordered_corners[3]) # Define width 
    distancewidth = distancewidth*width_solution_ratio # Adjusted number for ratio of width
    #old_angle = old_angle  #if need can give adjusted number for change angle
    # find  new corner from old angle
    x0_new = new_center_point[0][0] + distancewidth/2 * np.cos(old_angle) - distanceheight/2 * np.sin(old_angle)
    y0_new = new_center_point[0][1] - distancewidth/2 * np.sin(old_angle) - distanceheight/2 * np.cos(old_angle)
    x1_new = new_center_point[0][0] + distancewidth/2 * np.cos(old_angle) + distanceheight/2 * np.sin(old_angle)
    y1_new = new_center_point[0][1] - distancewidth/2 * np.sin(old_angle) + distanceheight/2 * np.cos(old_angle)
    x2_new = new_center_point[0][0] - distancewidth/2 * np.cos(old_angle) + distanceheight/2 * np.sin(old_angle)
    y2_new = new_center_point[0][1] + distancewidth/2 * np.sin(old_angle) + distanceheight/2 * np.cos(old_angle)
    x3_new = new_center_point[0][0] - distancewidth/2 * np.cos(old_angle) - distanceheight/2 * np.sin(old_angle)
    y3_new = new_center_point[0][1] + distancewidth/2 * np.sin(old_angle) - distanceheight/2 * np.cos(old_angle)
    corners = np.array([[x0_new,y0_new],[x1_new,y1_new],[x2_new,y2_new],[x3_new,y3_new]])
    return corners # Output: corners of analysis image
# Objective: Warp perspective of desired image
def warp_perspective_new(img,recorners ,output_size=(160, 104)): # Input: Full raw image , reordered cornerd of desired image
    dst = np.array([
        [output_size[0] - 1, 0],
        [0, 0],
        [0, output_size[1] - 1],
        [output_size[0] - 1, output_size[1] - 1]
    ], dtype='float32') # Define warp perspective parameter 
    M = cv2.getPerspectiveTransform(np.array(recorners, dtype='float32'), dst)# Define warp perspective parameter 
    warped = cv2.warpPerspective(img, M, output_size) # Warped 
    return warped # Warp of desired image
# Objective: Get base value for further adjust
def lightness_adjust(image, image_name): # Input: Depend on which part or image use for base-adjustment, in this case use top 500 row from full image
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # Read image in LAB color space
    L_channel = lab_image[:,:,0] # Choose only L value
    top_500_rows = L_channel[:500,: ] # Choose L value top 500 row
    avg_L_values = top_500_rows.mean(axis=0).flatten() # Average to get result in horizontral
    pixel_columns = [f"pixel {i+1}" for i in range(len(avg_L_values))] # Create column name
    data = pd.DataFrame({"Image_Name": [image_name], **{pixel_columns[i]: [avg_L_values[i]] for i in range(len(avg_L_values))}}) 
    # Create dataframe which each row represent each image which their name and each column with l value
    return data # Average L value from top 500 row for full image
# Objective: Get average L value in horizontral
def Average_lightness(image, image_name): # Input: Image of solution spectral band after warperspctive
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # Read image in LAB color space
    L_channel = lab_image[:,:,0] # Choose only L value
    avg_L_values = L_channel.mean(axis = 0).flatten() # Average to get result in horizontral
    pixel_columns = [f"pixel {i+1}" for i in range(len(avg_L_values))] # Create column name
    data = pd.DataFrame({"Image_Name": [image_name], **{pixel_columns[i]: [avg_L_values[i]] for i in range(len(avg_L_values))}}) 
    # Create dataframe which each row represent each image which their name and each column with l value
    return data # Average L in horizontral value from solution spectral band after warperspctive
# Objective: Get average hue value
def Read_Image(img_array): # Input: image
    image_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # Read image in RGB
    height, width, _ = image_rgb.shape # Get the size of the image
    num_rows = 50 # Define the number of rows to read (in this case, 50)
    left_column_hues = [] # Read the left column of pixels and convert to HSV
    for row in range(min(num_rows, height)):
        rgb_pixel = image_rgb[row, 0, :] # Get RGB values
        hsv_pixel = cv2.cvtColor(np.uint8([[rgb_pixel]]), cv2.COLOR_RGB2HSV)[0, 0] # Convert RGB to HSV
        left_column_hues.append(hsv_pixel[0]) # Append the hue value to the list
    average_hue = sum(left_column_hues) / len(left_column_hues) # Calculate the average hue value
    #print("Average Hue Value:", average_hue) # Display the average hue
    return average_hue
# Objective: To find minimum value for shift process
def local_minimum_find(img, img_name, Local_minimum_each_image_Df, min_column_number_df): 
    # Input: Image, image name, Blank dataframe to stack minimum value and column header which has minimum value of all image
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # Read image in LAB color space
    L_channel = lab_image[:, :, 0] # Choose only L value
    median_L_values_horizontal = np.median(L_channel, axis=0).flatten() # Get median value in horizontral
    pixel_columns_horizontal_df = pd.DataFrame({'median_L_values_horizontal': median_L_values_horizontal}) # Collect median L value with column name
    pixel_columns_horizontal_df_for_find_local_minimum = pixel_columns_horizontal_df.iloc[85:100] # CHoose only column 85 to 100
    min_column_number = pixel_columns_horizontal_df_for_find_local_minimum['median_L_values_horizontal'].idxmin() # Get minimum value with in those threshold
    min_column_numbers_df = pd.DataFrame({'min_column_number_+1_if_Pixel': [min_column_number]}) # Create dataframe to store minimum value
    min_column_number_df = pd.concat([min_column_number_df, min_column_numbers_df]) # Stack data every time its loop
    min_column_number_df = min_column_number_df.reset_index() # Reset index
    min_column_number_df = min_column_number_df.drop(columns = "index").astype(int) #3 Change type of value to int
    Local_minimum_each_image_Df = pd.concat([Local_minimum_each_image_Df, pd.DataFrame({'filename': [img_name], 'min_column_number_+1_if_Pixel': [min_column_number]})]) 
    # Stack data with their image name and minimum value
    Local_minimum_each_image_Df = Local_minimum_each_image_Df.reset_index() # Reset index
    Local_minimum_each_image_Df = Local_minimum_each_image_Df.drop(columns = "index") # Drop column name index
    median_minimum = np.median(min_column_number_df, axis=0).flatten() # Get median value from each minimum value
    median_minimum_scalar = median_minimum.item() # Change type to int
    median_minimum = int(median_minimum_scalar)
    return Local_minimum_each_image_Df, min_column_number_df,median_minimum 
    # Output is mini L value for each image in that interval, cloumn which have the minimum L value and median menimum from every image
# Objective: Shift image base on minimum value
def Shift_image_median_minimum_add_condition(warp_img,median_minimum): # Input: warpercpective image and median minimum value
    average_hue = Read_Image(warp_img) # use Read_Image function to get average hue value
    initial_crop_box = (40, 20, 140, 330) 
    # Indentifine crop box which is (X position of topleft, y position of top left, x position of bottom right, y position of bottom right)
    lab_image = cv2.cvtColor(warp_img, cv2.COLOR_BGR2LAB) # Read image in LAB color space
    L_channel = lab_image[:, :, 0] # Choose only L value
    median_L_values_horizontal = np.median(L_channel, axis=0).flatten() # Get median value in horizontral
    pixel_columns_horizontal_df = pd.DataFrame({'median_L_values_horizontal': median_L_values_horizontal}) # Collect median L value with column name
    pixel_columns_horizontal_df_for_find_local_minimum = pixel_columns_horizontal_df.iloc[85:100] # Choose only column 85 to 100
    min_column_number = pixel_columns_horizontal_df_for_find_local_minimum['median_L_values_horizontal'].idxmin() # Get minimum value with in those threshold
    local_diff_med = int(median_minimum) - min_column_number # Get difference between median minimum from all image and from each image
    x_shift = local_diff_med # Store difference value in name x_shift
    # print(x_shift) # Print
    if  average_hue < 205: # Create condition that if average hue lower than 205
        warp_img = cv2.rotate(warp_img, cv2.ROTATE_180) # Rotate warped image
        if x_shift > 0: 
            shifted_crop_box = (initial_crop_box[0] + x_shift, initial_crop_box[1], initial_crop_box[2]+ x_shift , initial_crop_box[3])
            warp_img_crop_add = warp_img[int(shifted_crop_box[1]):int(shifted_crop_box[3]), int(shifted_crop_box[0]):int(shifted_crop_box[2])]
            # print(shifted_crop_box)
        else: # Operate image if x_shift is negative
            shifted_crop_box = (initial_crop_box[0] + x_shift, initial_crop_box[1], initial_crop_box[2] + x_shift , initial_crop_box[3])
            warp_img_crop_add = warp_img[int(shifted_crop_box[1]):int(shifted_crop_box[3]), int(shifted_crop_box[0]):int(shifted_crop_box[2])]
            # print(shifted_crop_box)
    return warp_img_crop_add,x_shift #  Output: Warp perspective image with warp add condition and x_shift from each image
# Objective: Warp perspective with condition but not shift 
def condition(img_array): #input: warp desired image
    average_hue = Read_Image(img_array) # sent image to read with Read_image
    crop_box = (6,20, 200, 330) # Desired Crop_box with x of top left, y of bottom left, x of top right and y of bottom right
    if  average_hue < 205: #Ddfine condition before Operate picture
        img_array = cv2.rotate(img_array, cv2.ROTATE_180) # Rotate
        img_array_Crop = img_array[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] # Crop image if not approciate shape
    else:
        # Save the image to the output folder without rotation
        img_array_Crop = img_array[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] # Crop image if not approciate shape
    return img_array_Crop # Output: Warp add image
# Objective: Adjust L value form Air image 
def adjust_pixel_values(df):
    columns_to_drop = ['Concentration_1_ppm', 'Concentration_2_ppm', 'Solute_1', 'Solute_2', 'Solvent', 'Light_Source', 'Picture_Number'] # Chosse column to drop
    dropped_columns = df[columns_to_drop] # Store choosed column
    df = df.drop(columns=columns_to_drop) # Drop choosed column
    pixel_adjust = "pixel 25" # specific pixel to analyze
    pixel_1_avg = df[pixel_adjust].mean()# Calculate the average of the "pixel 1" column
    adjusted_df = pd.DataFrame()    # Create a new DataFrame to store the adjusted values
    for _, row in df.iterrows():    # Iterate over each row in the original DataFrame
        adjustment = row[pixel_adjust] - pixel_1_avg       # Calculate the adjustment value for the current row
        adjusted_row = row.to_frame().transpose()# Create a new row with adjusted values
        adjusted_row[pixel_adjust] -= adjustment
        for col in adjusted_row.columns: # Apply the same adjustment to all other columns with loop
            if col != pixel_adjust:
                adjusted_row[col] -= adjustment
        adjusted_df = pd.concat([adjusted_df, adjusted_row], ignore_index=True) # Append the adjusted row to the new DataFrame
    dropped_columns.reset_index(drop=True, inplace=True) # Reset indices of both DataFrames before concatenating
    adjusted_df.reset_index(drop=True, inplace=True) # Reset index
    adjusted_df = pd.concat([dropped_columns, adjusted_df], axis=1) # Concatenate the DataFrames along the columns axis
    return adjusted_df # Output: adjusted dataframe
# Objective: Create dictionary that seperate with each experiment base on solute name
def create_substance_dfs(data): # Input: dataframe after substact blank
    substance_dfs = {} # create blank dictionary
    unique_substances = data[['Solute_1', 'Solute_2']].drop_duplicates().values # drop all dupricate data leave only unique one
    for substance in unique_substances: # loop through unique value
        Solute_1, Solute_2 = substance # get data form substance in to solute_1 and solute_2
        if pd.isna(Solute_2): # create condition that if solute_2 is Null it will store as 'None'
            Solute_2 = 'None'
        key = f"{Solute_1}_{Solute_2.replace(' ', '')}" # Create key in name of each solute
        substance_dfs[key] = data[(data['Solute_1'] == Solute_1) & (data['Solute_2'] == Solute_2)] # store in dictionary
    return substance_dfs # Output: dictionary after cleaning
# Objective:Separate the data into pure and mixture components, and perform machine learning on each component.
def pure_mixture_seperation(all_dataframe): # Input: dictionary fromcreate_substance_dfs
    # Convert dictionary keys to a list
    dict_keys = list(all_dataframe.keys()) # Convert the keys of the input dictionary 'all_dataframe' to a list
    # Create empty dictionaries to store the results
    dict_combined = {} # Create an empty dictionary to store the combined dictionary results
    dict_mix = {} # Create an empty dictionary to store the mixture dictionary results
    metric_combined = {} # Create an empty dictionary to store the combined metric results
    metric_mix = {} # Create an empty dictionary to store the mixture metric results
    model_pure = {} # Create an empty dictionary to store the pure component models
    model_mix = {} # Create an empty dictionary to store the mixture component models
    model_combine = {} # Create an empty dictionary to store the combined models
    for key in dict_keys: # Iterate over the keys in the list 'dict_keys'
        # Split the key by '_'
        key_parts = key.split('_') # Split the current key by the underscore character
        # Check if the second part (index 1) is 'None'
        if key_parts[1] == 'None': # Check if the second part of the key (index 1) is 'None'
            # This is a pure component
            DATA = all_dataframe[key] # Retrieve the data for the current key from 'all_dataframe'
            columns_to_drop = ['Concentration_1_ppm', 'Concentration_2_ppm', 'Solute_1', 'Solute_2', 'Solvent', 'Light_Source', 'Picture_Number'] 
            # Define a list of columns to be dropped from the data
            X = DATA.drop(columns=columns_to_drop).values # Create the feature matrix 'X' by dropping the specified columns
            y = np.array(DATA['Concentration_1_ppm'].values).astype(float) # Create the target vector 'y' from the 'Concentration_1_ppm' column
            best_model, dict_df, metrics_df = machine_learning(X, y, key) # Call the 'machine_learning' function and store the results
            # Store the results in the respective dictionaries
            dict_combined[key] = dict_df # Store the dictionary result for the pure component in 'dict_combined'
            metric_combined[key] = metrics_df # Store the metric result for the pure component in 'metric_combined'
            model_pure[key] = best_model # Store the best model for the pure component in 'model_pure'
        else: # If the condition in the previous 'if' statement is false, execute this block
            # This is a mixture component
            DATA = all_dataframe[key] # Retrieve the data for the current key from 'all_dataframe'
            columns_to_drop = ['Concentration_1_ppm', 'Concentration_2_ppm', 'Solute_1', 'Solute_2', 'Solvent', 'Light_Source', 'Picture_Number'] # Define a list of columns to be dropped from the data
            X_1 = DATA.drop(columns=columns_to_drop).values # Create the feature matrix 'X_1' by dropping the specified columns
            y_1 = np.array(DATA['Concentration_1_ppm'].values).astype(float) # Create the target vector 'y_1' from the 'Concentration_1_ppm' column
            best_model_1, dict_df_1, metrics_df_1 = machine_learning(X_1, y_1, key) # Call the 'machine_learning' function and store the results for the first component
            X_2 = DATA.drop(columns=columns_to_drop).values # Create the feature matrix 'X_2' by dropping the specified columns
            y_2 = np.array(DATA['Concentration_2_ppm'].values).astype(float) # Create the target vector 'y_2' from the 'Concentration_2_ppm' column
            best_model_2, dict_df_2, metrics_df_2 = machine_learning(X_2, y_2, key) # Call the 'machine_learning' function and store the results for the second component
            # Store the results in the respective dictionaries
            dict_mix = {f"{key_parts[0]}_{key_parts[1]}": {key_parts[0]: dict_df_1, key_parts[1]: dict_df_2}} # Store the dictionary results for the mixture components in 'dict_mix'
            metric_mix = {f"{key_parts[0]}_{key_parts[1]}": {key_parts[0]: metrics_df_1, key_parts[1]: metrics_df_2}} # Store the metric results for the mixture components in 'metric_mix'
            model_mix = {f"{key_parts[0]}_{key_parts[1]}": {key_parts[0]: best_model_1, key_parts[1]: best_model_2}} # Store the best models for the mixture components in 'model_mix'
            model_combine[key] = model_mix # Store the combined model for the current key in 'model_combine'
            # Combine the dictionaries
            metric_combined = {**metric_combined, **metric_mix} # Combine the 'metric_combined' and 'metric_mix' dictionaries
            dict_combined = {**dict_combined, **dict_mix} # Combine the 'dict_combined' and 'dict_mix' dictionaries
    return model_pure, model_combine, dict_combined, metric_combined # Return the pure component models, combined models, combined dictionaries, and combined metrics
# Objective: Predict and get evaluation metric
def machine_learning(X, y, key):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11) # Split the data into train and test sets
    prediction_data_train_df = pd.DataFrame(y_train, columns=['Actual']) # Create a DataFrame for actual train values
    prediction_data_test_df = pd.DataFrame(y_test, columns=['Actual']) # Create a DataFrame for actual test values
    models = {
        'SVR': SVR(),
        'MLR': LinearRegression(),
        'RF' : RandomForestRegressor(),
        'MLP': MLPRegressor(max_iter=100000),
        'XGB': XGBRegressor()
    } 
    # Define the models to be trained
    param_grids = {
        'SVR': {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'rbf', 'sigmoid']
        },
        'MLR': {
           'fit_intercept': [True, False],
        },
        'RF': {
            'n_estimators': [50, 100, 200, 300, 400, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 15, 20]
        },
        'MLP': {
            'hidden_layer_sizes': [(50, 50), (75, 75), (100,100), (25,25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01]
        },
        'XGB': {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1]
        }
    } 
    # Define the parameter grids for each model
    selector = SelectKBest(score_func=f_regression, k=4 ) # Select the best 4 features using the f_regression score function
    kf = KFold(n_splits=10, shuffle=True, random_state=42) # Define a 10-fold cross-validation object
    loo = LeaveOneOut() # Define a leave-one-out cross-validation object
    results = {} # Initialize an empty dictionary to store results
    best_scores = {}  # Dictionary to keep track of the best score for each model
    best_y_trains = {}  # Dictionary to keep track of the y_train for the best fold for each model
    best_params = {} # Dictionary to store the best parameters for each model
    # Initialize lists to store evaluation metrics for each model
    mape_train_list = []
    mape_test_list = []
    mae_train_list = []
    mae_test_list = []
    r2_train_list = []
    r2_test_list = []
    mse_train_list = []
    mse_test_list = []
    model_names = []
        
    for model_name, model in models.items(): # Iterate over the models
        print(f"Training {model_name} - {key}...") # Print the model name and key
        best_score = float('inf') # Initialize the best score to infinity
        X_train_selected = selector.fit_transform(X_train, y_train) # Select the best features from the training data
        feature_scores = selector.scores_ # Get the feature scores
        feature_scores_tuples = [(i, feature_scores[i]) for i in selector.get_support(indices=True)] # Get the feature indices and scores
        sorted_features = sorted(feature_scores_tuples, key=lambda x: x[1], reverse=True) # Sort the features by score in descending order
        #random_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name],
        #scoring='r2', cv=kf) # Define a grid search object with leave-one-out cross-validation
        random_search = RandomizedSearchCV(estimator=model,param_distributions=param_grids[model_name],n_iter=5,
         scoring='r2',cv=kf,random_state=42)
        random_search.fit(X_train_selected, y_train) # Fit the grid search object to the training data
        best_model = random_search.best_estimator_ # Get the best model from the grid search
        best_params[model_name] = random_search.best_params_ # Store the best parameters for the model

        save_mape = os.path.join("newmodel", "Formalin")
        os.makedirs(save_mape, exist_ok=True)
        if not os.path.exists(save_mape):
            os.makedirs(save_mape)
        else:
            print(f"{save_mape}")
        
        model_filename = os.path.join(save_mape, f"{model_name}_model.pkl")
        with open(model_filename, "wb") as f: # Open a file to store the best model
            pickle.dump(best_model, f) # Save the best model to the file

        X_train_selected = selector.transform(X_train) # Transform the training data using the selected features
        predictions_train = best_model.predict(X_train_selected) # Make predictions on the training data
        predictions_train_df = pd.DataFrame(predictions_train, columns=[f"{model_name}_Train"]) # Create a DataFrame for train predictions
        prediction_data_train_df = pd.concat([prediction_data_train_df, predictions_train_df], axis=1) # Concatenate train predictions and actual values
        
        r2_train = r2_score(y_train, predictions_train) # Calculate R^2 score for train predictions
        mse_train = mean_squared_error(y_train, predictions_train) # Calculate MSE for train predictions
        mape_train = mean_absolute_percentage_error(y_train, predictions_train)*100 # Calculate MAPE for train predictions
        mae_train = np.mean(np.abs(y_train - predictions_train)) # Calculate MAE for train predictions
        
        X_test_selected = selector.transform(X_test) # Transform the test data using the selected features
        predictions_test = best_model.predict(X_test_selected) # Make predictions on the test data
        predictions_test_df = pd.DataFrame(predictions_test, columns=[f"{model_name}_Test"]) # Create a DataFrame for test predictions
        prediction_data_test_df = pd.concat([prediction_data_test_df, predictions_test_df], axis=1) # Concatenate test predictions and actual values
        
        r2_test = r2_score(y_test, predictions_test) # Calculate R^2 score for test predictions
        mse_test = mean_squared_error(y_test, predictions_test) # Calculate MSE for test predictions
        mape_test = mean_absolute_percentage_error(y_test, predictions_test)*100 # Calculate MAPE for test predictions
        mae_test = np.mean(np.abs(y_test - predictions_test)) # Calculate MAE for test predictions
        
        # Append evaluation metrics to respective lists
        mape_train_list.append(mape_train)
        mape_test_list.append(mape_test)
        mae_train_list.append(mae_train)
        mae_test_list.append(mae_test)
        r2_train_list.append(r2_train)
        r2_test_list.append(r2_test)
        mse_train_list.append(mse_train)
        mse_test_list.append(mse_test)
        model_names.append(model_name)
        
        print(f"{model_name} Train      - R^2: {r2_train:.2f}, MAPE: {mape_train:.2f}%, MSE: {mse_train:.2f}, MAE: {mae_train:.2f}") # Print train evaluation metrics
        print(f"{model_name} Test       - R^2: {r2_test:.2f}, MAPE: {mape_test:.2f}%, MSE: {mse_test:.2f}, MAE: {mae_test:.2f}") # Print test evaluation metrics
        print(f"Best parameters: {best_params}") # Print the best parameters for the model

        mape_filename = os.path.join(save_mape, f"{model_name}_MAPE.pkl")
        with open(mape_filename, "wb") as f:
            pickle.dump(mape_test, f)
        
        # Print selected features
        print("SelectedKBest feature indices and scores (sorted from high to low):") 
        for feature_idx, score in sorted_features: # Iterate over the sorted features
            print(f"Feature Index: {feature_idx}, Score: {score}") # Print the feature index and score
            selector_filename = os.path.join(save_mape, 'selector.pkl')
            with open(selector_filename, "wb") as f:
                pickle.dump(selector, f)
    
        print("-" * 50)
        Dict_with_Df = {"train": prediction_data_train_df, "test": prediction_data_test_df} # Create a dictionary with train and test prediction DataFrames
        # Create a DataFrame from the lists
        metrics_df = pd.DataFrame({
            'Model': model_names,
            'MAPE Train': mape_train_list,
            'MAPE Test': mape_test_list,
            'MAE Train': mae_train_list,
            'MAE Test': mae_test_list,
            #'R2 Train': r2_train_list,
            #'R2 Test': r2_test_list,
            'MSE Train': mse_train_list,
            'MSE Test': mse_test_list
        }) # Create a DataFrame with evaluation metrics for all models
        
    return best_model, Dict_with_Df, metrics_df # Return the best model, prediction DataFrames, and metrics DataFrame
# Objective: Plot between actual and predict data
def plot_data(data_dict): # Input: Y predict dictionary
    plt.rc('font', size=25) # Set the default font size for text to 18
    plt.rc('axes', titlesize=25) # Set the font size for plot titles to 20
    plt.rc('axes', labelsize=25) # Set the font size for axis labels to 18
    dict_keys = list(data_dict.keys()) # Convert the keys of the input dictionary 'data_dict' to a list
    for key in dict_keys: # Iterate over the keys in the list 'dict_keys'
        key_parts = key.split('_') # Split the current key by the underscore character
        solution = key_parts[0] # The first part of the key represents the solution
        sol_data = data_dict[key] # Retrieve the data for the current key from 'data_dict'
        if key_parts[1] == 'None': # Check if the second part of the key is 'None'
            train, test = sol_data.items()
            train_data_type, train_df = train
            test_data_type, test_df = test
            for i in range(1, train_df.shape[1]): # Iterate over columns starting from the second column
                plt.figure(figsize=(15, 15)) # Create a new figure with a size of 15x15 inches
                plt.plot(train_df.iloc[:, 0], train_df.iloc[:, i], 'bo', label=f'{train_data_type}') 
                # Plot the actual values (column 0) against the predictions (column i)
                plt.plot(test_df.iloc[:, 0], test_df.iloc[:, i], 'ro', label=f'{test_data_type}')
                model_name = train_df.columns[i].split('_')[0] # Extract the model name from the column header
                set_name = train_df.columns[i].split('_')[1] # Extract the set name (Train or Test) from the column header
                if solution == 'Carbosulfan':
                    plt.xlim(0, 1600)
                    plt.ylim(0, 1600)
                    plt.plot([0, 1600], [0, 1600], 'k--')
                    plt.xlabel('Actual concentration (ppm)') # Set the label for the x-axis
                    plt.ylabel("Prediction concentration (ppm)") # Set the label for the y-axis
                    plt.title(f"Prediction Value vs. Actual Value ({model_name}) - {solution}") # Set the plot title
                else:
                    plt.xlim(0, 10)
                    plt.ylim(0, 10)
                    plt.plot([0, 10], [0, 10], 'k--')
                    plt.xlabel('Actual concentration (%w/v)') # Set the label for the x-axis
                    plt.ylabel("Prediction concentration (ppm)") # Set the label for the y-axis
                    plt.title(f"Prediction Value vs. Actual Value ({model_name}) - {solution}") # Set the plot title
                plt.grid(True) # Show the grid lines
                plt.legend()
                plt.show() # Display the plot
        else:
            sol_data_keys = list(sol_data.keys())
            # Iterate over pairs of keys
            for i in range(len(sol_data_keys)-1):
                for j in range(i+1, len(sol_data_keys)):
                    solution_mix_1 = sol_data_keys[i]
                    solution_mix_2 = sol_data_keys[j]
                    # Get data for each solution mix
                    solution_data_1 = sol_data[solution_mix_1]
                    solution_data_2 = sol_data[solution_mix_2]
                    # Iterate over data types
                    for data_type, df_1 in solution_data_1.items():
                        df_2 = solution_data_2[data_type]
                        # Iterate over columns
                        for col_index in range(1, df_1.shape[1]):
                            plt.figure(figsize=(15, 15))
                            plt.plot(df_1.iloc[:, 0], df_1.iloc[:, col_index], 'bo', label=f'{solution_mix_1}')
                            plt.plot(df_2.iloc[:, 0], df_2.iloc[:, col_index], 'ro', label=f'{solution_mix_2}')
                            plt.xlim(0, 13)
                            plt.ylim(0, 13)
                            plt.plot([0, 13], [0, 13], 'k--')
                            
                            model_name = df_1.columns[col_index].split('_')[0]
                            set_name = df_1.columns[col_index].split('_')[1]
                            
                            plt.xlabel('Actual concentration (%w/v)')
                            plt.ylabel(f"{set_name} set prediction concentration (%w/v)")
                            plt.title(f"{set_name} Data vs. Actual Value ({model_name}) - {solution_mix_1} - {solution_mix_2} mixture ({data_type})")
                            plt.grid(True)
                            plt.legend()
                            plt.show()
# Objective: Plot trend of raw L data             
def plot_trend(df): # Input: dataframe that not substract and still have blank image data
    plt.rc('font', size=25) # Set the default font size for text to 18
    plt.rc('axes', titlesize=25) # Set the font size for plot titles to 20
    plt.rc('axes', labelsize=25) # Set the font size for axis labels to 18
    # Drop the "Light_Source" and "Picture_Number" columns from the input DataFrame
    plot_data = df.drop(columns=["Light_Source", "Picture_Number"]) 
    # Group the data by 'Solute_1', 'Concentration_1_ppm', 'Solute_2', 'Concentration_2_ppm', and 'Solvent'
    grouped_data_plot = plot_data.groupby(['Solute_1', 'Concentration_1_ppm', 'Solute_2', 'Concentration_2_ppm', 'Solvent']) 
    plot_data_group = grouped_data_plot.mean().reset_index() # Calculate the mean of the grouped data and reset the index
    # Get unique combinations of 'Solute_1' and 'Solute_2' columns
    unique_solute_combinations = plot_data_group[['Solute_1', 'Solute_2']].drop_duplicates().values # Get unique combinations of 'Solute_1' and 'Solute_2'
    for solute_1, solute_2 in unique_solute_combinations: # Iterate over the unique solute combinations
        # Filter data for the current solute combination
        # Filter the data for the current solute combination
        solute_combination_data = plot_data_group[(plot_data_group['Solute_1'] == solute_1) & (plot_data_group['Solute_2'] == solute_2)] 
        plt.figure(figsize=(20, 15)) # Create a new figure with a size of 20x15 inches
        pixels = range(1, 101) # Define a range of pixel values from 1 to 100
        # Plot each concentration's L values against pixel for the current solute combination
        for _, row in solute_combination_data.iterrows(): # Iterate over rows in the solute combination data 
            concentration_1 = row['Concentration_1_ppm'] # Get the concentration value for solute 1
            concentration_2 = row['Concentration_2_ppm'] # Get the concentration value for solute 2
            solvent = row['Solvent'] # Get the solvent value
            # Get the column names for L values
            columns = [col for col in solute_combination_data.columns if col not in ['Solute_1', 'Concentration_1_ppm', 'Solute_2', 'Concentration_2_ppm', 'Solvent']] 
            l_values = row[columns] # Get the L values for the current row
            label = f"{solute_1}_{concentration_1}_ppm_{solute_2}_{concentration_2}_ppm_{solvent}" # Create a label for the plot
            plt.plot(pixels, l_values, label=label) # Plot the L values against pixels with the corresponding label
        # Plot the 'Blank' series for the same solvent
        # Filter the data for 'Blank' solute and the current solvent
        blank_data = plot_data_group[(plot_data_group['Solute_1'] == 'Blank') & (plot_data_group['Solvent'] == solvent)] 
        if not blank_data.empty: # Check if the blank data is not empty
            for _, row in blank_data.iterrows(): # Iterate over rows in the blank data
                concentration_1 = row['Concentration_1_ppm'] # Get the concentration value for solute 1 (should be 0 for 'Blank')
                concentration_2 = row['Concentration_2_ppm'] # Get the concentration value for solute 2 (should be 0 for 'Blank')
                # Get the column names for L values
                columns = [col for col in blank_data.columns if col not in ['Solute_1', 'Concentration_1_ppm', 'Solute_2', 'Concentration_2_ppm', 'Solvent']] 
                l_values = row[columns] # Get the L values for the current row
                label = f"Blank_{concentration_1}_ppm_{concentration_2}_ppm_{solvent}" # Create a label for the 'Blank' plot
                plt.plot(pixels, l_values, label=label) # Plot the 'Blank' L values against pixels with the corresponding label
        plt.xlabel('Pixel') # Set the label for the x-axis
        plt.ylabel('L Value') # Set the label for the y-axis
        plt.title(f'L Value vs Pixel for Different Concentrations (Solute_1: {solute_1}, Solute_2: {solute_2})') # Set the plot title
        plt.legend() # Show the legend
        plt.show() # Display the plot
# Objective: Plot trend of raw L air data     
def plot_trend_air(df): # Input: dataframe that not substract and still have blank image data
    plt.rc('font', size=30) # Set the default font size for text to 18
    plt.rc('axes', titlesize=40) # Set the font size for plot titles to 20
    plt.rc('axes', labelsize=35) # Set the font size for axis labels to 18
    pixels = range(1, 155)
    plt.figure(figsize=(40, 20))
    for index, row in df.iterrows():
        plt.plot(pixels, row.values)
    plt.xlabel('Pixel') # Set the label for the x-axis
    plt.ylabel('L Value') # Set the label for the y-axis
    plt.title('L Value vs Pixel for Different Concentrations') # Set the plot title
    plt.xlim(0, 154)
    plt.ylim(0, 154)
    plt.show() # Display the plot
# Objective: Plot data trend after substract
def plot_trend_diff_L(df):
    plt.rc('font', size=18) # Set the default font size for text to 18
    plt.rc('axes', titlesize=20) # Set the font size for plot titles to 20
    plt.rc('axes', labelsize=18) # Set the font size for axis labels to 18
    plot_data = df.drop(columns=["Light_Source", "Picture_Number"]) # Drop the "Light_Source" and "Picture_Number" columns from the input DataFrame
    # Group the data by 'Solute_1', 'Concentration_1_ppm', 'Solute_2', 'Concentration_2_ppm', and 'Solvent'
    grouped_data_plot = plot_data.groupby(['Solute_1', 'Concentration_1_ppm', 'Solute_2', 'Concentration_2_ppm', 'Solvent']) 
    plot_data_group = grouped_data_plot.mean().reset_index() # Calculate the mean of the grouped data and reset the index
    # Get unique combinations of 'Solute_1' and 'Solute_2' columns
    unique_solute_combinations = plot_data_group[['Solute_1', 'Solute_2']].drop_duplicates().values # Get unique combinations of 'Solute_1' and 'Solute_2'
    for solute_1, solute_2 in unique_solute_combinations: # Iterate over the unique solute combinations
        # Filter data for the current solute combination
        # Filter the data for the current solute combination
        solute_combination_data = plot_data_group[(plot_data_group['Solute_1'] == solute_1) & (plot_data_group['Solute_2'] == solute_2)] 
        plt.figure(figsize=(20, 15)) # Create a new figure with a size of 20x15 inches
        pixels = range(1, 101) # Define a range of pixel values from 1 to 100
        # Plot each concentration's L values against pixel for the current solute combination
        for _, row in solute_combination_data.iterrows(): # Iterate over rows in the solute combination data
            concentration_1 = row['Concentration_1_ppm'] # Get the concentration value for solute 1
            concentration_2 = row['Concentration_2_ppm'] # Get the concentration value for solute 2
            solvent = row['Solvent'] # Get the solvent value
            # Get the column names for L values
            columns = [col for col in solute_combination_data.columns if col not in ['Solute_1', 'Concentration_1_ppm', 'Solute_2', 'Concentration_2_ppm', 'Solvent']] 
            l_values = row[columns] # Get the L values for the current row
            label = f"{solute_1}_{concentration_1}_ppm_{solute_2}_{concentration_2}_ppm_{solvent}" # Create a label for the plot
            plt.plot(pixels, l_values, label=label) # Plot the L values against pixels with the corresponding label
        plt.xlabel('Pixel') # Set the label for the x-axis
        plt.ylabel('L Value') # Set the label for the y-axis
        plt.title(f'L Value vs Pixel for Different Concentrations (Solute_1: {solute_1}, Solute_2: {solute_2})') # Set the plot title
        plt.xlim(0, 100)
        plt.ylim(-36, 90)
        plt.legend() # Show the legend
        plt.show() # Display the plot

#Referent
Local_minimum_each_image_Df = pd.DataFrame()  # Initialize an empty DataFrame to store local minimum data
min_column_number_df = pd.DataFrame()  # Initialize an empty DataFrame to store column numbers
Image_name_data_df_loop = []  # Initialize an empty list to store image name data DataFrames
Air_data_df = pd.DataFrame()
result_data = pd.DataFrame()  # Initialize an empty DataFrame to store results
result_datafull_Blank = pd.DataFrame()  # Initialize an empty DataFrame
result_anal_datafull = pd.DataFrame()  # Initialize an empty DataFrame

for filename in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, filename)
    img = cv2.imread(img_path)
    try: # Try to execute the following code
        corners = find_corners(img) # Find the corners of the image using the find_corners function
        distances = Linedistance(corners) # Calculate the distances between corners using the Linedistance function
        reordered_corners = corner_reorder(corners) # Reorder the corners using the corner_reorder function, passing corners as an argument
        fiveparam = FiveParam(reordered_corners) # Compute the five parameters (center, width, height, angle, and aspect ratio) using the FiveParam function
        center = fiveparam[0] # Extract the center from the FiveParam result
        angle = fiveparam[3] # Extract the angle from the FiveParam result
        Air_crop_img, Air_crop_box = Crop_by_avg(img) # Crop the image using the Crop_by_avg function
        marked_crop_img = Air_crop_img.copy() # Create a copy of the cropped image for marking
        cv2.drawContours(marked_crop_img, [reordered_corners], 0, (0, 255, 0), thickness=2) # Draw the reordered corners on the marked cropped image
        output_path_marked1 = os.path.join(OUTPUT_Crop_MARKED, f"marked_{filename}") # Construct the output path for the marked cropped image
        cv2.imwrite(output_path_marked1, Air_crop_img) # Save the cropped image as "marked_{filename}" in the OUTPUT_Crop_MARKED directory
        WARP_crop_img = Air_crop_img.copy() # Create a copy of the cropped image for warping
        warped_Air_image = warp_perspective_new(WARP_crop_img,reordered_corners) # Warp the cropped image using the warp_perspective_new function
        output_path_warp = os.path.join(OUTPUT_Crop_WARP, f"warp_{filename}") # Construct the output path for the warped image
        #cv2.imwrite(output_path_warp, warped_Air_image) # Commented out code for saving the warped image
        crop_box = np.array([Air_crop_box[0],Air_crop_box[1]]) # Create a numpy array from the crop box coordinates
        new_reordered_corners = crop_box + reordered_corners # Add the crop box coordinates to the reordered corners
        new_center = crop_box + center # Add the crop box coordinates to the center
        center_x, center_y = int(new_center[0, 0]), int(new_center[0, 1]) # Extract the x and y coordinates of the new center
        img_real_for_center = img.copy() # Create a copy of the original image for marking the center
        cv2.drawContours(img_real_for_center, [new_reordered_corners], 0, (0, 255, 0), 2) # Draw the new reordered corners on the marked image
        cv2.circle(img_real_for_center, (center_x, center_y), radius=1, color=(0, 255, 0), thickness=3) # Draw a circle at the new center on the marked image
        output_path_marked = os.path.join(OUTPUT_REAL_MARK, f"marked_{filename}") # Construct the output path for the marked image
        cv2.imwrite(output_path_marked, img_real_for_center) # Save the marked image with the new center and reordered corners
        Warp_add_Air_image = condition(warped_Air_image) # Apply the condition function to the warped image
        
        output_path_warp_add = os.path.join(output_crop_Warp_add, f"warp_{filename}") # Construct the output path for the conditioned warped image
        cv2.imwrite(output_path_warp_add, Warp_add_Air_image) # Save the conditioned warped image
        Full_image = img.copy() # Create a copy of the original image
        image_name = os.path.splitext(filename)[0] # Extract the image name without extension
        parts = image_name.split('_') # Split the image name into parts using underscores
        Picture_number = parts[7] # Extract the picture number from the parts list
        Concentration_1_ppm = parts[4] # Extract the first concentration value from the parts list
        Concentration_2_ppm = parts[6] # Extract the second concentration value from the parts list
        solvent = parts[2] # Extract the solvent information from the parts list
        solute_1 = parts[3] # Extract the first solute information from the parts list
        solute_2 = parts[5] # Extract the second solute information from the parts list
        Light_Source = parts[1] # Extract the light source information from the parts list
        Image_name_data = {'Light_Source': [Light_Source], 'Solute_1': [solute_1],'Solute_2': [solute_2], 'Solvent': [solvent], 'Picture_Number': [Picture_number], 
                           'Concentration_1_ppm': [Concentration_1_ppm],'Concentration_2_ppm': [Concentration_2_ppm] } # Create a dictionary with image name data
        Image_name_data_df = pd.DataFrame(Image_name_data) # Convert the image name data dictionary to a DataFrame
        Image_name_data_df_loop.append(Image_name_data_df) # Append the image name data DataFrame to the loop list
        img_air_df = Average_lightness(Warp_add_Air_image, Image_name_data)
        Air_data_df = pd.concat([Air_data_df,img_air_df])
        if angle > 0: # If the angle is positive (counter-clockwise rotation)
            distance_y = -np.tan(angle*rotate_counter_clockwise)*distance_x + more_red # Calculate the y distance based on the angle and other parameters
            angle = angle + angle # Double the angle value
        else: # If the angle is negative (clockwise rotation)
            distance_y = np.tan(angle+rotate_clockwise)*distance_x - more_red # Calculate the y distance based on the angle and other parameters
        if angle == 0: # If the angle is zero (no rotation)
            distance_y = -np.tan(angle*rotate_counter_clockwise)*distance_x # Calculate the y distance based on the angle and other parameters
            andgle = angle + perfect_morph # Adjust the angle value
        distance_y_new = -6 # Set a new value for the y distance
        angle = angle *180/np.pi # Convert the angle from radians to degrees
        angle = angle + angle_adjustment # Apply an adjustment to the angle value
        angle = angle/(180/np.pi) # Convert the angle back from degrees to radians
        ref_distance = np.array([[distance_x,distance_y]]) # Create a numpy array with the x and y distances
        new_Anal_center = new_center + ref_distance # Calculate the new analysis center by adding the reference distance to the new center
        # Calculate the new analysis position using the CenterToFiveParamFormRef function
        new_Anal_Position = np.intp(CenterToFiveParamFormRef(new_Anal_center, new_reordered_corners, angle, height_solution_ratio, width_solution_ratio, more_red)) 
        img_Anal_real = img_real_for_center.copy() # Create a copy of the marked image for analysis
        cv2.drawContours(img_Anal_real, [new_Anal_Position], 0, (0, 255, 0), 2) # Draw the new analysis position on the marked image
        output_path_marked_anal = os.path.join(OUTPUT_Anal_real, f"marked_{filename}") # Construct the output path for the marked analysis image
        cv2.imwrite(output_path_marked_anal, img_Anal_real) # Save the marked analysis image
        anal_distances = Linedistance(new_Anal_Position) # Calculate the distances for the analysis position using the Linedistance function
        img_real_anal = img.copy() # Create a copy of the original image for analysis
        warped_image_anal = warp_perspective_new(img_real_anal, new_Anal_Position) # Warp the analysis image using the warp_perspective_new function
        output_path_warp_anal = os.path.join(OUTPUT_ANAL_WARP, f"warp_{filename}") # Construct the output path for the warped analysis image
        cv2.imwrite(output_path_warp_anal, warped_image_anal) # Save the warped analysis image
        Local_minimum_each_image_Df, min_column_number_df,median_minimum = local_minimum_find(warped_image_anal, image_name, Local_minimum_each_image_Df, 
                                                                        min_column_number_df) # Find the local minimum using the local_minimum_find function
        #image_data = lightness_adjust(Full_image, image_name)# Adjust the lightness of the full image using the lightness_adjust function
        image_data = lightness_adjust(Warp_add_Air_image, image_name)
        # Concatenate the image data to the result_data DataFrame
        result_data = pd.concat([result_data, image_data], ignore_index=True) 
        # Print a message indicating successful processing, marking, and warping
        # print(f"Processed, marked, mark real, and warped: {output_path_marked}, {OUTPUT_REAL_MARK} ,{output_path_warp}") 
    except ValueError as e: # If a ValueError occurs during processing
        print(f"Error processing {filename}: {e}") # Print an error message with the filename and error description

Image_name_data_df_full = pd.concat(Image_name_data_df_loop, ignore_index=True)  # Concatenate all DataFrames in Image_name_data_df_loop into a single DataFrame

image_name_df = result_data.iloc[:, [0]]  # Extract the first column (image names) from result_data
result_data_noname = result_data.drop(result_data.columns[0], axis=1)  # Drop the first column from result_data
result_data_noname_average = result_data_noname.mean(axis=1)  # Calculate the row-wise mean of result_data_noname
median_data = result_data_noname.mean(axis=0)  # Calculate the column-wise mean of result_data_noname
median_data = median_data.values  # Convert median_data to a numpy array
median_data = np.median(median_data,axis=0)  # Calculate the median across all columns

save_mape = os.path.join("newmodel", "Formalin")
os.makedirs(save_mape, exist_ok=True)
if not os.path.exists(save_mape):
    os.makedirs(save_mape)
else:
    print(f"{save_mape}")

median_data_path = os.path.join(save_mape, 'median_data.pkl')
with open(median_data_path, 'wb') as f:
    pickle.dump(median_data, f)

adjust_data = median_data - result_data_noname_average  # Assuming adjust_data is a pandas Series
adjust_data_array = adjust_data.values  # Convert adjust_data to a numpy array
reshaped_adjust_data = adjust_data_array[:, np.newaxis]  # Reshape adjust_data_array to a column vector

Air_data = Air_data_df.drop(columns = ["Image_Name"]).values
New_Air_data = Air_data + reshaped_adjust_data
New_Air_data_df = pd.DataFrame(New_Air_data)
New_Air_data_df = pd.concat([Image_name_data_df_full, New_Air_data_df], axis = 1) 

# Anal
for filename_warp_img in os.listdir(OUTPUT_ANAL_WARP):  # Iterate over filenames in the OUTPUT_ANAL_WARP directory
    img_path = os.path.join(OUTPUT_ANAL_WARP, filename_warp_img)  # Construct the full image path
    warp_img = cv2.imread(img_path)  # Read the warped image
    warped_image_anal_add,x_shift = Shift_image_median_minimum_add_condition(warp_img,median_minimum)  # Call the Shift_image_median_minimum_add_condition function
    image_namefull_anal = os.path.splitext(filename)[0]  # Extract the image name without extension
    image_datafull_anal = Average_lightness(warped_image_anal_add, image_namefull_anal)  # Call the Average_lightness function
    result_anal_datafull = pd.concat([result_anal_datafull, image_datafull_anal], ignore_index=True)  # Concatenate image_datafull_anal to result_anal_datafull
    warp_img_crop_add,x_shift = Shift_image_median_minimum_add_condition(warp_img,median_minimum)  # Call the Shift_image_median_minimum_add_condition function
    # print(filename_warp_img)  # Print the current filename
    output_path_warp_adjust = os.path.join(OUTPUT_ANAL_WARP_add, f"marked_{filename_warp_img}")  # Construct the output path for the marked image
    cv2.imwrite(output_path_warp_adjust, warp_img_crop_add)  # Save the marked image

result_data_full_noname_anal = result_anal_datafull.drop(result_data.columns[0], axis=1)  # Drop the first column from result_anal_datafull
result_data_full_noname_anal = result_data_full_noname_anal.values  # Convert result_data_full_noname_anal to a numpy array
new_result_data_anal = result_data_full_noname_anal + reshaped_adjust_data  # Add reshaped_adjust_data to result_data_full_noname_anal

new_result_data_anal_df = pd.DataFrame(new_result_data_anal)  # Convert new_result_data_anal to a DataFrame
Crude_result_df = new_result_data_anal_df #new_result_data_Blank_df   # Assign new_result_data_anal_df to Crude_result_df
Crude_result_df = Crude_result_df.astype(float)  # Convert Crude_result_df to float type
What_happen1 = Crude_result_df.insert(0,"Image_Name",image_name_df)  # Insert a new column "Image_Name" with image_name_df data
#Air_data_df.to_excel(r"C:\Users\ACER\OneDrive\Desktop\air_df_before.xlsx", sheet_name='Sheet2', index=False)
Crude_result_df.to_excel(OUTPUT_L, sheet_name='Sheet1', index=False)  # Save Crude_result_df to an Excel file
DATA_FOR_not_done = Crude_result_df  # Assign Crude_result_df to DATA_FOR_not_done
DATA_FOR_not_done.columns = image_datafull_anal.columns.values  # Update column names of DATA_FOR_not_done
DATA_FOR_not_done = DATA_FOR_not_done.drop(DATA_FOR_not_done.columns[0], axis = 1)  # Drop the first column from DATA_FOR_not_done
DATA_FOR_not_done = pd.concat([Image_name_data_df_full, DATA_FOR_not_done], axis = 1)  # Concatenate Image_name_data_df_full with DATA_FOR_not_done

DATA_with_Solute = DATA_FOR_not_done[DATA_FOR_not_done['Solute_1'] != "Blank"]  # Filter rows where 'Solute_1' is not "Blank"
DATA_BLANK = DATA_FOR_not_done[DATA_FOR_not_done.Solute_1 == "Blank"]  # Filter rows where 'Solute_1' is "Blank"
DATA_BLANK = DATA_BLANK.drop(columns = ["Picture_Number","Concentration_1_ppm","Solute_2","Concentration_2_ppm"])  # Drop specified columns from DATA_BLANK
grouped_data = DATA_BLANK.groupby(['Light_Source', 'Solute_1', 'Solvent',])  # Group DATA_BLANK by specified columns
DATA_BLANK_grouped_average = grouped_data.median().assign(median=lambda x: x.median()).drop(columns = ["median"])  # Calculate median for each group and assign a 'median' column
DATA_BLANK_grouped_average = DATA_BLANK_grouped_average.reset_index()

data_blank_path = os.path.join(save_mape, 'DATA_BLANK_grouped_average.pkl')
with open(data_blank_path, 'wb') as f:
    pickle.dump(DATA_BLANK_grouped_average, f)

DATA_FOR_TRAIN = DATA_with_Solute.copy()  # Create a copy of DATA_with_Solute and assign it to DATA_FOR_TRAIN
for index, row in DATA_BLANK_grouped_average.iterrows():  # Iterate over rows in DATA_BLANK_grouped_average
    mask = (DATA_with_Solute['Light_Source'] == row['Light_Source']) & \
           (DATA_with_Solute['Solvent'] == row['Solvent'])  # Create a mask based on 'Light_Source' and 'Solvent'
    for col in DATA_with_Solute.filter(like='pixel').columns:  # Iterate over columns containing 'pixel'
        DATA_FOR_TRAIN.loc[mask, col] = row[col] - DATA_with_Solute.loc[mask, col]  # Perform operations on DATA_FOR_TRAIN

data_train_path = os.path.join(save_mape, 'DATA_FOR_TRAIN.xlsx')      
DATA_FOR_TRAIN.to_excel(data_train_path, sheet_name='1', index=False)

substance_dfs = create_substance_dfs(DATA_FOR_TRAIN)  # Call the create_substance_dfs function
# plot_trend(DATA_FOR_not_done)  # Call the plot_trend function
columns_to_drop = ['Concentration_1_ppm', 'Concentration_2_ppm', 'Solute_1', 'Solute_2', 'Solvent', 'Light_Source', 'Picture_Number']
Air_data_df = Air_data_df.drop(columns = "Image_Name")
New_Air_data_df = New_Air_data_df.drop(columns = columns_to_drop)
# plot_trend_air(Air_data_df)
# plot_trend_air(New_Air_data_df)
# plot_trend_diff_L(DATA_FOR_TRAIN)  # Call the plot_trend_diff_L function
best_model_pure,best_model_mixture,dict_combined, metric_combined = pure_mixture_seperation(substance_dfs)  # Call the pure_mixture_seperation function
# plot_data(dict_combined)  # Call the plot_data function

end_time = time.time()  # Record the end time
execution_time = end_time - start_time  # Calculate the execution time
# print("Run time: {:.2f} seconds".format(execution_time))  # Print the execution time