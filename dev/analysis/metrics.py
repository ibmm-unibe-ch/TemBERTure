def eval_metrics(avg_pred): 
    """
    Computes evaluation metrics (MSE, MAE, R-squared) for the complete dataset and subsets based on 'cls_label'.
    
    Args:
    - avg_pred: DataFrame with columns 'tm' (actual values), 'prediction', and 'cls_label'.
    
    Returns:
    - Dictionary containing evaluation metrics for the entire dataset ('All'), mesophilic ('Meso'), and thermophilic ('Thermo') subsets.
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    
    ####################################
    ###### metrics for cumulative ######
    ####################################
    all_evaluation_results = {}
    mse = mean_squared_error(avg_pred['tm'], avg_pred['prediction'])
    r2 = r2_score(avg_pred['tm'], avg_pred['prediction'])
    mae = mean_absolute_error(avg_pred['tm'], avg_pred['prediction'])

    evaluation_results_cls = {}
    evaluation_results_cls['Mean Squared Error'] = mse
    evaluation_results_cls['Mean Absolute Error'] = mae
    evaluation_results_cls['R-squared Score'] = r2

    # Adding the dictionary for cls_label = 0 to the main dictionary
    all_evaluation_results['All'] = evaluation_results_cls

    print(all_evaluation_results)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")


    ####################################
    ###### metrics for mesophilic ######
    ####################################
    filtered_data = avg_pred[avg_pred['cls_label'] == 0]

    y_filtered = filtered_data['tm']
    predictions_filtered = filtered_data['prediction']

    mse_filtered = mean_squared_error(y_filtered, predictions_filtered)
    mae_filtered = mean_absolute_error(y_filtered, predictions_filtered)
    r2_filtered = r2_score(y_filtered, predictions_filtered)

    evaluation_results_cls_0 = {}
    evaluation_results_cls_0['Mean Squared Error'] = mse_filtered
    evaluation_results_cls_0['Mean Absolute Error'] = mae_filtered
    evaluation_results_cls_0['R-squared Score'] = r2_filtered

    # Adding the dictionary for cls_label = 0 to the main dictionary
    all_evaluation_results['Non-thermo'] = evaluation_results_cls_0

    print(all_evaluation_results)

    print(f"Mean Squared Error (Filtered): {mse_filtered}")
    print(f"Mean Absolute Error (Filtered): {mae_filtered}")
    print(f"R-squared Score (Filtered): {r2_filtered}")

    ####################################
    ##### metrics for thermophilic #####
    ####################################
    filtered_data = avg_pred[avg_pred['cls_label'] == 1]

    y_filtered = filtered_data['tm']
    predictions_filtered = filtered_data['prediction']

    mse_filtered = mean_squared_error(y_filtered, predictions_filtered)
    mae_filtered = mean_absolute_error(y_filtered, predictions_filtered)
    r2_filtered = r2_score(y_filtered, predictions_filtered)

    evaluation_results_cls_1 = {}
    evaluation_results_cls_1['Mean Squared Error'] = mse_filtered
    evaluation_results_cls_1['Mean Absolute Error'] = mae_filtered
    evaluation_results_cls_1['R-squared Score'] = r2_filtered

    # Adding the dictionary for cls_label = 0 to the main dictionary
    all_evaluation_results['Thermo'] = evaluation_results_cls_1

    print(f"Mean Squared Error (Filtered): {mse_filtered}")
    print(f"Mean Absolute Error (Filtered): {mae_filtered}")
    print(f"R-squared Score (Filtered): {r2_filtered}")
    
    print(all_evaluation_results)
    
    return all_evaluation_results


def eval_metrics_meso(avg_pred): 
    """
    Computes evaluation metrics (MSE, MAE, R-squared) for the complete dataset and subsets based on 'cls_label'.
    
    Args:
    - avg_pred: DataFrame with columns 'tm' (actual values), 'prediction', and 'cls_label'.
    
    Returns:
    - Dictionary containing evaluation metrics for the entire dataset ('All'), mesophilic ('Meso'), and thermophilic ('Thermo') subsets.
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    
    ####################################
    ###### metrics for cumulative ######
    ####################################
    all_evaluation_results = {}
    mse = mean_squared_error(avg_pred['tm'], avg_pred['prediction'])
    r2 = r2_score(avg_pred['tm'], avg_pred['prediction'])
    mae = mean_absolute_error(avg_pred['tm'], avg_pred['prediction'])

    evaluation_results_cls = {}
    evaluation_results_cls['Mean Squared Error'] = mse
    evaluation_results_cls['Mean Absolute Error'] = mae
    evaluation_results_cls['R-squared Score'] = r2

    # Adding the dictionary for cls_label = 0 to the main dictionary
    all_evaluation_results['Cumulative'] = evaluation_results_cls

    print(all_evaluation_results)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")


    ####################################
    ###### metrics for mesophilic ######
    ####################################
    filtered_data = avg_pred[avg_pred['cls_label'] == 0]

    y_filtered = filtered_data['tm']
    predictions_filtered = filtered_data['prediction']

    mse_filtered = mean_squared_error(y_filtered, predictions_filtered)
    mae_filtered = mean_absolute_error(y_filtered, predictions_filtered)
    r2_filtered = r2_score(y_filtered, predictions_filtered)

    evaluation_results_cls_0 = {}
    evaluation_results_cls_0['Mean Squared Error'] = mse_filtered
    evaluation_results_cls_0['Mean Absolute Error'] = mae_filtered
    evaluation_results_cls_0['R-squared Score'] = r2_filtered

    # Adding the dictionary for cls_label = 0 to the main dictionary
    all_evaluation_results['Non-thermo'] = evaluation_results_cls_0

    print(all_evaluation_results)

    print(f"Mean Squared Error (Filtered): {mse_filtered}")
    print(f"Mean Absolute Error (Filtered): {mae_filtered}")
    print(f"R-squared Score (Filtered): {r2_filtered}")

    
    return all_evaluation_results

def eval_metrics_thermo(avg_pred): 
    """
    Computes evaluation metrics (MSE, MAE, R-squared) for the complete dataset and subsets based on 'cls_label'.
    
    Args:
    - avg_pred: DataFrame with columns 'tm' (actual values), 'prediction', and 'cls_label'.
    
    Returns:
    - Dictionary containing evaluation metrics for the entire dataset ('All'), mesophilic ('Meso'), and thermophilic ('Thermo') subsets.
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    
    ####################################
    ###### metrics for cumulative ######
    ####################################
    all_evaluation_results = {}
    mse = mean_squared_error(avg_pred['tm'], avg_pred['prediction'])
    r2 = r2_score(avg_pred['tm'], avg_pred['prediction'])
    mae = mean_absolute_error(avg_pred['tm'], avg_pred['prediction'])

    evaluation_results_cls = {}
    evaluation_results_cls['Mean Squared Error'] = mse
    evaluation_results_cls['Mean Absolute Error'] = mae
    evaluation_results_cls['R-squared Score'] = r2

    # Adding the dictionary for cls_label = 0 to the main dictionary
    all_evaluation_results['All'] = evaluation_results_cls

    print(all_evaluation_results)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared Score: {r2}")

    ####################################
    ##### metrics for thermophilic #####
    ####################################
    filtered_data = avg_pred[avg_pred['cls_label'] == 1]

    y_filtered = filtered_data['tm']
    predictions_filtered = filtered_data['prediction']

    mse_filtered = mean_squared_error(y_filtered, predictions_filtered)
    mae_filtered = mean_absolute_error(y_filtered, predictions_filtered)
    r2_filtered = r2_score(y_filtered, predictions_filtered)

    evaluation_results_cls_1 = {}
    evaluation_results_cls_1['Mean Squared Error'] = mse_filtered
    evaluation_results_cls_1['Mean Absolute Error'] = mae_filtered
    evaluation_results_cls_1['R-squared Score'] = r2_filtered

    # Adding the dictionary for cls_label = 0 to the main dictionary
    all_evaluation_results['Thermo'] = evaluation_results_cls_1

    print(f"Mean Squared Error (Filtered): {mse_filtered}")
    print(f"Mean Absolute Error (Filtered): {mae_filtered}")
    print(f"R-squared Score (Filtered): {r2_filtered}")
    
    print(all_evaluation_results)
    
    return all_evaluation_results

