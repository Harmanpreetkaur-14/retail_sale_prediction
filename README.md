# retail_sale_prediction
ML project predicting customer purchase amounts from a 550 K‑row retail dataset; includes data cleaning, visual exploration, model training/evaluation (best: RandomForest), and exports predictions to CSV.
## Steps included

1. **Data loading & preprocessing**  
   - Read `data.csv`, handle missing values, encode categorical features.
2. **Exploratory analysis**  
   - Visualize distributions and relationships using Seaborn/Matplotlib.
3. **Modeling**  
   - Split data, train multiple regressors (Linear, DecisionTree, RandomForest, ExtraTrees).
   - Evaluate using MSE, R², MAPE; RandomForest performed best (~54.9 % R², 0.35 % MAPE).
4. **Output**  
   - Generate `submission.csv` with predicted purchase amounts.
  ## libraries used
   - pandas
   - numpy
   - seaborn
   - matplotlib
   - sklearn modules: preprocessing.LabelEncoder
        model_selection (cross_val_score, train_test_split)
        metrics (mean_squared_error, r2_score, mean_absolute_percentage_error)
        regressors (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, ExtraTreesRegressor)
