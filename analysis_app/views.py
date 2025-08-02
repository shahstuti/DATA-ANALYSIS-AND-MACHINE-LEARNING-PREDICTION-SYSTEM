import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import os
import uuid
import logging
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

class Dataset:
    def __init__(self, id, name, path):
        self.id = id
        self.name = name
        self.path = path

    @staticmethod
    def get_from_session(request, dataset_id=None):
        current_dataset_id = request.session.get('current_dataset_id')
        if dataset_id and str(dataset_id) != str(current_dataset_id):
            return None

        file_path = request.session.get('current_dataset_path')
        dataset_name = request.session.get('current_dataset_name')

        if file_path and os.path.exists(file_path):
            return Dataset(current_dataset_id, dataset_name, file_path)
        return None

    def get_dataframe(self):
        try:
            return pd.read_csv(self.path)
        except Exception as e:
            raise ValueError(f"Error loading dataframe from {self.path}: {e}")

def get_current_dataset_id(request):
    return request.session.get('current_dataset_id')

def load_dataset_from_session(request, dataset_id=None):
    dataset = Dataset.get_from_session(request, dataset_id)
    if dataset:
        try:
            df = dataset.get_dataframe()
            return dataset, df
        except ValueError as e:
            messages.error(request, str(e))
            return None, None
    messages.error(request, 'Dataset not found or session expired.')
    return None, None

def update_session_dataset(request, df, dataset_name="cleaned_dataset.csv"):
    try:
        unique_filename = f"cleaned_data_{uuid.uuid4().hex}.csv"
        save_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
        os.makedirs(save_dir, exist_ok=True)
        cleaned_file_path = os.path.join(save_dir, unique_filename)
        df.to_csv(cleaned_file_path, index=False)

        request.session['current_dataset_path'] = cleaned_file_path
        request.session['current_dataset_name'] = dataset_name
        request.session.modified = True
        return True
    except Exception as e:
        messages.error(request, f"Error saving cleaned data: {e}")
        return False

def upload_csv(request):
    if request.method == 'POST' and 'csv_file' in request.FILES:
        csv_file = request.FILES['csv_file']
        dataset_name = request.POST.get('dataset_name', csv_file.name)
        dataset_id = str(uuid.uuid4())
        save_dir = os.path.join(settings.MEDIA_ROOT, 'datasets')
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{dataset_id}_{csv_file.name}")
        
        try:
            with open(file_path, 'wb+') as destination:
                for chunk in csv_file.chunks():
                    destination.write(chunk)
            
            request.session['current_dataset_id'] = dataset_id
            request.session['current_dataset_path'] = file_path
            request.session['current_dataset_name'] = dataset_name
            request.session.modified = True
            
            messages.success(request, 'File uploaded successfully!')
            return redirect('analysis_app:analysis', dataset_id=dataset_id)
        except Exception as e:
            messages.error(request, f'Error saving file: {str(e)}')
    
    return render(request, 'upload.html')

def analysis(request, dataset_id=None):
    dataset, df = load_dataset_from_session(request, dataset_id)
    if df is None or df.empty:
        return redirect('analysis_app:upload_csv')

    try:
        total_cells = df.shape[0] * df.shape[1]
        columns_info = []
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'missing': df[col].isnull().sum(),
                'sample_values': df.head()[col].tolist()
            }
            columns_info.append(col_info)
        
        context = {
            'dataset': dataset,
            'shape': df.shape,
            'head': df.head().to_dict('records'),
            'total_cells': total_cells,
            'columns_info': columns_info,
            'columns': df.columns.tolist()
        }
        return render(request, 'analysis.html', context)
    except Exception as e:
        logger.error(f"Error analyzing data: {str(e)}")
        messages.error(request, f'Error analyzing data: {str(e)}')
        return redirect('analysis_app:upload_csv')

def prediction(request, dataset_id=None):
    dataset = Dataset.get_from_session(request, dataset_id)
    if not dataset:
        return redirect('analysis_app:upload_csv')
    
    request.session['current_dataset_id'] = str(dataset.id)
    request.session.modified = True

    try:
        df = dataset.get_dataframe()
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

        regression_algorithms = [
            'LinearRegression',
            'RandomForestRegressor',
            'DecisionTreeRegressor',
            'SVR',
            'KNeighborsRegressor'
        ]

        classification_algorithms = [
            'LogisticRegression',
            'RandomForestClassifier',
            'DecisionTreeClassifier',
            'SVC',
            'KNeighborsClassifier'
        ]

        context = {
            'dataset': dataset,
            'all_columns': df.columns.tolist(),
            'numeric_columns': numeric_columns,
            'regression_algorithms': json.dumps(regression_algorithms),
            'classification_algorithms': json.dumps(classification_algorithms),
        }
        return render(request, 'prediction.html', context)
    except Exception as e:
        messages.error(request, str(e))
        return redirect('analysis_app:upload_csv')

def run_prediction(request, dataset_id):
    logger.info("Starting prediction request")
    
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST method is allowed.'})
    
    if not dataset_id:
        return JsonResponse({'success': False, 'error': 'No dataset id provided.'})
    
    dataset, df = load_dataset_from_session(request, dataset_id)
    if df is None or df.empty:
        return JsonResponse({'success': False, 'error': 'Dataset not found or empty.'})
    
    try:
        features = request.POST.getlist('x_axis')
        target = request.POST.get('y_axis')
        algorithm_type = request.POST.get('algorithm_type')
        algorithm_name = request.POST.get('algorithm')
        
        logger.info(f"Prediction params: features={features}, target={target}, "
                   f"type={algorithm_type}, algorithm={algorithm_name}")
        
        if not all([features, target, algorithm_type, algorithm_name]):
            return JsonResponse({'success': False, 'error': 'Missing required parameters.'})
        
        df_model = df[features + [target]].dropna()
        if df_model.empty:
            return JsonResponse({'success': False, 'error': 'No valid data after cleaning.'})
        
        X = df_model[features]
        y = df_model[target]
        
        is_classification = (algorithm_type == 'classification')
        if is_classification and (y.dtype == 'object' or y.dtype == 'bool'):
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = None
        if algorithm_name == 'LinearRegression':
            model = LinearRegression()
        elif algorithm_name == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000)
        elif algorithm_name == 'RandomForestRegressor':
            model = RandomForestRegressor(random_state=42)
        elif algorithm_name == 'RandomForestClassifier':
            model = RandomForestClassifier(random_state=42)
        elif algorithm_name == 'SVR':
            model = SVR()
        elif algorithm_name == 'SVC':
            model = SVC()
        elif algorithm_name == 'DecisionTreeRegressor':
            model = DecisionTreeRegressor(random_state=42)
        elif algorithm_name == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(random_state=42)
        elif algorithm_name == 'KNeighborsRegressor':
            model = KNeighborsRegressor()
        elif algorithm_name == 'KNeighborsClassifier':
            model = KNeighborsClassifier()
        else:
            return JsonResponse({'success': False, 'error': 'Invalid algorithm selected.'})
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results = {
            'algorithm': algorithm_name,
            'features': features,
            'target': target,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'dataset_id': dataset.id,
            'dataset_name': dataset.name,
            'is_classification': is_classification
        }
        
        if is_classification:
            results['accuracy'] = float(accuracy_score(y_test, y_pred))
            results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        else:
            results['mse'] = float(mean_squared_error(y_test, y_pred))
            results['r2'] = float(r2_score(y_test, y_pred))
            results['rmse'] = float(np.sqrt(results['mse']))
        
        if 'prediction_results' not in request.session:
            request.session['prediction_results'] = []
        request.session['prediction_results'].append(results)
        request.session.modified = True
        
        logger.info("Prediction completed successfully")
        return JsonResponse({'success': True, 'results': results})
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return JsonResponse({'success': False, 'error': str(e)})

def prediction_redirect(request):
    dataset_id = get_current_dataset_id(request)
    if not dataset_id:
        messages.error(request, 'No active dataset found.')
        return redirect('analysis_app:upload_csv')
    
    dataset, df = load_dataset_from_session(request, dataset_id)
    if df is None:
        return redirect('analysis_app:upload_csv')
        
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    regression_algorithms = [
        'LinearRegression', 'RandomForestRegressor',
        'DecisionTreeRegressor', 'SVR', 'KNeighborsRegressor'
    ]
    
    classification_algorithms = [
        'LogisticRegression', 'RandomForestClassifier',
        'DecisionTreeClassifier', 'SVC', 'KNeighborsClassifier'
    ]
    
    context = {
        'dataset': dataset,
        'all_columns': df.columns.tolist(),
        'numeric_columns': numeric_columns,
        'regression_algorithms': json.dumps(regression_algorithms),
        'classification_algorithms': json.dumps(classification_algorithms),
    }
    
    return render(request, 'prediction.html', context)

def model_comparison(request, dataset_id=None):
    dataset = Dataset.get_from_session(request, dataset_id)
    if not dataset:
        return redirect('analysis_app:upload_csv')

    request.session['current_dataset_id'] = str(dataset.id)
    request.session.modified = True

    try:
        df = dataset.get_dataframe()
    except Exception as e:
        messages.error(request, str(e))
        return redirect('analysis_app:upload_csv')

    all_prediction_results = request.session.get('prediction_results', [])
    if not isinstance(all_prediction_results, list):
        all_prediction_results = []
        request.session['prediction_results'] = all_prediction_results
        request.session.modified = True

    prediction_results = [
        result for result in all_prediction_results 
        if str(result.get('dataset_id', '')) == str(dataset.id)
    ]

    best_classification = None
    best_regression = None
    for result in prediction_results:
        if result.get('is_classification'):
            if best_classification is None or result.get('accuracy', 0) > best_classification.get('accuracy', 0):
                best_classification = result
        else:
            if best_regression is None or result.get('r2', 0) > best_regression.get('r2', 0):
                best_regression = result

    for result in prediction_results:
        if result.get('is_classification') and 'accuracy' in result:
            result['accuracy_percent'] = round(result['accuracy'] * 100, 1)

    if request.method == 'POST' and request.content_type == 'application/json':
        data = json.loads(request.body)
        if data.get('action') == 'clear_results':
            request.session['prediction_results'] = [
                result for result in all_prediction_results 
                if str(result.get('dataset_id', '')) != str(dataset.id)
            ]
            request.session.modified = True
            logger.info(f"Cleared prediction results for dataset_id={dataset.id}")
            return JsonResponse({
                'success': True,
                'message': 'Prediction results cleared successfully.'
            })

    context = {
        'dataset': dataset,
        'prediction_results': prediction_results,
        'best_classification': best_classification,
        'best_regression': best_regression
    }
    
    return render(request, 'model_comparison.html', context)

def cleaning(request, dataset_id=None):
    dataset, df = load_dataset_from_session(request, dataset_id)
    if df is None or df.empty:
        return redirect('analysis_app:upload_csv')

    try:
        columns_info = {}
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique(),
                'sample_values': df[col].dropna().head(10).tolist()
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info['is_numeric'] = True
                col_info['mean'] = df[col].mean()
                col_info['median'] = df[col].median()
                col_info['mode'] = df[col].mode().iloc[0] if not df[col].mode().empty else None
            else:
                col_info['is_numeric'] = False
                col_info['mode'] = df[col].mode().iloc[0] if not df[col].mode().empty else None
            
            columns_info[col] = col_info
        
        context = {
            'dataset': dataset,
            'columns_info': columns_info,
            'columns': df.columns.tolist()
        }
        return render(request, 'cleaning.html', context)
    except Exception as e:
        logger.error(f"Error in data cleaning: {str(e)}")
        messages.error(request, f'Error during data cleaning: {str(e)}')
        return redirect('analysis_app:upload_csv')

def clean_data(request, dataset_id):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST method is allowed.'})
    
    request.session['current_dataset_id'] = str(dataset_id)
    request.session.modified = True
    
    dataset, df = load_dataset_from_session(request, dataset_id)
    if df is None or df.empty:
        return JsonResponse({'success': False, 'error': 'Dataset not found or empty.'})
    
    try:
        cleaning_operations = json.loads(request.POST.get('cleaning_operations', '{}'))
        column_renames = json.loads(request.POST.get('column_renames', '{}'))
        dtype_changes = json.loads(request.POST.get('dtype_changes', '{}'))

        if '_global' in cleaning_operations and 'remove_duplicates' in cleaning_operations['_global']:
            df = df.drop_duplicates()
            del cleaning_operations['_global']

        for column, operations in cleaning_operations.items():
            if column in df.columns:
                for operation in operations:
                    if operation == 'remove_null':
                        df = df.dropna(subset=[column])
                    elif operation == 'fill_mean' and pd.api.types.is_numeric_dtype(df[column]):
                        df[column].fillna(df[column].mean(), inplace=True)
                    elif operation == 'fill_median' and pd.api.types.is_numeric_dtype(df[column]):
                        df[column].fillna(df[column].median(), inplace=True)
                    elif operation == 'fill_mode':
                        mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else None
                        if mode_value is not None:
                            df[column].fillna(mode_value, inplace=True)
        
        if column_renames:
            df = df.rename(columns=column_renames)
        
        for column, new_dtype in dtype_changes.items():
            if column in df.columns:
                try:
                    if new_dtype == 'int64':
                        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                    elif new_dtype == 'float64':
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    elif new_dtype == 'datetime64':
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif new_dtype == 'object':
                        df[column] = df[column].astype(str)
                except Exception as e:
                    return JsonResponse({
                        'success': False,
                        'error': f'Error converting {column} to {new_dtype}: {str(e)}'
                    })

        if update_session_dataset(request, df, dataset.name + "_cleaned"):
            return JsonResponse({'success': True, 'new_shape': df.shape})
        else:
            return JsonResponse({'success': False, 'error': 'Error saving cleaned dataset'})

    except Exception as e:
        logger.error(f"Error in clean_data: {str(e)}")
        return JsonResponse({'success': False, 'error': str(e)})

def graphs(request, dataset_id=None):
    dataset, df = load_dataset_from_session(request, dataset_id)
    if df is None or df.empty:
        return redirect('analysis_app:upload_csv')

    try:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

        chart_types = ['line', 'scatter', 'bar', 'count', 'box', 'histogram', 'heatmap']

        context = {
            'dataset': dataset,
            'numeric_columns': numeric_columns,
            'categorical_columns': categorical_columns,
            'datetime_columns': datetime_columns,
            'all_columns': df.columns.tolist(),
            'chart_types': chart_types
        }
        return render(request, 'graphs.html', context)
    except Exception as e:
        logger.error(f"Error in graphs view: {str(e)}")
        messages.error(request, f'Error loading visualization options: {str(e)}')
        return redirect('analysis_app:upload_csv')

def generate_chart(request, dataset_id):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST method is allowed.'})

    request.session['current_dataset_id'] = str(dataset_id)
    request.session.modified = True
    
    dataset, df = load_dataset_from_session(request, dataset_id)
    if df is None or df.empty:
        return JsonResponse({'success': False, 'error': 'Dataset not found or empty.'})
    
    try:
        chart_type = request.POST.get('chart_type')
        x_axis = request.POST.get('x_axis')
        y_axis = request.POST.getlist('y_axis')
        color = request.POST.get('color', 'blue')
        title = request.POST.get('title', f'{chart_type.title()} Chart')
        
        if not all([chart_type, x_axis]) or (chart_type != 'count' and not y_axis):
            return JsonResponse({'success': False, 'error': 'Missing required parameters.'})

        plt.figure(figsize=(10, 6))

        if chart_type == 'line':
            for y in y_axis:
                plt.plot(df[x_axis], df[y], label=y, color=color)
            plt.legend()
        elif chart_type == 'scatter':
            for y in y_axis:
                plt.scatter(df[x_axis], df[y], alpha=0.5, label=y, color=color)
            plt.legend()
        elif chart_type == 'bar':
            data = df.groupby(x_axis)[y_axis].mean()
            data.plot(kind='bar', color=color)
        elif chart_type == 'count':
            df[x_axis].value_counts().plot(kind='bar', color=color)
        elif chart_type == 'box':
            df.boxplot(column=y_axis, by=x_axis, color=dict(boxes=color, whiskers=color, medians=color, caps=color))
        elif chart_type == 'histogram':
            for y in y_axis:
                plt.hist(df[y].dropna(), bins=30, color=color, alpha=0.7)
        elif chart_type == 'heatmap':
            corr = df.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        else:
            return JsonResponse({'success': False, 'error': 'Invalid chart type.'})

        plt.title(title)
        plt.xlabel(x_axis)
        if chart_type != 'count':
            plt.ylabel(', '.join(y_axis))
        plt.xticks(rotation=45)
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()

        chart = base64.b64encode(image_png).decode('utf-8')
        return JsonResponse({
            'success': True,
            'chart': chart
        })

    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        return JsonResponse({'success': False, 'error': str(e)})
