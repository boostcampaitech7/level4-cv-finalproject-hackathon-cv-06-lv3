import shutil
import argparse

from time import time

import numpy as np
# import fireducks.pandas as pd
import fireducks.pandas as pd
from django.core.files.base import ContentFile

from rest_framework.views import APIView
from rest_framework.response import Response

from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from data_processing.models import FlowModel, ConcatColumnModel, SurrogateMatricModel, SurrogateResultModel, SearchResultModel, FeatureImportanceModel, OptimizationModel

from hackathon.src.dynamic_pipeline import preprocess_dynamic
from hackathon import surrogate_model, search_model


def flow_progress(flow, progress):
    '''
    flow의 progress 업데이트
    '''
    flow.progress = progress
    flow.save()

def is_number(s):
    try:
        float(s)
        if '_' in s:
            return False
        return True
    except ValueError:
        return False
    
def calculate_change_rate(ground_truth, predicted):
    try:
        sum_gt = np.sum(ground_truth)
        sum_pred = np.sum(predicted)
        if sum_gt == 0:
            for gt, pred in zip(ground_truth, predicted):
                print(f'{gt = }, {pred = }')
    except:
        return 0

    return round((sum_pred - sum_gt) / sum_gt * 100, 2)
    

def update_model_instances(flow, model_cls, df, column_field, defaults_mapping):
    model_cls.objects.filter(flow=flow).delete()
    for _, row in df.iterrows():
        column_instance = ConcatColumnModel.objects.get(
            flow=flow, column_name=row[column_field])
        defaults = {field: row[source_field]
                    for field, source_field in defaults_mapping.items()}

        model_cls.objects.create(
            flow=flow,
            column=column_instance,
            **defaults
        )


class ProcessingView(APIView):
    '''
    concat된 csv 파일의 전처리 수행
    '''
    @swagger_auto_schema(
        operation_description="전처리 및 surrogate model, search model 학습 수행",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'flow_id': openapi.Schema(
                    type=openapi.TYPE_INTEGER,
                    description="ID of the Concated csv file",
                ),
            },
        ),
        responses={
            200: openapi.Response(description="Preprocessing completed successfully"),
            400: openapi.Response(description="Invalid flow ID"),
            404: openapi.Response(description="File not found"),
        },
    )
    def post(self, request, *args, **kwargs):
        """
        Preprocesses the concatenated CSV file, trains surrogate models,
        and updates/creates related database entries.
        """
        # Validate and retrieve the flow instance.
        flow_id = request.data.get("flow_id")
        if not flow_id:
            return Response({"error": "No flow_id provided"}, status=400)

        try:
            flow = FlowModel.objects.get(id=flow_id)
        except FlowModel.DoesNotExist:
            return Response({"error": "File not found"}, status=404)

        # Retrieve column information.
        concat_columns = ConcatColumnModel.objects.filter(flow=flow)
        cat_cols = concat_columns.filter(
            column_type='categorical').values_list('column_name', flat=True)
        num_cols = concat_columns.filter(
            column_type='numerical').values_list('column_name', flat=True)
        text_cols = concat_columns.filter(
            column_type='text').values_list('column_name', flat=True)

        # Read the concatenated CSV and perform preprocessing.
        concat_df = pd.read_csv(flow.concat_csv)
        concat_df.drop(columns=ConcatColumnModel.objects.filter(flow=flow, column_type='unavailable').values_list('column_name', flat=True), inplace=True)
        flow_progress(flow, 1)
        preprocessed_df, df_scaled, dtype_info, scaler_info = preprocess_dynamic(concat_df)

        # Save the preprocessed CSV.
        preprocessed_filename = f'{flow.flow_name}_preprocessed.csv'
        flow.preprocessed_csv.save(
            preprocessed_filename, ContentFile(preprocessed_df.to_csv(index=False)))
        flow_progress(flow, 2)

        # Retrieve output columns for surrogate modeling.
        output_columns = ConcatColumnModel.objects.filter(
            flow=flow, property_type='output'
        ).values_list('column_name', flat=True)

        flow_progress(flow, 3)

        # Define common arguments for surrogate model training.
        common_args = {
            "target": output_columns,
            "data_path": flow.preprocessed_csv.path,
            "flow_id": flow_id,
            "seed": 40
        }

        # Train the CatBoost-based surrogate model.
        catboost_args = argparse.Namespace(**common_args, model='catboost')
        df_rank_cat, df_eval_cat, df_importance, model_path_cat = surrogate_model.main(
            catboost_args, scaler_info)

        # Train the TabPFN-based surrogate model.
        tabpfn_args = argparse.Namespace(**common_args, model='tabpfn')
        df_rank_tab, df_eval_tab, model_path_tab = surrogate_model.main(
            tabpfn_args, scaler_info)
        flow_progress(flow, 4)

        # Choose the model with the higher average r_squared.
        if df_eval_cat['r2'].mean() > df_eval_tab['r2'].mean():
            surrogate_model_name = 'catboost'
            df_rank, df_eval, model_path = df_rank_cat, df_eval_cat, model_path_cat
        else:
            surrogate_model_name = 'tabpfn'
            df_rank, df_eval, model_path = df_rank_tab, df_eval_tab, model_path_tab

        # Save the chosen model file.
        model_filename = model_path.split('/')[-1]
        flow.model.save(model_filename, ContentFile(open(model_path, 'rb').read()))

        # Clean up temporary files.
        shutil.rmtree('./temp')
        print(f'{df_rank = }')
        print(f'{df_eval = }')
        print(f'{df_importance = }')
    
        # Update or create SurrogateResultModel instances.
        update_model_instances(flow, SurrogateResultModel, df_rank, 'column_name', {
                               'ground_truth': 'y_test', 'predicted': 'y_pred', 'rank': 'rank'})

        # Update or create SurrogateMatricModel instances.
        update_model_instances(flow, SurrogateMatricModel, df_eval, 'target', {
                               'rmse': 'rmse', 'r_squared': 'r2', 'mae': 'mae'})

        # Update or create FeatureImportanceModel instances.
        update_model_instances(flow, FeatureImportanceModel, df_importance, 'feature', {
                               'importance': 'importance'})

        flow_progress(flow, 5)

        controllable_columns = list(ConcatColumnModel.objects.filter(
            flow=flow, property_type='controllable').values_list('column_name', flat=True))
        
        optimize = {}
        importance_column = {}
        controllable_columns_range = {}

        for column in controllable_columns:

            optimization = OptimizationModel.objects.get(
                column__flow=flow, column__column_name=column)
            
            if optimization.optimize_goal == 1:
                controllable_columns_range[column] = (preprocessed_df[column].min(), preprocessed_df[column].max())
                continue
            elif optimization.optimize_goal == 2:
                optimize[column] = 'maximize'
                importance_column[column] = optimization.optimize_order
            elif optimization.optimize_goal == 3:
                optimize[column] = 'minimize'
                importance_column[column] = optimization.optimize_order

            if is_number(optimization.minimum_value) and is_number(optimization.maximum_value):
                controllable_columns_range[column] = (float(optimization.minimum_value), float(optimization.maximum_value))
            else:
                controllable_columns_range[column] = (optimization.minimum_value, optimization.maximum_value)
            
            
        target_column = list(ConcatColumnModel.objects.filter(
            flow=flow, property_type='output').values_list('column_name', flat=True))

        user_request_target = [OptimizationModel.objects.get(
            column__flow=flow, column__column_name=target).maximum_value for target in target_column]
        
        # print(f'control_name: {controllable_columns}')
        # print(f'control_range: {controllable_columns_range}')
        # print(f'target: {target_column}')
        # print(f'importance: {importance_column}')
        # print(f'optimize: {optimize}')
        # print(f'user_request_target: {user_request_target}')

        if surrogate_model_name == 'catboost':
            model_path = flow.model.path.removesuffix('.cbm')
        elif surrogate_model_name == 'tabpfn':
            model_path = flow.model.path.removesuffix('.pkl')


        search_args = argparse.Namespace(
            model=surrogate_model_name,
            search_model='k_means',
            data_path=flow.preprocessed_csv.path,
            control_name=controllable_columns,
            control_range=controllable_columns_range,
            target=target_column,
            importance=importance_column,
            optimize=optimize,
            flow_id=flow_id,
            seed=40,
            user_request_target=user_request_target,
            model_path=model_path
        )

        x_opt = search_model.main(search_args, scaler_info)
        # print(x_opt)
        x_opt['average_change_rate'] = x_opt.apply(lambda row: calculate_change_rate(row['ground_truth'], row['predicted']), axis=1)

        update_model_instances(flow, SearchResultModel, x_opt, 'column_name', {
                               'ground_truth': 'ground_truth', 'predicted': 'predicted', 'average_change_rate': 'average_change_rate'})
        flow_progress(flow, 6)

        return Response({"message": "Processing completed successfully"}, status=200)
