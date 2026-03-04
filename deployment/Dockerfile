FROM public.ecr.aws/lambda/python:3.11

COPY lambda_handler.py ${LAMBDA_TASK_ROOT}

RUN pip install --upgrade pip && pip install \
    scikit-learn==1.4.2 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scipy==1.11.4 \
    joblib==1.4.2 \
    boto3

CMD ["lambda_handler.lambda_handler"]