FROM public.ecr.aws/lambda/python:3.12

COPY ./*.py ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir pandas

CMD ["get_cpu_family_lambda.handler"]