FROM public.ecr.aws/d3j8x8q7/olympus-base-python:latest

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir .[dev,all_extras]

CMD ["/bin/bash"]
