FROM python:3.9

# Create a new user and set it up
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code into the container
COPY --chown=user . /app

# Set the command to run Streamlit when the container starts
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860"]
