import os
import psycopg2
import pandas as pd

DATABASE_URL=os.getenv('roach_url2')

try:
  connection = psycopg2.connect(DATABASE_URL)

  with connection.cursor() as cur:
      cur.execute("SELECT now()")
      res = cur.fetchall()
      connection.commit()
      print(res)

  # Create a cursor object
  mycursor = connection.cursor()
  print("DB Connection Successful")  
  
  mycursor.close()
  connection.close()

except Exception as error:
    print("Error while connecting to CockroachDB", error)


def insert_follow_up_actions(follow_up_tables):
    """
    Inserts records from follow_up_tables into the follow_up_actions table in the PostgreSQL database.

    Args:
    follow_up_tables (dict): A dictionary where keys are categories and values are pandas DataFrames
                             with columns ["Call ID", "CSR ID", "Category", "Action"].
    db_params (dict): A dictionary with the database connection parameters: dbname, user, password, host, port.
    """
    try:
        # Connect to the PostgreSQL database
        connection = psycopg2.connect(DATABASE_URL)
        cur = connection.cursor()

        # Iterate over the follow_up_tables and insert data into the PostgreSQL table
        for category, df in follow_up_tables.items():
            for index, row in df.iterrows():
                call_id = row["Call ID"]
                csr_id = row["CSR ID"]
                category = row["Category"]
                action = row["Action"]

                # Insert each row into the PostgreSQL table, UUID is auto-generated
                cur.execute("""
                    INSERT INTO follow_up_actions (call_id, csr_id, category, action)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (call_id, category, csr_id)
                    DO UPDATE SET 
                        action = EXCLUDED.action
                """, (call_id, csr_id, category, action))

        # Commit the transaction
        connection.commit()
        print("Data inserted to follow_up_actions successfully")

    except Exception as e:
        print(f"An error occurred: {e}")
        connection.rollback()  # Rollback in case of error

    finally:
        # Close the connection
        if cur:
            cur.close()
        if connection:
            connection.close()

def insert_assessment_data(table_data):
    """
    Inserts the assessment data into the call_assessment_quality_scores table in the PostgreSQL database.
    
    Args:
    table_data (pd.DataFrame): The DataFrame containing assessment data.
    db_params (dict): A dictionary with the database connection parameters: dbname, user, password, host, port.
    """
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        # Iterate over the DataFrame and insert data into the PostgreSQL table
        for index, row in table_data.iterrows():
            call_id = row["Call ID"]
            csr_id = row["CSR ID"]
            category = row["Category"]
            score = row["Score"]
            explanation = row["Explanation"]

            # Insert each row into the PostgreSQL table, UUID is auto-generated
            cur.execute("""
                INSERT INTO call_assessment_quality_scores (call_id, csr_id, category, score, explanation)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (call_id, category, csr_id)
                DO UPDATE SET 
                    score = EXCLUDED.score,
                    explanation = EXCLUDED.explanation
            """, (call_id, csr_id, category, score, explanation))

        # Commit the transaction
        conn.commit()
        print("Data inserted to call_assessment_quality_scores successfully")

    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()  # Rollback in case of error

    finally:
        # Close the connection
        if cur:
            cur.close()
        if conn:
            conn.close()

import psycopg2
from datetime import datetime

def insert_call_data(call_scenario, audio_file):
    # Establish connection to the PostgreSQL database
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Prepare the current date and time
    call_date = datetime.now().date()
    call_time = datetime.now().time()

    # Status is set to 'new' by default when a call record is created
    status = 'new'

    # Prepare the SQL statement for inserting data
    sql = """
    INSERT INTO customer_calls (call_ID, CSR_ID, call_date, call_time, call_transcript_audio_file, status)
    VALUES (%s, %s, %s, %s, %s, %s)
    """

    # Execute the SQL command, and provide the data
    try:
        cur.execute(sql, (call_scenario["call_ID"], call_scenario["CSR_ID"], call_date, call_time, audio_file, status))
        conn.commit()  # Commit the transaction
        print("Data inserted successfully")
    except Exception as e:
        print("An error occurred:", e)
        conn.rollback()  # Rollback the transaction on error
    finally:
        cur.close()  # Close the cursor
        conn.close()  # Close the connection

