import pandas as pd
import random

# Define the attributes of the log data
LOG_ATTRIBUTES = [
    "Timestamp", "Log Level", "Event ID", "User ID", "Session ID", 
    "Source IP Address", "Destination IP Address", "Host Name", 
    "Application Name", "Process ID", "Thread ID", "File Name", 
    "Line Number", "Method Name", "Event Type", "Action Performed", 
    "Status Code", "Response Time", "Resource Accessed", "Bytes Sent", 
    "Bytes Received", "Error Message", "Exception StackTrace", 
    "User Agent", "Operating System"
]

# Example values for generating random queries
VALUES = {
    "Log Level": ["INFO", "ERROR", "DEBUG", "WARN"],
    "Event ID": list(range(1000, 1100)),
    "User ID": list(range(1, 21)),
    "Status Code": [200, 400, 404, 500],
    "Response Time": list(range(100, 1001)),
    "Source IP Address": [f"192.168.0.{i}" for i in range(1, 51)],
    "Destination IP Address": [f"10.0.0.{i}" for i in range(1, 51)],
    "Action Performed": ["LOGIN", "LOGOUT", "DOWNLOAD", "UPLOAD"],
    "Timestamp": ["2023-01-01", "2023-01-02", "2023-02-01", "2023-03-01"],
}

# Generate random natural language queries and SQL queries
def generate_queries(num_samples=100):
    nl_queries = []
    sql_queries = []

    for _ in range(num_samples):
        attribute = random.choice(LOG_ATTRIBUTES)
        value = random.choice(VALUES.get(attribute, ["VALUE_PLACEHOLDER"]))
        
        if attribute == "Timestamp":
            nl_query = f"Get logs from {value}"
            sql_query = f"SELECT * FROM logs WHERE {attribute.lower()} = '{value}'"
        elif attribute == "Response Time":
            operator = random.choice([">", "<", ">=", "<="])
            nl_query = f"Find logs where {attribute.lower()} is {operator} {value}ms"
            sql_query = f"SELECT * FROM logs WHERE {attribute.lower()} {operator} {value}"
        elif attribute == "Status Code":
            nl_query = f"Retrieve logs with status code {value}"
            sql_query = f"SELECT * FROM logs WHERE {attribute.lower().replace(' ', '_')} = {value}"
        elif attribute == "Action Performed":
            nl_query = f"List all logs where action performed is {value}"
            sql_query = f"SELECT * FROM logs WHERE {attribute.lower().replace(' ', '_')} = '{value}'"
        else:
            nl_query = f"Show logs where {attribute.lower()} is '{value}'"
            sql_query = f"SELECT * FROM logs WHERE {attribute.lower().replace(' ', '_')} = '{value}'"

        nl_queries.append(nl_query)
        sql_queries.append(sql_query)

    return pd.DataFrame({"nl_query": nl_queries, "sql_query": sql_queries})

# Generate and save the dataset
dataset = generate_queries(500)  # Adjust the number of samples as needed
dataset.to_csv("nl_to_sql_log_dataset.csv", index=False)

print("Dataset generated and saved to 'nl_to_sql_log_dataset.csv'.")
