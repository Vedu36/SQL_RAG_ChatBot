# run_workflow.py
import json
from prepare_sql_context import prepare_sql_context
from answer_sql import answer_query

def main():
    user_query = "how many accountants do we have on the bench?"
    
    # Phase 1: generate context
    context = prepare_sql_context(user_query)
    with open("sql_context.json", "w") as f:
        json.dump(context, f, indent=2)
    
    # Phase 2: answer the query
    answer_query(user_query, context_path="sql_context.json")

if __name__ == "__main__":
    main()