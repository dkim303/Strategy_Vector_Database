import argparse
import ollama
from sentence_transformers import SentenceTransformer
import logging

from vector_db_package.database_utils import (
    get_config,
    get_connection,
    setup_logging,
)

from vector_db_package.schema_utils import get_advisors

from vector_db_package.advising_utils import (
    run_advisor_query,
    get_advisor_desc,
)

from vector_db_package.ollama_utils import check_ollama_connection

def main(config_file_name: str):
    run_status = "Failed"
    conn = None
    cur = None

    try:
        print("Start of Query Program:")
        postgres_info, table_info, logging_info, sentence_transformer, ollama_info = get_config(config_file_name)

        log_file = setup_logging(logging_info)
        logging.info("Program Started")
        logging.info("Log file: %s", log_file)

        postgres_schema = postgres_info.get("schema")
        advisors_table = table_info.get("advisors")
        documents_table = table_info.get("documents")
        chunks_table = table_info.get("chunks")
        advisors_documents_table = table_info.get("advisor_documents")

        conn, cur = get_connection(postgres_info)    
        ST_model = SentenceTransformer(sentence_transformer.get("model"))
        ollama_model = ollama_info.get("model")

        K_input = input("Enter K value (Recommended 30): ").strip()
        if not K_input.isdigit():
            raise ValueError(f"Invalid K value selected: {K_input}")
        K = int(K_input)

        if K <= 0:
            raise ValueError("K must be greater than 0")
        
        logging.info(f"K value selected: {K}")

        print("Available Advisors: ")
        existing_advisors = get_advisors(cur, postgres_schema, advisors_table)
        if not existing_advisors:
            raise ValueError("No advisors found. Create an advisor before querying.")

        for advisor_id, advisor_name in existing_advisors:
            print(f"[{advisor_id}]: {advisor_name}")

        advisor_id_input = input("Select Advisor: ").strip()
        if not advisor_id_input.isdigit():
            raise ValueError(f"Invalid advisor selected: {advisor_id_input}")
        advisor_id = int(advisor_id_input)

        valid_advisor_ids = {int(row[0]) for row in existing_advisors}

        if advisor_id not in valid_advisor_ids:
            raise ValueError(f"Invalid advisor selected: {advisor_id}")

        advisor_desc = get_advisor_desc(cur,
                                        postgres_schema,
                                        advisors_table,
                                        advisor_id)
        logging.info("Selected Advisor ID: %s", advisor_id)

        query_text = input("Enter Prompt: ").strip()
        if not query_text:
            raise ValueError("Prompt cannot be empty")
        logging.info("User query length: %s characters", len(query_text))

        top_K_df = run_advisor_query(cur,
                            postgres_info,
                            table_info,
                            ST_model,
                            advisor_id,
                            query_text,
                            K)
        
        if top_K_df.empty:
            raise ValueError("No relevant chunks found for this advisor/query.")
        logging.info("Retrieved chunks: %s", len(top_K_df))
        
        context_text = "\n\n".join(
                f"""[Context {i+1}]
            Score: {row["score"]:.4f}
            Document ID: {row["document_id"]}
            Chunk ID: {row["chunk_id"]}
            Text:
            {row["chunk_text"]}"""
                for i, row in top_K_df.iterrows()
        )

        chat_message = [
                {"role": "system", "content": f"{advisor_desc}"},
                {"role": "user", "content": f"{query_text}"},
                {"role": "user", "content": f"{context_text}"}
            ]
        
        check_ollama_connection()
        response = ollama.chat(
            model=ollama_model,
            messages=chat_message
        )

        answer = response["message"]["content"]
        print(answer)

        logging.info("Ollama response length: %s characters", len(answer))
        run_status = "Success"

    except Exception as e:
        print(f"Error: {e}")
        logging.exception(f"Error: {e}")
        run_status = "Failed"

    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

        print(f"Terminating program with status {run_status}")
        logging.info("Terminated program with status: %s", run_status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Configuration File Path")
    args = parser.parse_args()

    main(args.config_file)