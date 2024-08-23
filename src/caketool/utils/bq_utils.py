from google.cloud import bigquery

ONE_GIGABYTE = (1024 ** 3)
ONE_TERABYTE = (1024 * ONE_GIGABYTE)
DEFAULT_BQ_CLI = bigquery.Client

def safety_query(
        query_statement: str, 
        client: bigquery.Client = DEFAULT_BQ_CLI, 
        gb_limit=50, 
        price_for_one_terabyte = 8.44
    ) -> bigquery.QueryJob:
    """
    Query data from Big Query. Check the cost (Gb and $) before executing the query.
    If the data size exceeds the limitation, raise error.
    """

    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=True)
    query_job = client.query(query_statement, job_config=job_config)
    bytes_processed = query_job.total_bytes_processed
    gigabytes_processed = bytes_processed / ONE_GIGABYTE
    print(f"This query will process {bytes_processed / ONE_GIGABYTE} Gigabytes.")
    print(f"This query will process {bytes_processed / ONE_TERABYTE * price_for_one_terabyte} dollars.")
    if gigabytes_processed >= gb_limit:
        raise ValueError(f"The data size exceeds the limitation >= {bytes_processed} GB")
    query_job = client.query((query_statement))
    return query_job
 